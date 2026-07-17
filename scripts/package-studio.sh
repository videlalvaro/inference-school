#!/bin/zsh

set -euo pipefail

repo_root=${0:A:h:h}
configuration=${1:-debug}
app_path=${STUDIO_APP_PATH:-"$repo_root/dist/Inference School Studio.app"}
app_path=${app_path:A}
dist_root="$repo_root/dist"

if [[ "$configuration" != debug && "$configuration" != release ]]; then
    print -u2 "configuration must be debug or release"
    exit 64
fi
if [[ "$app_path" != "$dist_root/"*.app ]]; then
    print -u2 "STUDIO_APP_PATH must name an app inside $dist_root"
    exit 64
fi

cd "$repo_root"
swift package resolve

typeset -A locked_revisions
pin_index=0
while locked_identity=$(
    /usr/bin/plutil -extract "pins.$pin_index.identity" raw -o - Package.resolved \
        2> /dev/null
); do
    locked_revision=$(
        /usr/bin/plutil -extract "pins.$pin_index.state.revision" raw -o - Package.resolved
    )
    locked_identity=${locked_identity:l}
    if [[ -n "${locked_revisions[$locked_identity]-}" ]]; then
        print -u2 "duplicate dependency identity in Package.resolved: $locked_identity"
        exit 1
    fi
    locked_revisions[$locked_identity]=$locked_revision
    (( pin_index += 1 ))
done
if (( ${#locked_revisions} == 0 )); then
    print -u2 "Package.resolved contains no dependency pins"
    exit 1
fi

checkout_paths=("$repo_root"/.build/checkouts/*(N/))
if (( ${#checkout_paths} != ${#locked_revisions} )); then
    print -u2 "SwiftPM checkout count does not match Package.resolved"
    exit 1
fi
checkout_index_listing=$(mktemp "${TMPDIR:-/tmp}/inference-school-index.XXXXXX")
trap 'rm -f "$checkout_index_listing"' EXIT
for checkout_path in $checkout_paths; do
    checkout_identity=${checkout_path:t:l}
    locked_revision=${locked_revisions[$checkout_identity]-}
    if [[ -z "$locked_revision" ]]; then
        print -u2 "SwiftPM checkout is not present in Package.resolved: $checkout_identity"
        exit 1
    fi
    if ! checkout_revision=$(git -C "$checkout_path" rev-parse HEAD); then
        print -u2 "SwiftPM checkout is not a readable Git worktree: $checkout_identity"
        exit 1
    fi
    if [[ "$checkout_revision" != "$locked_revision" ]]; then
        print -u2 "SwiftPM checkout revision does not match Package.resolved: $checkout_identity"
        exit 1
    fi
    if ! git -C "$checkout_path" ls-files -v -z > "$checkout_index_listing"; then
        print -u2 "could not inspect SwiftPM checkout index: $checkout_identity"
        exit 1
    fi
    checkout_index_state=""
    while IFS= read -r -d '' tracked_entry; do
        if [[ "$tracked_entry" != "H "* ]]; then
            checkout_index_state=${tracked_entry[1]}
            break
        fi
    done < "$checkout_index_listing"
    if [[ -n "$checkout_index_state" ]]; then
        print -u2 "SwiftPM checkout contains nonstandard Git index state: $checkout_identity"
        exit 1
    fi
    if ! checkout_status=$(
        git -C "$checkout_path" status --porcelain --untracked-files=all --ignored
    ); then
        print -u2 "could not inspect SwiftPM checkout: $checkout_identity"
        exit 1
    fi
    if [[ -n "$checkout_status" ]]; then
        print -u2 "SwiftPM checkout contains uncommitted files: $checkout_identity"
        exit 1
    fi
    unset "locked_revisions[$checkout_identity]"
done
if (( ${#locked_revisions} != 0 )); then
    print -u2 "Package.resolved contains dependencies without SwiftPM checkouts"
    exit 1
fi
rm -f "$checkout_index_listing"
trap - EXIT

swift build -c "$configuration" --product inference-school-runner
swift build -c "$configuration" --product inference-school-studio
bin_path=$(swift build -c "$configuration" --show-bin-path)

resource_accessors=("$bin_path"/*.build/DerivedSources/resource_bundle_accessor.swift(N))
(( ${#resource_accessors} > 0 ))
for accessor_path in $resource_accessors; do
    sed -i '' \
        -e 's/Bundle\.main\.bundleURL\.appendingPathComponent/(Bundle.main.resourceURL ?? Bundle.main.bundleURL).appendingPathComponent/g' \
        -e 's#let buildPath = ".*/#let buildPath = "/__inference_school_build__/#' \
        "$accessor_path"
    touch "$accessor_path"
done
swift build -c "$configuration" --product inference-school-runner
swift build -c "$configuration" --product inference-school-studio

staging_path="${app_path:r}.staging.app"
rm -rf "$staging_path"
mkdir -p "${app_path:h}"
mkdir -p \
    "$staging_path/Contents/MacOS" \
    "$staging_path/Contents/Helpers" \
    "$staging_path/Contents/Resources/Course/Sources" \
    "$staging_path/Contents/Resources/Legal"

cp "$bin_path/inference-school-studio" "$staging_path/Contents/MacOS/inference-school-studio"
cp "$bin_path/inference-school-runner" "$staging_path/Contents/Helpers/inference-school-runner"
checkout_root=${repo_root:A}/.build/checkouts
checkout_source_prefix="$checkout_root/"
strings_output=$(mktemp "${TMPDIR:-/tmp}/inference-school-strings.XXXXXX")
trap 'rm -f "$strings_output"' EXIT
for executable_path in \
    "$staging_path/Contents/MacOS/inference-school-studio" \
    "$staging_path/Contents/Helpers/inference-school-runner"
do
    if ! /usr/bin/strings -a "$executable_path" > "$strings_output"; then
        print -u2 "could not inspect packaged executable: $executable_path"
        exit 1
    fi
    # Debug builds may retain dependency source locations. Reject only paths that
    # can make the packaged executables depend on this repository at runtime.
    leaked_checkout_paths=$(
        while IFS= read -r embedded_string; do
            [[ "$embedded_string" == *"$repo_root"* ]] || continue
            if [[ "$embedded_string" == "$checkout_source_prefix"* \
                && -f "$embedded_string" ]]
            then
                canonical_embedded_path=${embedded_string:A}
                if [[ "$canonical_embedded_path" == "$checkout_source_prefix"* ]]; then
                    continue
                fi
            fi
            print -r -- "$embedded_string"
        done < "$strings_output"
    )
    if [[ -n "$leaked_checkout_paths" ]]; then
        print -u2 "packaged executable contains the source checkout path: $executable_path"
        print -u2 "$leaked_checkout_paths"
        exit 1
    fi
done
rm -f "$strings_output"
trap - EXIT

required_bundles=(
    InferenceSchool_InferenceSchoolExercises.bundle
    InferenceSchool_InferenceSchoolSolutions.bundle
    InferenceSchool_InferenceSchoolStudio.bundle
    InferenceSchool_InferenceSchoolStudio.bundle/Resources/Diagram/index.html
    InferenceSchool_InferenceSchoolStudio.bundle/Resources/Diagram/diagram.js
    InferenceSchool_InferenceSchoolStudio.bundle/Resources/Diagram/diagram.css
    InferenceSchool_InferenceSchoolStudio.bundle/Resources/Editor/index.html
    InferenceSchool_InferenceSchoolStudio.bundle/Resources/Editor/editor.js
    InferenceSchool_InferenceSchoolStudio.bundle/Resources/Editor/editor.css
    swiftui-math_SwiftUIMath.bundle
    textual_Textual.bundle
)
for bundle_name in $required_bundles; do
    ditto \
        "$bin_path/$bundle_name" \
        "$staging_path/Contents/Resources/$bundle_name"
done

course_root="$staging_path/Contents/Resources/Course"
cp Package.swift "$course_root/Package.swift"
ditto Problems "$course_root/Problems"
course_source_directories=(
    InferenceSchoolCore
    InferenceSchoolExercises
    InferenceSchoolSolutions
    InferenceSchoolRunnerProtocol
    InferenceSchoolCLI
    InferenceSchoolCLIEntry
)
for directory_name in $course_source_directories; do
    ditto "Sources/$directory_name" "$course_root/Sources/$directory_name"
done

mkdir -p "$course_root/docs"
cp docs/MATH-PRIMER.md "$course_root/docs/MATH-PRIMER.md"
mkdir -p "$course_root/Tests"
ditto Tests/InferenceSchoolCoreTests "$course_root/Tests/InferenceSchoolCoreTests"

cp Packaging/Studio/Info.plist "$staging_path/Contents/Info.plist"
cp LICENSE THIRD_PARTY_NOTICES.md "$staging_path/Contents/Resources/Legal/"

plutil -lint "$staging_path/Contents/Info.plist"
plutil -lint Packaging/Studio/Studio.entitlements
plutil -lint Packaging/Studio/Runner.entitlements
[[ $(plutil -extract CFBundlePackageType raw "$staging_path/Contents/Info.plist") == "APPL" ]]
required_paths=(
    Contents/MacOS/inference-school-studio
    Contents/Helpers/inference-school-runner
    Contents/Resources/Course/Package.swift
    Contents/Resources/Course/Problems/000-start-here/README.md
    Contents/Resources/Course/Problems/001-vector-dot/README.md
    Contents/Resources/Course/docs/MATH-PRIMER.md
    Contents/Resources/Legal/LICENSE
    Contents/Resources/Legal/THIRD_PARTY_NOTICES.md
    Contents/Resources/InferenceSchool_InferenceSchoolExercises.bundle
    Contents/Resources/InferenceSchool_InferenceSchoolSolutions.bundle
    Contents/Resources/InferenceSchool_InferenceSchoolStudio.bundle
    Contents/Resources/swiftui-math_SwiftUIMath.bundle
    Contents/Resources/textual_Textual.bundle
)
for relative_path in $required_paths; do
    [[ -e "$staging_path/$relative_path" ]]
done

source_lessons=(Problems/*/README.md(N))
packaged_lessons=("$course_root"/Problems/*/README.md(N))
if (( ${#source_lessons} != ${#packaged_lessons} )); then
    print -u2 "packaged lesson count does not match the source course"
    exit 1
fi
for lesson_path in $source_lessons; do
    problem_directory=${lesson_path:h:t}
    while IFS= read -r relative_target; do
        extension=${relative_target:e}
        case "${extension:l}" in
            md|swift|metal) ;;
            *)
                print -u2 "unsupported lesson link target: $lesson_path -> $relative_target"
                exit 1
                ;;
        esac
        packaged_target="$course_root/Problems/$problem_directory/$relative_target"
        packaged_target=${packaged_target:A}
        if [[ "$packaged_target" != "$course_root"/* || ! -f "$packaged_target" ]]; then
            print -u2 "missing packaged lesson link target: $lesson_path -> $relative_target"
            exit 1
        fi
    done < <(
        grep -Eo '\]\(\.\./\.\./[^)#?]+\)' "$lesson_path" \
            | sed -E 's/^\]\((.*)\)$/\1/' \
            || true
    )
done

codesign \
    --force \
    --sign - \
    --identifier dev.inferenceschool.studio.runner \
    --entitlements Packaging/Studio/Runner.entitlements \
    --timestamp=none \
    "$staging_path/Contents/Helpers/inference-school-runner"
codesign \
    --force \
    --sign - \
    --entitlements Packaging/Studio/Studio.entitlements \
    --timestamp=none \
    "$staging_path"
codesign --verify --strict --verbose=2 "$staging_path"
"$repo_root/scripts/smoke-test-studio-diagram.sh" "$staging_path"

rm -rf "$app_path"
mv "$staging_path" "$app_path"
echo "$app_path"
