#!/bin/zsh

set -euo pipefail

repo_root=${0:A:h:h}
configuration=${1:-debug}
app_path=${STUDIO_APP_PATH:-"$repo_root/dist/Inference School Studio.app"}
app_path=${app_path:A}

if [[ "$configuration" != debug && "$configuration" != release ]]; then
    print -u2 "configuration must be debug or release"
    exit 64
fi
if [[ "$app_path" != *.app || "$app_path" == / || "$app_path" == "$HOME" || "$app_path" == "$repo_root" ]]; then
    print -u2 "STUDIO_APP_PATH must name a safe .app destination"
    exit 64
fi

cd "$repo_root"
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
for executable_path in \
    "$staging_path/Contents/MacOS/inference-school-studio" \
    "$staging_path/Contents/Helpers/inference-school-runner"
do
    if /usr/bin/strings -a "$executable_path" | grep -F -- "$repo_root" > /dev/null; then
        print -u2 "packaged executable contains the source checkout path: $executable_path"
        exit 1
    fi
done

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