#!/bin/zsh

set -euo pipefail

repo_root=${0:A:h:h}
configuration=${1:-debug}
app_path=${STUDIO_APP_PATH:-"$repo_root/dist/LeetLLM Studio.app"}
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
swift build -c "$configuration" --product leetllm-runner
swift build -c "$configuration" --product leetllm-studio
bin_path=$(swift build -c "$configuration" --show-bin-path)

resource_accessors=("$bin_path"/*.build/DerivedSources/resource_bundle_accessor.swift(N))
(( ${#resource_accessors} > 0 ))
for accessor_path in $resource_accessors; do
    sed -i '' \
        's/Bundle\.main\.bundleURL\.appendingPathComponent/(Bundle.main.resourceURL ?? Bundle.main.bundleURL).appendingPathComponent/g' \
        "$accessor_path"
    touch "$accessor_path"
done
swift build -c "$configuration" --product leetllm-runner
swift build -c "$configuration" --product leetllm-studio

staging_path="${app_path:r}.staging.app"
rm -rf "$staging_path"
mkdir -p "${app_path:h}"
mkdir -p \
    "$staging_path/Contents/MacOS" \
    "$staging_path/Contents/Helpers" \
    "$staging_path/Contents/Resources/Course/Sources" \
    "$staging_path/Contents/Resources/Legal"

cp "$bin_path/leetllm-studio" "$staging_path/Contents/MacOS/leetllm-studio"
cp "$bin_path/leetllm-runner" "$staging_path/Contents/Helpers/leetllm-runner"

required_bundles=(
    LeetLLM_LeetLLMExercises.bundle
    LeetLLM_LeetLLMSolutions.bundle
    LeetLLM_LeetLLMStudio.bundle
    LeetLLM_LeetLLMStudio.bundle/Resources/Diagram/index.html
    LeetLLM_LeetLLMStudio.bundle/Resources/Diagram/diagram.js
    LeetLLM_LeetLLMStudio.bundle/Resources/Diagram/diagram.css
    LeetLLM_LeetLLMStudio.bundle/Resources/Editor/index.html
    LeetLLM_LeetLLMStudio.bundle/Resources/Editor/editor.js
    LeetLLM_LeetLLMStudio.bundle/Resources/Editor/editor.css
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
    LeetLLMCore
    LeetLLMExercises
    LeetLLMSolutions
    LeetRunnerProtocol
    LeetLLMCLI
    LeetLLMCLIEntry
)
for directory_name in $course_source_directories; do
    ditto "Sources/$directory_name" "$course_root/Sources/$directory_name"
done

mkdir -p "$course_root/docs"
cp docs/MATH-PRIMER.md "$course_root/docs/MATH-PRIMER.md"
mkdir -p "$course_root/Tests"
ditto Tests/LeetLLMCoreTests "$course_root/Tests/LeetLLMCoreTests"

cp Packaging/Studio/Info.plist "$staging_path/Contents/Info.plist"
cp LICENSE THIRD_PARTY_NOTICES.md "$staging_path/Contents/Resources/Legal/"

plutil -lint "$staging_path/Contents/Info.plist"
plutil -lint Packaging/Studio/Studio.entitlements
plutil -lint Packaging/Studio/Runner.entitlements
[[ $(plutil -extract CFBundlePackageType raw "$staging_path/Contents/Info.plist") == "APPL" ]]
required_paths=(
    Contents/MacOS/leetllm-studio
    Contents/Helpers/leetllm-runner
    Contents/Resources/Course/Package.swift
    Contents/Resources/Course/Problems/000-start-here/README.md
    Contents/Resources/Course/Problems/001-vector-dot/README.md
    Contents/Resources/Course/docs/MATH-PRIMER.md
    Contents/Resources/Legal/LICENSE
    Contents/Resources/Legal/THIRD_PARTY_NOTICES.md
    Contents/Resources/LeetLLM_LeetLLMExercises.bundle
    Contents/Resources/LeetLLM_LeetLLMSolutions.bundle
    Contents/Resources/LeetLLM_LeetLLMStudio.bundle
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
    --identifier dev.leetllm.studio.runner \
    --entitlements Packaging/Studio/Runner.entitlements \
    --timestamp=none \
    "$staging_path/Contents/Helpers/leetllm-runner"
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