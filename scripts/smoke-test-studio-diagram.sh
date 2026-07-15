#!/bin/zsh

set -euo pipefail

app_path=${1:A}
result_file="$HOME/Library/Containers/dev.inferenceschool.studio/Data/tmp/inference-school-diagram-smoke-result.txt"

if [[ "$app_path" != *.app || ! -d "$app_path" ]]; then
    print -u2 "usage: $0 /path/to/Inference School Studio.app"
    exit 64
fi

rm -f "$result_file"
open -n -W "$app_path" --args --diagram-smoke-test

if [[ ! -f "$result_file" ]]; then
    print -u2 "DIAGRAM_SMOKE_FAIL signed app did not write a result"
    exit 1
fi

result=$(<"$result_file")
print -r -- "$result"
[[ "$result" == DIAGRAM_SMOKE_PASS\ * ]]