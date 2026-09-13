#!/usr/bin/env bash
set -euo pipefail

source scripts/python_env.sh

RECORDING="${1:-artifacts/showcase/who-farted-run.json}"
OUTPUT="${2:-artifacts/showcase/rapid-v4.mp4}"
E002C="${3:-results/e002/pfl3-convergence-v1.json}"
FC2="${4:-results/route/fc2-goal-interface-audit-v1.json}"

"$PYTHON_BIN" -m fly_sniff.recorded_showcase_v4 \
  "$RECORDING" \
  --e002c-report "$E002C" \
  --fc2-audit "$FC2" \
  --output "$OUTPUT"

if command -v ffprobe >/dev/null 2>&1; then
  dims="$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$OUTPUT")"
  [[ "$dims" == "1080x1350" ]] || { echo "unexpected video dimensions: $dims" >&2; exit 2; }
  echo "verified dimensions: $dims"
fi

echo "$OUTPUT"
