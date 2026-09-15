#!/usr/bin/env bash
set -euo pipefail

RECORDING="${1:-artifacts/showcase/who-farted-run.json}"
OUTPUT="${2:-artifacts/showcase/who-farted-rapid-v3.mp4}"
E002C="${3:-results/e002/pfl3-convergence-v1.json}"
FC2="${4:-results/route/fc2-goal-interface-audit-v1.json}"

EXPECTED_E002C="46cf03f75dd6956722f1874877070f06243afc1cdae385cb459263422b14a0c2"
EXPECTED_FC2="49e416da2c11681892e62cfbe233495c5c46dc323069c9c2524e7a0ffbc5e860"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

observed_e002c="$(sha256_file "$E002C")"
observed_fc2="$(sha256_file "$FC2")"

if [[ "$observed_e002c" != "$EXPECTED_E002C" ]]; then
  echo "E002c artifact hash mismatch: $observed_e002c" >&2
  exit 3
fi
if [[ "$observed_fc2" != "$EXPECTED_FC2" ]]; then
  echo "FC2 artifact hash mismatch: $observed_fc2" >&2
  exit 4
fi

echo "verified sealed mechanism artifacts"

python -m fly_sniff.recorded_showcase_v3 \
  "$RECORDING" \
  --e002c-report "$E002C" \
  --fc2-audit "$FC2" \
  --output "$OUTPUT"

if command -v ffprobe >/dev/null 2>&1; then
  dims="$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$OUTPUT")"
  if [[ "$dims" != "1080x1350" ]]; then
    echo "unexpected video dimensions: $dims" >&2
    exit 2
  fi
  echo "verified dimensions: $dims"
fi

echo "$OUTPUT"
