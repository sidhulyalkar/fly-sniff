#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/python_env.sh

AUTHORITY="${AUTHORITY:-authority/door-response-authority-v1.json}"
CONFIG="${CONFIG:-configs/e003b_fast_preflight_v1.json}"
OUTPUT="${OUTPUT:-results/e003/complex-odor-fast-preflight-v1.json}"

if [[ ! -f "$AUTHORITY" ]]; then
  cat >&2 <<EOF
Missing pinned odor-response authority: $AUTHORITY

Build it from a complete long CSV with columns receptor,odorant,response:
  "$PYTHON_BIN" -m fly_sniff.odor_authority responses.csv \
    --source-name DoOR.data \
    --source-version <PINNED_VERSION> \
    --source-commit <PINNED_COMMIT> \
    --output "$AUTHORITY"

The scientific runner intentionally refuses to substitute toy receptor values.
EOF
  exit 2
fi

"$PYTHON_BIN" -m fly_sniff.complex_odor_benchmark \
  --config "$CONFIG" \
  --authority "$AUTHORITY" \
  --output "$OUTPUT"

echo "$OUTPUT"
