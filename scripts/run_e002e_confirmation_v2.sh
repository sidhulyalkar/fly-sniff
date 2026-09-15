#!/usr/bin/env bash
set -euo pipefail

source scripts/python_env.sh

BUNDLE=${1:-data/cache/steering-scaffold-v1}
OUTPUT=${2:-results/e002/pfl3-descending-steering-v2.json}

"$PYTHON_BIN" -m fly_sniff.pfl3_descending_probe_v2 \
  "$BUNDLE" \
  --output "$OUTPUT"
