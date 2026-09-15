#!/usr/bin/env bash
set -euo pipefail

source scripts/python_env.sh

BUNDLE=${1:-data/cache/steering-scaffold-v1}
OUTPUT=${2:-results/e004/rewire-feasibility-v1.json}

"$PYTHON_BIN" -m fly_sniff.rewire_feasibility \
  "$BUNDLE" \
  --output "$OUTPUT"
