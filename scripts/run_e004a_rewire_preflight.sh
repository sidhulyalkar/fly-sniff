#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/python_env.sh

BUNDLE="${BUNDLE:-data/cache/steering-scaffold-v1}"
CONFIG="${CONFIG:-configs/e004a_rewire_prequalification_v1.json}"
OUTPUT="${OUTPUT:-results/e004/rewire-prequalification-v1.json}"

if [[ ! -d "$BUNDLE" ]]; then
  echo "Missing graph bundle: $BUNDLE" >&2
  echo "Run: bash scripts/run_steering_scaffold_probe.sh" >&2
  exit 2
fi

"$PYTHON_BIN" -m fly_sniff.rewire_preflight \
  "$BUNDLE" \
  --config "$CONFIG" \
  --output "$OUTPUT"

echo "$OUTPUT"
