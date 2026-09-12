#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VPY="$ROOT/.venv/bin/python"
ANNOTATIONS="${ANNOTATIONS:-data/raw/body-annotations-male-cns-v1.0.feather}"
WEIGHTS="${WEIGHTS:-data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather}"
CONFIG="${CONFIG:-configs/literature_route_audit_v1.json}"
OUTPUT="${OUTPUT:-results/route/literature-route-audit-v1.json}"

if [[ ! -x "$VPY" ]]; then
  echo "ERROR: missing virtualenv Python: $VPY" >&2
  exit 2
fi
for path in "$ANNOTATIONS" "$WEIGHTS" "$CONFIG"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required input missing: $path" >&2
    exit 3
  fi
done

echo "===== CHECKOUT ====="
git branch --show-current
git rev-parse HEAD
git status --short

echo "===== TESTS ====="
"$ROOT/.venv/bin/ruff" check src tests
"$VPY" -m pytest -q

echo "===== LITERATURE-GUIDED ROUTE AUDIT ====="
mkdir -p "$(dirname "$OUTPUT")"
"$VPY" -m fly_sniff.literature_route_audit \
  "$ANNOTATIONS" \
  "$WEIGHTS" \
  --config "$CONFIG" \
  --output "$OUTPUT"

echo "===== OUTPUT ====="
ls -lh "$OUTPUT"
echo "Share this file: $OUTPUT"
