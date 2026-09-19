#!/usr/bin/env bash
set -euo pipefail

GRAPH="${1:-}"
if [[ -z "$GRAPH" ]]; then
  echo "usage: $0 /path/to/signed-graphbundle [output-dir]" >&2
  exit 2
fi

OUT="${2:-$HOME/fly-sniff-data/artifacts/flyarc-ls20-v1}"

if [[ "$(python -c 'import sys; print(int(sys.version_info >= (3, 12)))')" != "1" ]]; then
  echo "FlyARC requires Python 3.12+ for the current arc-agi toolkit." >&2
  exit 2
fi

python -m pytest -q \
  tests/test_flyarc.py \
  tests/test_flyarc_prepare.py \
  tests/test_flyarc_render.py \
  tests/test_flyarc_live_runner.py

fly-sniff-arc "$GRAPH" \
  --game ls20 \
  --steps 500 \
  --max-nodes 4096 \
  --allow-candidate \
  --output "$OUT"

GIF="$OUT.gif"
fly-sniff-arc-render "$OUT" --output "$GIF" --fps 8

echo
echo "FlyARC artifacts:"
echo "$OUT/comparison.json"
echo "$OUT/receipt.json"
echo "$GIF"
echo "$GIF.receipt.json"
