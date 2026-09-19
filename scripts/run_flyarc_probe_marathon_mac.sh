#!/usr/bin/env bash
set -euo pipefail

GRAPH="${1:-$HOME/fly-sniff-data/malecns-v1.0/flyarc-core-4096}"
SOURCE_RUN="${2:-$HOME/fly-sniff-data/artifacts/flyarc-ls20-firstlight-4ed40fa}"
HOURS="${FLYARC_HOURS:-8}"
PROJECTIONS="${FLYARC_PROJECTIONS:-8}"
REWIRES="${FLYARC_REWIRES:-64}"
RANDOMS="${FLYARC_RANDOMS:-32}"
HEAD="$(git rev-parse --short HEAD)"
OUT="${3:-$HOME/fly-sniff-data/artifacts/flyarc-probes-v1-$HEAD}"

echo "== FlyARC representation marathon =="
echo "graph:       $GRAPH"
echo "source run:  $SOURCE_RUN"
echo "output:      $OUT"
echo "hours:       $HOURS"
echo "projections: $PROJECTIONS"
echo "rewires/p:   $REWIRES"
echo "randoms/p:   $RANDOMS"
echo

if [[ ! -f "$GRAPH/manifest.json" ]]; then
  echo "missing GraphBundle manifest: $GRAPH/manifest.json" >&2
  exit 2
fi
if [[ ! -f "$SOURCE_RUN/receipt.json" ]]; then
  echo "missing first-light receipt: $SOURCE_RUN/receipt.json" >&2
  exit 2
fi

python -m pytest -q tests/test_flyarc_probes.py

mkdir -p "$OUT"

CMD=(
  fly-sniff-arc-marathon
  "$GRAPH"
  "$SOURCE_RUN"
  --output "$OUT"
  --hours "$HOURS"
  --projection-count "$PROJECTIONS"
  --rewires-per-projection "$REWIRES"
  --randoms-per-projection "$RANDOMS"
)

echo
echo "== Starting resumable marathon =="
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu "${CMD[@]}" 2>&1 | tee "$OUT/console.log"
else
  "${CMD[@]}" 2>&1 | tee "$OUT/console.log"
fi

echo
echo "== FlyARC probe report =="
cat "$OUT/report.md"

echo
echo "saved:"
echo "$OUT/report.md"
echo "$OUT/aggregate.json"
echo "$OUT/summary.csv"
echo "$OUT/memory_curve.png"
echo "$OUT/primary_metric_distribution.png"
echo "$OUT/receipt.json"

if command -v open >/dev/null 2>&1; then
  open "$OUT/memory_curve.png"
  open "$OUT/primary_metric_distribution.png"
fi
