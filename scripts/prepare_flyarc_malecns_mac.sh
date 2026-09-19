#!/usr/bin/env bash
set -euo pipefail

ROOT="${FLYARC_DATA_ROOT:-$HOME/fly-sniff-data/malecns-v1.0}"
RAW="$ROOT/raw"
GRAPH="${FLYARC_GRAPH_OUT:-$ROOT/flyarc-core-4096}"
BASE="https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"

mkdir -p "$RAW"

download() {
  local name="$1"
  local target="$RAW/$name"
  if [[ -f "$target" ]]; then
    echo "resuming/verifying existing download: $target"
  fi
  curl -L --fail --retry 4 --retry-delay 2 -C - \
    "$BASE/$name" \
    -o "$target"
}

download "body-annotations-male-cns-v1.0-minconf-0.5.feather"
download "body-neurotransmitters-male-cns-v1.0.feather"
download "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

if [[ -e "$GRAPH" ]]; then
  echo "refusing to overwrite existing graph: $GRAPH" >&2
  exit 2
fi

fly-sniff-arc-prepare \
  --annotations "$RAW/body-annotations-male-cns-v1.0-minconf-0.5.feather" \
  --neurotransmitters "$RAW/body-neurotransmitters-male-cns-v1.0.feather" \
  --connectivity "$RAW/connectome-weights-male-cns-v1.0-minconf-0.5.feather" \
  --max-nodes 4096 \
  --min-weight 1 \
  --output "$GRAPH"

echo
echo "FlyARC MaleCNS core:"
echo "$GRAPH"
echo "$GRAPH/manifest.json"
