#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
RECORDING="${1:-${RECORDING:-}}"
OUTPUT="${2:-${OUTPUT:-artifacts/showcase/who-farted-recorded-v2.mp4}}"
EVIDENCE_CONFIG="${EVIDENCE_CONFIG:-configs/showcase_evidence_v2.json}"
SECONDS="${SECONDS:-16}"
FPS="${FPS:-30}"

if [[ ! -x "$PY" ]]; then
  echo "Missing repo Python: $PY" >&2
  exit 2
fi
if [[ -z "$RECORDING" ]]; then
  echo "Usage: bash scripts/render_showcase_v2.sh <recording.json> [output.mp4]" >&2
  exit 2
fi
for path in "$RECORDING" "$EVIDENCE_CONFIG"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 2
  fi
done

mkdir -p "$(dirname "$OUTPUT")"

MPLBACKEND=Agg "$PY" -m fly_sniff.recorded_showcase_v2 \
  "$RECORDING" \
  --evidence-config "$EVIDENCE_CONFIG" \
  --output "$OUTPUT" \
  --seconds "$SECONDS" \
  --fps "$FPS"

if command -v ffprobe >/dev/null 2>&1 && [[ "$OUTPUT" == *.mp4 ]]; then
  DIMENSIONS="$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height \
    -of csv=s=x:p=0 "$OUTPUT")"
  if [[ "$DIMENSIONS" != "1080x1350" ]]; then
    echo "Encoded showcase dimensions drifted: $DIMENSIONS (expected 1080x1350)" >&2
    exit 2
  fi
  echo "verified encoded dimensions: $DIMENSIONS"
fi

echo "showcase v2: $OUTPUT"
