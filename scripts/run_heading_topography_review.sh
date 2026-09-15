#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
AUDIT="${AUDIT:-results/route/heading-route-audit-v1.json}"
OUTPUT="${OUTPUT:-results/route/heading-topography-review-v1.json}"

if [[ ! -f "$AUDIT" ]]; then
  echo "missing heading route audit: $AUDIT" >&2
  exit 2
fi

$PYTHON -m ruff check src/fly_sniff/heading_topography_review.py tests/test_heading_topography_review.py
$PYTHON -m pytest -q tests/test_heading_topography_review.py
$PYTHON -m fly_sniff.heading_topography_review "$AUDIT" --output "$OUTPUT"

echo "heading topography review: $OUTPUT"
$PYTHON -c 'import hashlib,sys; p=sys.argv[1]; print("sha256", hashlib.sha256(open(p,"rb").read()).hexdigest())' "$OUTPUT"
