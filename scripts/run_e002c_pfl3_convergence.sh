#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
E002B="${E002B:-results/e002/goal-channel-qualified-v1.json}"
HEADING="${HEADING:-results/route/heading-route-audit-v1.json}"
HEADING_TOPO="${HEADING_TOPO:-results/route/heading-topography-review-v1.json}"
ROUTE="${ROUTE:-results/route/integration-route-audit-v1.json}"
SIGN="${SIGN:-results/route/integration-sign-audit-v2.json}"
TOPO="${TOPO:-results/route/integration-topography-audit-v1.json}"
SEAL="${SEAL:-results/e002/e002c-input-seal-v1.json}"
OUTPUT="${OUTPUT:-results/e002/pfl3-convergence-v1.json}"

for path in "$E002B" "$HEADING" "$HEADING_TOPO" "$ROUTE" "$SIGN" "$TOPO"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required E002c input: $path" >&2
    exit 2
  fi
done

if [[ -n "$(git status --porcelain)" ]]; then
  echo "E002c official run requires a clean git worktree" >&2
  git status --short >&2
  exit 2
fi

$PYTHON -m ruff check \
  src/fly_sniff/e002c_input_seal.py \
  src/fly_sniff/pfl3_convergence_probe.py \
  src/fly_sniff/heading_topography_review.py \
  tests/test_pfl3_convergence_probe.py \
  tests/test_heading_topography_review.py \
  tests/test_e002b_qualification.py
$PYTHON -m pytest -q \
  tests/test_pfl3_convergence_probe.py \
  tests/test_heading_topography_review.py \
  tests/test_e002b_qualification.py

$PYTHON -m fly_sniff.e002c_input_seal \
  --e002b-qualification "$E002B" \
  --heading-route "$HEADING" \
  --heading-topography "$HEADING_TOPO" \
  --integration-route "$ROUTE" \
  --integration-sign "$SIGN" \
  --integration-topography "$TOPO" \
  --output "$SEAL"

$PYTHON -m fly_sniff.pfl3_convergence_probe \
  --e002b-qualification "$E002B" \
  --heading-route "$HEADING" \
  --heading-topography "$HEADING_TOPO" \
  --integration-route "$ROUTE" \
  --integration-sign "$SIGN" \
  --integration-topography "$TOPO" \
  --input-seal "$SEAL" \
  --output "$OUTPUT"

echo "E002c output: $OUTPUT"
$PYTHON -c 'import hashlib,sys; p=sys.argv[1]; print("sha256", hashlib.sha256(open(p,"rb").read()).hexdigest())' "$OUTPUT"
