#!/usr/bin/env bash
set -euo pipefail

source scripts/python_env.sh

E002C="${1:-results/e002/pfl3-convergence-v1.json}"
FC2="${2:-results/route/fc2-goal-interface-audit-v1.json}"
OUTPUT="${3:-results/showcase/showcase-evidence-local-v3.json}"

"$PYTHON_BIN" -m fly_sniff.showcase_evidence_seal \
  --e002c "$E002C" \
  --fc2 "$FC2" \
  --output "$OUTPUT"
