#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
ANNOTATIONS="${ANNOTATIONS:-data/raw/body-annotations-male-cns-v1.0.feather}"
WEIGHTS="${WEIGHTS:-data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather}"
INTEGRATION_AUDIT="${INTEGRATION_AUDIT:-results/route/integration-route-audit-v1.json}"
REPORT="${REPORT:-results/route/integration-sign-audit-v1.json}"

if [[ ! -x "$PY" ]]; then
  echo "Missing repo Python: $PY" >&2
  echo "Run: RECREATE_VENV=1 bash scripts/bootstrap_macos_science.sh" >&2
  exit 2
fi
for path in "$ANNOTATIONS" "$WEIGHTS" "$INTEGRATION_AUDIT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 2
  fi
done

mkdir -p "$(dirname "$REPORT")"

"$PY" -m fly_sniff.integration_sign_audit \
  "$ANNOTATIONS" \
  "$WEIGHTS" \
  "$INTEGRATION_AUDIT" \
  --output "$REPORT"
