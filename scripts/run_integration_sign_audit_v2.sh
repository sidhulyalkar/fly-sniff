#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
WEIGHTS="${WEIGHTS:-data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather}"
AUDIT="${AUDIT:-results/route/integration-route-audit-v1.json}"
AUTHORITY="${AUTHORITY:-authority/malecns-v1.0-integration-transmitter-evidence.json}"
CONFIG="${CONFIG:-configs/integration_sign_audit_v2.json}"
REPORT="${REPORT:-results/route/integration-sign-audit-v2.json}"

if [[ ! -x "$PY" ]]; then
  echo "Missing repo Python: $PY" >&2
  exit 2
fi
for path in "$WEIGHTS" "$AUDIT" "$AUTHORITY" "$CONFIG"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 2
  fi
done

mkdir -p "$(dirname "$REPORT")"

"$PY" -m fly_sniff.integration_sign_audit_v2 \
  "$WEIGHTS" \
  "$AUDIT" \
  "$AUTHORITY" \
  --config "$CONFIG" \
  --output "$REPORT"

"$PY" - <<PY
import json
from pathlib import Path

report = json.loads(Path("$REPORT").read_text())
print("\nIntegration sign audit v2 summary")
print("ready_for_modeled_sign_probe:", report["ready_for_modeled_sign_probe"])
print("ready_with_sensitivity_requirements:", report["ready_with_sensitivity_requirements"])
if report["prediction_confidence_order"]:
    lowest = report["prediction_confidence_order"][0]
    print("lowest prediction confidence:", lowest)
print("mandatory sensitivity cases:", len(report["mandatory_sensitivity_cases"]))
print("report:", "$REPORT")
PY
