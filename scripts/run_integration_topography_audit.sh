#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
AUDIT="${AUDIT:-results/route/integration-route-audit-v1.json}"
CONFIG="${CONFIG:-configs/integration_topography_audit_v1.json}"
REPORT="${REPORT:-results/route/integration-topography-audit-v1.json}"

if [[ ! -x "$PY" ]]; then
  echo "Missing repo Python: $PY" >&2
  exit 2
fi
for path in "$AUDIT" "$CONFIG"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 2
  fi
done

mkdir -p "$(dirname "$REPORT")"

"$PY" -m fly_sniff.integration_topography_audit \
  "$AUDIT" \
  --config "$CONFIG" \
  --output "$REPORT"

"$PY" - <<PY
import json
from pathlib import Path

report = json.loads(Path("$REPORT").read_text())
print("\nIntegration topography audit summary")
print("directional_role_status:", report["directional_role_status"])
for name in ("PFNa_family", "PFNm_family", "PFNp_family", "hDeltaC", "hDeltaG", "PFL3"):
    row = report["populations"].get(name)
    if row:
        print(
            name,
            "column_parse=",
            f"{row['column_parsed_count']}/{row['body_id_count']}",
            "columns=",
            row["columns"],
        )
print("report:", "$REPORT")
PY
