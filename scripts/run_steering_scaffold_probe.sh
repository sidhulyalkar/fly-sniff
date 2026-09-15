#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
AUDIT="${AUDIT:-results/route/literature-route-audit-v1.json}"
BUNDLE="${BUNDLE:-data/cache/steering-scaffold-v1}"
REPORT="${REPORT:-results/e002/steering-scaffold-v2.json}"

if [[ ! -x "$PY" ]]; then
  echo "Missing repo Python: $PY" >&2
  echo "Run: RECREATE_VENV=1 bash scripts/bootstrap_macos_science.sh" >&2
  exit 2
fi
if [[ ! -f "$AUDIT" ]]; then
  echo "Missing route audit: $AUDIT" >&2
  echo "Run: bash scripts/run_literature_route_audit.sh" >&2
  exit 2
fi

AUDIT_SHA="$($PY - <<'PY'
from fly_sniff.steering_scaffold import sha256_file
print(sha256_file("results/route/literature-route-audit-v1.json"))
PY
)"
EXPECTED_SHA="$($PY - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path("configs/steering_scaffold_candidate_v1.json").read_text())
print(cfg["evidence_audit_sha256"])
PY
)"

if [[ "$AUDIT" == "results/route/literature-route-audit-v1.json" && "$AUDIT_SHA" != "$EXPECTED_SHA" ]]; then
  echo "Route audit hash mismatch." >&2
  echo "observed: $AUDIT_SHA" >&2
  echo "expected: $EXPECTED_SHA" >&2
  exit 2
fi

rm -rf "$BUNDLE"
mkdir -p "$(dirname "$REPORT")"

"$PY" -m fly_sniff.steering_scaffold \
  "$AUDIT" \
  --output "$BUNDLE"

"$PY" -m fly_sniff.steering_probe \
  "$BUNDLE" \
  --output "$REPORT"

"$PY" - <<PY
import json
from pathlib import Path
r = json.loads(Path("$REPORT").read_text())
print("\nSteering scaffold probe summary")
print("protocol:", r["protocol"])
print("passed:", r["passed"])
print("left_turn:", r["probe"]["left_turn"])
print("right_turn:", r["probe"]["right_turn"])
print("pfl3_cut_peak_turn:", r["probe"]["pfl3_cut_peak_turn"])
print("bundle manifest sha256:", r["input_bundle"]["files"]["manifest.json"]["sha256"])
print("report:", "$REPORT")
PY
