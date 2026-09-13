#!/usr/bin/env bash
set -euo pipefail

source scripts/python_env.sh

RECORDING="${1:-artifacts/showcase/who-farted-run.json}"
OUTPUT="${2:-artifacts/showcase/who-farted-v4.mp4}"

for required in \
  results/e002/pfl3-convergence-v1.json \
  results/route/fc2-goal-interface-audit-v1.json \
  results/route/heading-route-audit-v1.json \
  results/route/heading-topography-review-v1.json \
  configs/steering_scaffold_candidate_v1.json \
  "$RECORDING"
do
  if [[ ! -f "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 2
  fi
done

bash scripts/run_e002d_phase_crosswalk.sh
bash scripts/run_e002d_phase_probe.sh

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

path = Path("results/e002/pfl3-phase-comparison-v1.json")
report = json.loads(path.read_text())
print(
    f"E002d passed={report['passed']} "
    f"gates={report['passed_gate_count']}/{report['gate_count']} "
    f"primary_threshold={report['primary_structural_threshold']}"
)
for gate in report["gates"]:
    print(f"  {'PASS' if gate['passed'] else 'FAIL'}  {gate['name']}")
if not report["passed"]:
    raise SystemExit(
        "E002d did not qualify. Result is preserved; renderer may still show E002c, "
        "but do not promote a phase-comparison claim."
    )
PY

bash scripts/render_showcase_v4.sh "$RECORDING" "$OUTPUT"

echo "E002d result: results/e002/pfl3-phase-comparison-v1.json"
echo "Phase crosswalk: results/e002/pfl3-phase-crosswalk-v1.json"
echo "Showcase: $OUTPUT"
