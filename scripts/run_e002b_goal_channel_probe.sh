#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
ROUTE="${ROUTE:-results/route/integration-route-audit-v1.json}"
SIGN="${SIGN:-results/route/integration-sign-audit-v2.json}"
TOPO="${TOPO:-results/route/integration-topography-audit-v1.json}"
WEIGHTS="${WEIGHTS:-data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather}"
RUNTIME="${RUNTIME:-configs/e002b_probe_runtime_v1.json}"
OUTPUT="${OUTPUT:-results/e002/goal-channel-v1.json}"

for path in "$ROUTE" "$SIGN" "$TOPO" "$WEIGHTS" "$RUNTIME"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required E002b input: $path" >&2
    exit 2
  fi
done

SEED="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["seed"])' "$RUNTIME")"
STEPS="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["steps"])' "$RUNTIME")"
PULSE="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["pulse_steps"])' "$RUNTIME")"
AMPLITUDE="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["drive_amplitude"])' "$RUNTIME")"

$PYTHON - <<'PY'
import json
from pathlib import Path
p = json.loads(Path("configs/e002b_goal_channel_protocol_v1.json").read_text())
r = json.loads(Path("configs/e002b_probe_runtime_v1.json").read_text())
if r["structural_thresholds"] != p["fixed_structural_thresholds"]:
    raise SystemExit("E002b runtime threshold list drifted from preregistered protocol")
if not r.get("frozen_before_first_real_probe"):
    raise SystemExit("E002b runtime is not marked frozen before first real probe")
print("E002b frozen runtime:", r)
PY

$PYTHON -m ruff check src/fly_sniff/goal_channel_probe.py tests/test_goal_channel_probe.py
$PYTHON -m pytest -q tests/test_goal_channel_probe.py

$PYTHON -m fly_sniff.goal_channel_probe \
  "$ROUTE" \
  "$SIGN" \
  "$TOPO" \
  --weights "$WEIGHTS" \
  --seed "$SEED" \
  --steps "$STEPS" \
  --pulse-steps "$PULSE" \
  --drive-amplitude "$AMPLITUDE" \
  --output "$OUTPUT"

echo "E002b output: $OUTPUT"
$PYTHON -c 'import hashlib,sys; p=sys.argv[1]; print("sha256", hashlib.sha256(open(p,"rb").read()).hexdigest())' "$OUTPUT"
