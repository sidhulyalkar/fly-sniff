#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
INPUT="${INPUT:-results/route/goal-relay-audit-v1.json}"
OUTPUT="${OUTPUT:-results/route/fc2-goal-interface-audit-v1.json}"

if [[ ! -f "$INPUT" ]]; then
  echo "missing goal-relay audit: $INPUT" >&2
  exit 2
fi

$PYTHON -m ruff check src/fly_sniff/fc2_goal_interface_audit.py tests/test_fc2_goal_interface_audit.py
$PYTHON -m pytest -q tests/test_fc2_goal_interface_audit.py
$PYTHON -m fly_sniff.fc2_goal_interface_audit "$INPUT" --output "$OUTPUT"

echo "FC2 goal-interface output: $OUTPUT"
$PYTHON -c 'import hashlib,sys; p=sys.argv[1]; print("sha256", hashlib.sha256(open(p,"rb").read()).hexdigest())' "$OUTPUT"
