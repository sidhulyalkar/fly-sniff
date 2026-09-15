#!/usr/bin/env bash
set -euo pipefail

source scripts/python_env.sh

RECORDING="${1:-artifacts/showcase/who-farted-run.json}"
OUTPUT="${2:-results/e003/sensory-goal-encoder-v1.json}"

"$PYTHON_BIN" -m fly_sniff.sensory_goal_encoder \
  "$RECORDING" \
  --output "$OUTPUT"
