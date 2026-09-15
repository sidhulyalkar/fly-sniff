#!/usr/bin/env bash
set -euo pipefail

source scripts/python_env.sh

bash scripts/run_e002d_phase_crosswalk.sh

"$PYTHON_BIN" -m fly_sniff.pfl3_phase_probe \
  --protocol configs/e002d_pfl3_phase_comparison_protocol_v3.json \
  --runtime configs/e002d_phase_probe_runtime_v1.json \
  --crosswalk results/e002/pfl3-phase-crosswalk-v1.json \
  --e002c results/e002/pfl3-convergence-v1.json \
  --output results/e002/pfl3-phase-comparison-v1.json
