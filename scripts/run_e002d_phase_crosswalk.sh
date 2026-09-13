#!/usr/bin/env bash
set -euo pipefail

python -m fly_sniff.pfl3_phase_crosswalk \
  --config configs/e002d_pfl3_phase_comparison_protocol_v3.json \
  --e002c results/e002/pfl3-convergence-v1.json \
  --fc2 results/route/fc2-goal-interface-audit-v1.json \
  --heading results/route/heading-route-audit-v1.json \
  --steering configs/steering_scaffold_candidate_v1.json \
  --output results/e002/pfl3-phase-crosswalk-v1.json
