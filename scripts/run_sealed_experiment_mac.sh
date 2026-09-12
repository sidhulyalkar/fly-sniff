#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXPECTED_BRANCH="${EXPECTED_BRANCH:-feat/sealed-graph-experiment-v1}"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" && "${ALLOW_OTHER_BRANCH:-0}" != "1" ]]; then
  echo "ERROR: expected branch '$EXPECTED_BRANCH', found '$CURRENT_BRANCH'." >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" && "${ALLOW_DIRTY:-0}" != "1" ]]; then
  echo "ERROR: working tree is dirty; seal and training must start from an exact checkout." >&2
  exit 3
fi

BUNDLE="${BUNDLE:-${1:-}}"
if [[ -z "$BUNDLE" || ! -d "$BUNDLE" ]]; then
  echo "Usage: BUNDLE=/path/to/sealed-graphbundle bash scripts/run_sealed_experiment_mac.sh" >&2
  exit 4
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "ERROR: .venv is missing. Run the Mac bootstrap first." >&2
  exit 5
fi

E001_RUN="${E001_RUN:-}"
if [[ -z "$E001_RUN" && -f results/e001/LATEST ]]; then
  E001_RUN="$(cat results/e001/LATEST)"
fi
if [[ -z "$E001_RUN" || ! -d "$E001_RUN" ]]; then
  echo "ERROR: set E001_RUN to the exact passing Mac E001 run directory." >&2
  exit 6
fi

STAGED_ROOT="$E001_RUN/staged-route-v1"
ROLE_REVIEW="$E001_RUN/role-review-v1/role_review.json"
if [[ ! -f "$ROLE_REVIEW" ]]; then
  ROLE_REVIEW="$E001_RUN/role_review.json"
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short=12 HEAD)"
OUT="results/sealed-experiment/$RUN_ID"
mkdir -p "$OUT"
ln -sfn "$RUN_ID" results/sealed-experiment/LATEST

EVIDENCE_OUT="$OUT/e001-evidence-deduplicated"
SIGN_OUT="$OUT/sign-authority"
CANDIDATE_MANIFEST="$OUT/candidate-graph-v1.json"
MATCHED="$OUT/matched-training-v1.json"
INTACT="$OUT/intact-training-v1.json"
TRAINED_E002="$OUT/trained-e002-v1.json"
FINAL_MANIFEST="$OUT/sealed-trained-final-v1.json"
FINAL_LOCK="$OUT/final-run-consumed-v1.json"
FINAL_OUT="$OUT/final-v1"

for required in \
  "$STAGED_ROOT/staged_trace_report.json" \
  "$STAGED_ROOT/handoff_audit.json" \
  "$ROLE_REVIEW"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: missing required E001 artifact: $required" >&2
    exit 7
  fi
done

cat > "$OUT/checkout.txt" <<EOF
branch=$CURRENT_BRANCH
sha=$(git rev-parse HEAD)
bundle=$BUNDLE
e001_run=$E001_RUN
EOF

# Rebuild only the evidence summary with unique-edge accounting. The underlying
# staged E001 route and role review are immutable inputs and are not rerun here.
.venv/bin/fly-sniff-e001-evidence \
  "$STAGED_ROOT" \
  "$ROLE_REVIEW" \
  --policy configs/role_review_v1.json \
  --authority authority/olfactory-navigation-literature-v1.json \
  --output "$EVIDENCE_OUT"

.venv/bin/fly-sniff-sign-authority \
  "$BUNDLE" \
  --authority authority/malecns-v1.0-steering-sign-evidence.json \
  --authority authority/malecns-v1.0-body-sign-overrides-v1.json \
  --output "$SIGN_OUT"

.venv/bin/fly-sniff-seal-candidate \
  "$BUNDLE" \
  "$STAGED_ROOT" \
  "$ROLE_REVIEW" \
  "$EVIDENCE_OUT/e001_evidence.json" \
  "$SIGN_OUT/source_sign_authority.json" \
  --policy configs/candidate_graph_v1.json \
  --task-config configs/task_optimization_v1.json \
  --output "$CANDIDATE_MANIFEST"

.venv/bin/fly-sniff-verify-candidate \
  "$BUNDLE" \
  "$CANDIDATE_MANIFEST" \
  --task-config configs/task_optimization_v1.json

# This is the first point at which navigation performance is allowed to exist.
.venv/bin/fly-sniff-sealed-train \
  "$BUNDLE" \
  "$CANDIDATE_MANIFEST" \
  --config configs/task_optimization_v1.json \
  --output "$MATCHED"

.venv/bin/fly-sniff-redteam-budget \
  "$MATCHED" \
  --config configs/task_optimization_v1.json \
  --output "$OUT/execution-budget-audit-v1.json"

.venv/bin/fly-sniff-redteam-cem-replay \
  "$MATCHED" \
  --config configs/task_optimization_v1.json \
  --output "$OUT/cem-replay-v1.json"

.venv/bin/fly-sniff-redteam-rewires \
  "$BUNDLE" \
  "$MATCHED" \
  --output "$OUT/rewire-ensemble-diagnostics-v1.json"

.venv/bin/python - "$MATCHED" "$INTACT" <<'PY'
import json
import sys
from pathlib import Path

matched = json.loads(Path(sys.argv[1]).read_text())
intact = matched["results"]["intact"]
Path(sys.argv[2]).write_text(json.dumps(intact, indent=2, sort_keys=True) + "\n")
PY

.venv/bin/fly-sniff-redteam-training \
  "$BUNDLE" \
  "$INTACT" \
  --config configs/task_optimization_v1.json \
  --output "$OUT/shortcut-diagnostics-v1.json"

.venv/bin/fly-sniff-sealed-qualify \
  "$BUNDLE" \
  "$CANDIDATE_MANIFEST" \
  "$MATCHED" \
  --config configs/task_optimization_v1.json \
  --output "$TRAINED_E002"

.venv/bin/fly-sniff-sealed-freeze-final \
  "$BUNDLE" \
  "$CANDIDATE_MANIFEST" \
  "$MATCHED" \
  "$TRAINED_E002" \
  --config configs/task_optimization_v1.json \
  --output "$FINAL_MANIFEST"

cat > "$OUT/READY_FOR_FINAL.md" <<EOF
# READY FOR FINAL

The candidate graph, sign authority, sensory normalization, matched topology training,
optimizer budget, CEM replay, rewire diagnostics, shortcut diagnostics, trained E002,
and final seed manifest are frozen under commit $(git rev-parse HEAD).

No held-out or OOD final episode has been evaluated by this script unless RUN_FINAL=1.

Candidate manifest: $CANDIDATE_MANIFEST
Matched training: $MATCHED
Trained E002: $TRAINED_E002
Final manifest: $FINAL_MANIFEST
EOF

if [[ "${RUN_FINAL:-0}" != "1" ]]; then
  echo
  echo "Development lane complete. Final namespace remains untouched."
  echo "Review: $OUT/READY_FOR_FINAL.md"
  echo "When ready to consume final exactly once:"
  echo "  RUN_FINAL=1 BUNDLE='$BUNDLE' E001_RUN='$E001_RUN' bash scripts/run_sealed_experiment_mac.sh"
  exit 0
fi

.venv/bin/fly-sniff-sealed-final \
  "$BUNDLE" \
  "$CANDIDATE_MANIFEST" \
  "$FINAL_MANIFEST" \
  "$MATCHED" \
  "$TRAINED_E002" \
  --config configs/task_optimization_v1.json \
  --output "$FINAL_OUT" \
  --final-lock "$FINAL_LOCK" \
  --arm-final

echo "Final v1 consumed. Do not rerun this protocol."
echo "Result: $FINAL_OUT/gold_report.json"
