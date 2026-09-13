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
  echo "ERROR: working tree is dirty; sealed execution requires an exact checkout." >&2
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

if [[ "${RUN_TRAIN:-0}" == "1" && "${RUN_FINAL:-0}" == "1" ]]; then
  echo "ERROR: RUN_TRAIN and RUN_FINAL are mutually exclusive phases." >&2
  exit 6
fi

# Phase 3: consume an already trained/frozen final exactly once. This path never
# rebuilds the candidate, retrains, requalifies, or refreezes anything.
if [[ "${RUN_FINAL:-0}" == "1" ]]; then
  OUT="${FINAL_RUN_DIR:-}"
  if [[ -z "$OUT" && -L results/sealed-experiment/LATEST ]]; then
    OUT="results/sealed-experiment/$(readlink results/sealed-experiment/LATEST)"
  fi
  if [[ -z "$OUT" || ! -d "$OUT" ]]; then
    echo "ERROR: set FINAL_RUN_DIR to the exact frozen development run directory." >&2
    exit 7
  fi

  CANDIDATE_MANIFEST="$OUT/candidate-graph-v1.json"
  MATCHED="$OUT/matched-training-v1.json"
  TRAINED_E002="$OUT/trained-e002-v1.json"
  FINAL_MANIFEST="$OUT/sealed-trained-final-v1.json"
  FINAL_LOCK="$OUT/final-run-consumed-v1.json"
  FINAL_OUT="$OUT/final-v1"
  for required in \
    "$OUT/SEALED_BEFORE_PERFORMANCE.md" \
    "$OUT/READY_FOR_FINAL.md" \
    "$OUT/navigation-seed-plan-v1.json" \
    "$CANDIDATE_MANIFEST" \
    "$MATCHED" \
    "$TRAINED_E002" \
    "$FINAL_MANIFEST"; do
    if [[ ! -f "$required" ]]; then
      echo "ERROR: frozen final prerequisite missing: $required" >&2
      exit 8
    fi
  done

  .venv/bin/fly-sniff-verify-candidate \
    "$BUNDLE" \
    "$CANDIDATE_MANIFEST" \
    --task-config configs/task_optimization_v1.json

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
  exit 0
fi

# Phase 2: create development performance only from an already sealed candidate.
# The sealed experiment command independently enforces HEAD == candidate.code_ref.
if [[ "${RUN_TRAIN:-0}" == "1" ]]; then
  OUT="${TRAIN_RUN_DIR:-}"
  if [[ -z "$OUT" && -L results/sealed-experiment/LATEST ]]; then
    OUT="results/sealed-experiment/$(readlink results/sealed-experiment/LATEST)"
  fi
  if [[ -z "$OUT" || ! -d "$OUT" ]]; then
    echo "ERROR: set TRAIN_RUN_DIR to the seal-only run directory." >&2
    exit 9
  fi

  CANDIDATE_MANIFEST="$OUT/candidate-graph-v1.json"
  MATCHED="$OUT/matched-training-v1.json"
  INTACT="$OUT/intact-training-v1.json"
  TRAINED_E002="$OUT/trained-e002-v1.json"
  FINAL_MANIFEST="$OUT/sealed-trained-final-v1.json"
  for required in \
    "$OUT/SEALED_BEFORE_PERFORMANCE.md" \
    "$OUT/navigation-seed-plan-v1.json" \
    "$CANDIDATE_MANIFEST"; do
    if [[ ! -f "$required" ]]; then
      echo "ERROR: development prerequisite missing: $required" >&2
      exit 10
    fi
  done
  if [[ -e "$MATCHED" || -e "$TRAINED_E002" || -e "$FINAL_MANIFEST" ]]; then
    echo "ERROR: development artifacts already exist in $OUT; refusing another v1 training run." >&2
    exit 11
  fi

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

The candidate graph and all train/validation/rewire/held-out/OOD seeds were sealed before any
navigation performance. Matched topology training, optimizer budget reconstruction, CEM replay,
rewire diagnostics, shortcut diagnostics, trained E002, and the final manifest are now frozen
under commit $(git rev-parse HEAD).

No held-out or OOD final episode has been evaluated.

Candidate manifest: $CANDIDATE_MANIFEST
Seed plan: $OUT/navigation-seed-plan-v1.json
Matched training: $MATCHED
Trained E002: $TRAINED_E002
Final manifest: $FINAL_MANIFEST
EOF

  echo
  echo "Development phase complete. Final namespace remains untouched."
  echo "Review: $OUT/READY_FOR_FINAL.md"
  echo "When ready to consume this exact run once:"
  echo "  RUN_FINAL=1 FINAL_RUN_DIR='$OUT' BUNDLE='$BUNDLE' bash scripts/run_sealed_experiment_mac.sh"
  exit 0
fi

# Phase 1 (default): seal all graph/sign/normalization/seed assumptions and STOP
# before any navigation objective is evaluated.
E001_RUN="${E001_RUN:-}"
if [[ -z "$E001_RUN" && -f results/e001/LATEST ]]; then
  E001_RUN="$(cat results/e001/LATEST)"
fi
if [[ -z "$E001_RUN" || ! -d "$E001_RUN" ]]; then
  echo "ERROR: set E001_RUN to the exact passing Mac E001 run directory." >&2
  exit 12
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
SEED_PLAN="$OUT/navigation-seed-plan-v1.json"
CANDIDATE_MANIFEST="$OUT/candidate-graph-v1.json"

for required in \
  "$STAGED_ROOT/staged_trace_report.json" \
  "$STAGED_ROOT/handoff_audit.json" \
  "$ROLE_REVIEW"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: missing required E001 artifact: $required" >&2
    exit 13
  fi
done

cat > "$OUT/checkout.txt" <<EOF
phase=seal-only
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

# Materialize every train, validation, rewire, held-out, and OOD seed now, before
# the first navigation objective exists. The candidate manifest embeds this plan.
.venv/bin/fly-sniff-seed-plan \
  --config configs/task_optimization_v1.json \
  --output "$SEED_PLAN"

.venv/bin/fly-sniff-seal-candidate \
  "$BUNDLE" \
  "$STAGED_ROOT" \
  "$ROLE_REVIEW" \
  "$EVIDENCE_OUT/e001_evidence.json" \
  "$SIGN_OUT/source_sign_authority.json" \
  --seed-plan "$SEED_PLAN" \
  --policy configs/candidate_graph_v1.json \
  --task-config configs/task_optimization_v1.json \
  --output "$CANDIDATE_MANIFEST"

.venv/bin/fly-sniff-verify-candidate \
  "$BUNDLE" \
  "$CANDIDATE_MANIFEST" \
  --task-config configs/task_optimization_v1.json

cat > "$OUT/SEALED_BEFORE_PERFORMANCE.md" <<EOF
# SEALED BEFORE PERFORMANCE

The exact GraphBundle membership, biological-edge accounting, source-body transmitter/sign
authority, role membership, PFN/odor sensory normalization, task config, exact train/validation/
rewire/held-out/OOD seed lists, source lineage, and experiment code are sealed under commit
$(git rev-parse HEAD).

No navigation training, validation objective, held-out episode, or OOD episode has been run by
this invocation.

Candidate manifest: $CANDIDATE_MANIFEST
Seed plan: $SEED_PLAN
Sign authority: $SIGN_OUT/source_sign_authority.json
E001 evidence: $EVIDENCE_OUT/e001_evidence.json
EOF

echo
echo "Candidate + seed seal complete. Navigation performance has NOT been created."
echo "Review: $OUT/SEALED_BEFORE_PERFORMANCE.md"
echo "Only after accepting this exact manifest, start development optimization with:"
echo "  RUN_TRAIN=1 TRAIN_RUN_DIR='$OUT' BUNDLE='$BUNDLE' bash scripts/run_sealed_experiment_mac.sh"
