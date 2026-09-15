#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VPY="$ROOT/.venv/bin/python"
TRACE="$ROOT/.venv/bin/fly-sniff-trace"
AUDIT="$ROOT/.venv/bin/fly-sniff-audit-trace"
HANDOFF="$ROOT/.venv/bin/fly-sniff-science-handoff"
DOCTOR="$ROOT/.venv/bin/fly-sniff-doctor"

ANNOTATIONS="${ANNOTATIONS:-data/raw/body-annotations-male-cns-v1.0.feather}"
WEIGHTS="${WEIGHTS:-data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather}"
TRACE_DIR="${TRACE_DIR:-data/cache/staged-route-v1}"
AUDIT_OUT="${AUDIT_OUT:-results/trace/staged-route-v1-audit.json}"
HANDOFF_OUT="${HANDOFF_OUT:-results/handoff/science-handoff-v1.json}"

for path in "$VPY" "$TRACE" "$AUDIT" "$HANDOFF" "$DOCTOR"; do
  if [[ ! -x "$path" ]]; then
    echo "ERROR: missing environment executable: $path" >&2
    echo "Run: RECREATE_VENV=1 bash scripts/bootstrap_macos_science.sh" >&2
    exit 2
  fi
done

for path in "$ANNOTATIONS" "$WEIGHTS"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required raw MaleCNS file is missing: $path" >&2
    exit 3
  fi
done

echo "===== ENVIRONMENT ====="
"$DOCTOR"

echo "===== CHECKOUT ====="
git branch --show-current
git rev-parse HEAD
git status --short

echo "===== CORRECTED STRUCTURAL TRACE ====="
mkdir -p "$TRACE_DIR"
"$TRACE" \
  "$ANNOTATIONS" \
  "$WEIGHTS" \
  --source 'ORN' \
  --source '^Or' \
  --source '^Ir' \
  --target '^DNa02' \
  --max-hops 6 \
  --min-weight 5 \
  --fanout 30 \
  --output "$TRACE_DIR"

echo "===== STRUCTURAL AUDIT ====="
mkdir -p "$(dirname "$AUDIT_OUT")"
"$AUDIT" "$TRACE_DIR" --output "$AUDIT_OUT"

echo "===== SCIENCE HANDOFF ====="
mkdir -p "$(dirname "$HANDOFF_OUT")"
"$HANDOFF" \
  "$ANNOTATIONS" \
  "$WEIGHTS" \
  --trace-dir "$TRACE_DIR" \
  --output "$HANDOFF_OUT"

echo "===== RESULT FILES ====="
ls -lh \
  "$TRACE_DIR/trace_report.json" \
  "$TRACE_DIR/path_provenance.csv" \
  "$AUDIT_OUT" \
  "$HANDOFF_OUT"

echo
echo "Preferred file to share: $HANDOFF_OUT"
echo "Compact summary:"
"$VPY" - <<PY
import json
from pathlib import Path

p = json.loads(Path("$HANDOFF_OUT").read_text())
summary = {
    "git_sha": p["runtime"]["git_sha"],
    "trace_passed": p["trace_audit"]["passed"],
    "trace_counts": p["trace_audit"]["counts"],
    "olfactory_seed_union": p["olfactory_seed_hypothesis"]["union_count"],
    "anchor_corridor_counts": {
        k: v["retained_in_corridor_count"]
        for k, v in p["anchor_corridor_membership"].items()
    },
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
