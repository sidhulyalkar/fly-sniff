#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
ANNOTATIONS="${ANNOTATIONS:-data/raw/body-annotations-male-cns-v1.0.feather}"
WEIGHTS="${WEIGHTS:-data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather}"
CONFIG="${CONFIG:-configs/persistent_goal_alias_audit_v1.json}"
OUTPUT="${OUTPUT:-results/route/persistent-goal-alias-audit-v1.json}"

if [[ ! -x "$PY" ]]; then
  echo "Missing repo Python: $PY" >&2
  echo "Run: RECREATE_VENV=1 bash scripts/bootstrap_macos_science.sh" >&2
  exit 2
fi
for path in "$ANNOTATIONS" "$WEIGHTS" "$CONFIG"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 2
  fi
done

mkdir -p "$(dirname "$OUTPUT")"

"$PY" -m fly_sniff.literature_route_audit \
  "$ANNOTATIONS" \
  "$WEIGHTS" \
  --config "$CONFIG" \
  --output "$OUTPUT"

"$PY" - <<PY
import json
from pathlib import Path
r = json.loads(Path("$OUTPUT").read_text())
print("\nPersistent-goal alias audit summary")
print("population counts:", {k: v["count"] for k, v in r["populations"].items()})
for row in r["direct_predictions"]:
    sweep = {x["min_weight"]: x["edge_pairs"] for x in row["threshold_sweep"]}
    print(f"{row['source']}->{row['target']}", sweep)
print("report:", "$OUTPUT")
PY
