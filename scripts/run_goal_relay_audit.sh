#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
ANNOTATIONS="${ANNOTATIONS:-data/raw/body-annotations-male-cns-v1.0.feather}"
WEIGHTS="${WEIGHTS:-data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather}"
CONFIG="${CONFIG:-configs/goal_relay_audit_v1.json}"
OUTPUT="${OUTPUT:-results/route/goal-relay-audit-v1.json}"

for path in "$PY" "$ANNOTATIONS" "$WEIGHTS" "$CONFIG"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 2
  fi
done

mkdir -p "$(dirname "$OUTPUT")"

"$PY" -m fly_sniff.literature_route_audit \
  "$ANNOTATIONS" \
  "$WEIGHTS" \
  --config "$CONFIG" \
  --output "$OUTPUT" \
  --max-edges 400 \
  --max-paths 500

"$PY" - <<PY
import json
from pathlib import Path
p = Path("$OUTPUT")
r = json.loads(p.read_text())
print("\nGoal-relay audit summary")
print("protocol:", r["protocol"])
print("git_sha:", r["runtime"]["git_sha"])
print("population_counts:", {k: v["count"] for k, v in r["populations"].items()})
print("two_hop_paths:")
for row in r["two_hop_predictions"]:
    print(f"  {row['source']} -> {row['via']} -> {row['target']}: {row['observed_path_count']}")
print("output:", p)
PY
