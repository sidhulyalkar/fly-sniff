#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
ANNOTATIONS="${ANNOTATIONS:-data/raw/body-annotations-male-cns-v1.0.feather}"
WEIGHTS="${WEIGHTS:-data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather}"
CONFIG="${CONFIG:-configs/heading_route_audit_v1.json}"
REPORT="${REPORT:-results/route/heading-route-audit-v1.json}"

if [[ ! -x "$PY" ]]; then
  echo "Missing repo Python: $PY" >&2
  exit 2
fi
for path in "$ANNOTATIONS" "$WEIGHTS" "$CONFIG"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 2
  fi
done

mkdir -p "$(dirname "$REPORT")"

"$PY" -m fly_sniff.literature_route_audit \
  "$ANNOTATIONS" \
  "$WEIGHTS" \
  --config "$CONFIG" \
  --output "$REPORT"

"$PY" - <<PY
import json
from pathlib import Path

report = json.loads(Path("$REPORT").read_text())
print("\nHeading route audit summary")
print("git_sha:", report["runtime"]["git_sha"])
print("population counts:")
for name in ("EPG", "Delta7", "PFL3"):
    print(" ", name, report["populations"][name]["count"])
print("direct support:")
for row in report["direct_predictions"]:
    print(
        " ",
        f"{row['source']}->{row['target']}",
        "edges=",
        row["observed_edge_pairs"],
        "weight_sum=",
        row["observed_weight_sum"],
    )
print("two-hop support:")
for row in report["two_hop_predictions"]:
    print(
        " ",
        f"{row['source']}->{row['via']}->{row['target']}",
        "paths=",
        row["observed_path_count"],
    )
print("report:", "$REPORT")
PY
