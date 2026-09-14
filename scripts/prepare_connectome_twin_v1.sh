#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/python_env.sh

SWC_DIR="${SWC_DIR:-}"
RECORDING="${RECORDING:-artifacts/showcase/who-farted-run.json}"
CROSSWALK="${CROSSWALK:-results/e002/pfl3-phase-crosswalk-v1.json}"
STEERING_CONFIG="${STEERING_CONFIG:-configs/steering_scaffold_candidate_v1.json}"
E002E="${E002E:-results/e002/pfl3-descending-steering-v2.json}"
OUT_DIR="${OUT_DIR:-artifacts/connectome-twin}"
WEB_DATA="${WEB_DATA:-web/connectome-twin/public/data}"
BODY_IDS="$OUT_DIR/body-ids-v1.json"
SKELETONS="$OUT_DIR/route-skeletons-v1.json"
STREAM="$OUT_DIR/replay-stream-v1.json"

if [[ -z "$SWC_DIR" ]]; then
  cat >&2 <<'EOF'
Set SWC_DIR to the official MaleCNS skeletons-swc directory.
Example after downloading selected SWCs:
  SWC_DIR=/path/to/skeletons-swc bash scripts/prepare_connectome_twin_v1.sh

Official source:
  gs://flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/
EOF
  exit 2
fi
if [[ ! -d "$SWC_DIR" ]]; then
  echo "SWC directory not found: $SWC_DIR" >&2
  exit 2
fi
if [[ ! -f "$RECORDING" || ! -f "$CROSSWALK" || ! -f "$STEERING_CONFIG" ]]; then
  echo "Missing recording/crosswalk/steering config required for connectome twin prep." >&2
  exit 2
fi

mkdir -p "$OUT_DIR" "$WEB_DATA"

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

crosswalk = json.loads(Path("$CROSSWALK").read_text())
steering = json.loads(Path("$STEERING_CONFIG").read_text())
metadata = {}
ids = []
for row in crosswalk["records"]:
    body_id = int(row["body_id"])
    ids.append(body_id)
    metadata[str(body_id)] = {
        "role": "PFL3",
        "readout_side": row.get("readout_side"),
        "column": row.get("column"),
        "pb_label": row.get("pb_label"),
    }
for role in ("steer_left", "steer_right"):
    for value in steering["candidate_roles"][role]:
        body_id = int(value)
        ids.append(body_id)
        metadata[str(body_id)] = {
            "role": "DNa02",
            "readout_side": role,
            "column": None,
            "pb_label": None,
        }
ids = list(dict.fromkeys(ids))
Path("$BODY_IDS").write_text(
    json.dumps({"body_ids": ids, "metadata": metadata}, indent=2, sort_keys=True) + "\n"
)
print(f"connectome-twin exact route body IDs: {len(ids)}")
PY

"$PYTHON_BIN" -m fly_sniff.connectome_twin_assets \
  "$SWC_DIR" \
  --body-ids-file "$BODY_IDS" \
  --max-segments-per-neuron 1000000 \
  --output "$SKELETONS"

# Add display metadata derived only from the frozen crosswalk and steering-role files.
# Geometry itself remains the exact published MaleCNS centerline data.
"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

body_payload = json.loads(Path("$BODY_IDS").read_text())
assets = json.loads(Path("$SKELETONS").read_text())
metadata = body_payload.get("metadata", {})
counts = {}
for neuron in assets["neurons"]:
    meta = metadata.get(str(int(neuron["body_id"])), {})
    neuron["display_metadata"] = meta
    role = str(meta.get("role") or "unclassified")
    counts[role] = counts.get(role, 0) + 1
assets["display_metadata_source"] = {
    "crosswalk": "$CROSSWALK",
    "steering_config": "$STEERING_CONFIG",
    "claim": "Display labels only; role colors are not measured activity.",
}
assets["role_counts"] = counts
Path("$SKELETONS").write_text(json.dumps(assets, separators=(",", ":")) + "\n")
print("connectome-twin roles:", counts)
PY

MECHANISM_ARGS=()
if [[ -f "$E002E" ]] && "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
r = json.loads(Path("$E002E").read_text())
raise SystemExit(0 if r.get("passed") is True else 1)
PY
then
  MECHANISM_ARGS=(--mechanism-report "$E002E")
else
  echo "E002e passed report not present; exporting behavior timeline without promoted E002e mechanism." >&2
fi

"$PYTHON_BIN" -m fly_sniff.connectome_twin_stream \
  "$RECORDING" \
  --connectome-assets "$SKELETONS" \
  "${MECHANISM_ARGS[@]}" \
  --output "$STREAM"

cp "$SKELETONS" "$WEB_DATA/route-skeletons-v1.json"
cp "$STREAM" "$WEB_DATA/replay-stream-v1.json"

echo
printf 'Prepared:\n  %s\n  %s\n' "$WEB_DATA/route-skeletons-v1.json" "$WEB_DATA/replay-stream-v1.json"
echo
echo "Run the client:"
echo "  cd web/connectome-twin && npm install && npm run dev"
