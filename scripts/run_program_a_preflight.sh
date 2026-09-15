#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/python_env.sh

mkdir -p authority results/calibration results

echo "== Program A: latent-wiring preflight =="
echo "python=$PYTHON_BIN"
echo

echo "[1/7] Build typed EvidenceLedger from frozen authorities"
"$PYTHON_BIN" -m fly_sniff.evidence.bootstrap \
  configs/evidence_bootstrap_v1.json \
  --output authority/evidence-ledger-v1.json

echo
echo "[2/7] Bind physiology probes to independent evidence"
"$PYTHON_BIN" -m fly_sniff.physiology_probe_binding \
  --config configs/physiology_calibration_v1.json \
  --ledger authority/evidence-ledger-v1.json \
  --output authority/physiology-probe-bindings-v1.json

echo
echo "[3/7] Validate frozen Program A protocols"
"$PYTHON_BIN" -m fly_sniff.connectome_necessity configs/connectome_necessity_v1.json
"$PYTHON_BIN" -m fly_sniff.physiology_calibration configs/physiology_calibration_v1.json
"$PYTHON_BIN" -m fly_sniff.null_factory_v2 configs/null_factory_v2.json
"$PYTHON_BIN" -m fly_sniff.zero_shot_experiment validate configs/zero_shot_latent_wiring_v1.json

echo
echo "[4/7] Optionally wrap a completed PR #23 structural audit"
STRUCTURE_ENVELOPE="authority/zero-shot-olfactory-structure-evidence-v1.json"
if [[ -n "${OLFACTORY_MOTION_AUDIT:-}" ]]; then
  rm -f "$STRUCTURE_ENVELOPE"
  "$PYTHON_BIN" -m fly_sniff.zero_shot_envelopes structure \
    "$OLFACTORY_MOTION_AUDIT" \
    --output "$STRUCTURE_ENVELOPE"
else
  echo "SKIP: set OLFACTORY_MOTION_AUDIT=/path/to/olfactory-motion-audit.json"
fi

echo
echo "[5/7] Optionally wrap a fully qualified PR #22 plume evidence set"
PLUME_ENVELOPE="authority/zero-shot-experimental-plume-evidence-v1.json"
if [[ -n "${PLUME_SOURCE_RECEIPT:-}" && -n "${PLUME_ARCHIVE_RECEIPT:-}" && -n "${PLUME_CUE_REFERENCE_RECEIPT:-}" && -n "${PLUME_PHYSICAL_GEOMETRY_RECEIPT:-}" ]]; then
  rm -f "$PLUME_ENVELOPE"
  "$PYTHON_BIN" -m fly_sniff.zero_shot_envelopes plume \
    --source "$PLUME_SOURCE_RECEIPT" \
    --archive "$PLUME_ARCHIVE_RECEIPT" \
    --cue-reference "$PLUME_CUE_REFERENCE_RECEIPT" \
    --physical-geometry "$PLUME_PHYSICAL_GEOMETRY_RECEIPT" \
    --output "$PLUME_ENVELOPE"
else
  echo "SKIP: plume envelope needs all four qualified receipts:"
  echo "  PLUME_SOURCE_RECEIPT"
  echo "  PLUME_ARCHIVE_RECEIPT"
  echo "  PLUME_CUE_REFERENCE_RECEIPT"
  echo "  PLUME_PHYSICAL_GEOMETRY_RECEIPT"
fi

echo
echo "[6/7] Optionally attach frozen null metadata to a real graph bundle"
if [[ -n "${NULL_SOURCE_BUNDLE:-}" ]]; then
  "$PYTHON_BIN" -m fly_sniff.null_metadata \
    "$NULL_SOURCE_BUNDLE" \
    --output "${NULL_METADATA_BUNDLE:-data/cache/program-a-null-metadata-v1}" \
    --report "${NULL_METADATA_REPORT:-results/null-metadata-v1.json}"
else
  echo "SKIP: set NULL_SOURCE_BUNDLE=/path/to/qualified-or-candidate-graph-bundle"
fi

echo
echo "[7/7] Compute fail-closed zero-shot readiness"
ARGS=(
  --zero-shot configs/zero_shot_latent_wiring_v1.json
  --physiology configs/physiology_calibration_v1.json
  --nulls configs/null_factory_v2.json
  --necessity configs/connectome_necessity_v1.json
  --artifact "evidence_ledger=authority/evidence-ledger-v1.json"
  --artifact "physiology_calibrated_model=results/calibration/physiology-calibrated-model-v1.json"
  --artifact "experimental_plume_receipt=$PLUME_ENVELOPE"
  --artifact "olfactory_motion_structural_audit=$STRUCTURE_ENVELOPE"
  --artifact "null_factory_protocol=configs/null_factory_v2.json"
  --artifact "connectome_necessity_protocol=configs/connectome_necessity_v1.json"
  --output results/science-readiness-v1.json
)
"$PYTHON_BIN" -m fly_sniff.science_readiness "${ARGS[@]}"

echo
echo "Program A preflight complete."
echo "The readiness report may correctly remain blocked while evidence is unresolved."
echo "See: results/science-readiness-v1.json"
