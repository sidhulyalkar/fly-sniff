#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:-data/raw/dna02}"
BASE_QC_DIR="${2:-data/cache/dna02-threshold-adjudication-v1}"
OUT_DIR="${3:-data/cache/dna02-threshold-high-review-v1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

FILES=(
  "${RAW_DIR}/180410_gfp_3G_ss730_dual_08_data_for_SH_with_lat_vel.mat"
  "${RAW_DIR}/180430_gfp_3G_ss730_dual_12_data_for_SH_with_lat_vel.mat"
  "${RAW_DIR}/180501_gfp_3G_ss730_dual_13_data_for_SH_with_lat_vel.mat"
  "${RAW_DIR}/180517_gfp_3G_ss730_dual_14_data_for_SH_with_lat_vel.mat"
)

for path in "${FILES[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing frozen DNa02 source file: ${path}" >&2
    exit 2
  fi
done

PROMINENCE="${BASE_QC_DIR}/prominence-audit-v1.json"
FULL_QC="${BASE_QC_DIR}/threshold-qc-v1.json"
for path in "${PROMINENCE}" "${FULL_QC}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing first-light QC artifact: ${path}" >&2
    echo "Run scripts/run_dna02_threshold_adjudication_mac.sh first." >&2
    exit 3
  fi
done

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

"${PYTHON_BIN}" -m pip install -e '.[dev,source]'
mkdir -p "${OUT_DIR}"

# Re-run the exact global prominence calculation only for the unresolved high-threshold
# candidates. With exactly three candidates, the existing deterministic raw-window
# renderer displays all three instead of only first/middle/last anchors.
fly-sniff-dna02-threshold-adjudicate \
  "${FILES[@]}" \
  --prominence-audit "${PROMINENCE}" \
  --candidate-quantiles 0.995 0.999 0.9995 \
  --out-dir "${OUT_DIR}/exact-global"

# Waveform families are rendered from the already content-addressed full QC receipt.
# This step does not reopen raw data and does not select a threshold.
fly-sniff-dna02-threshold-waveform-review \
  --qc "${FULL_QC}" \
  --candidate-quantiles 0.995 0.999 0.9995 0.9999 \
  --out-dir "${OUT_DIR}"

git rev-parse HEAD > "${OUT_DIR}/code-ref.txt"

SHARE_ZIP="${OUT_DIR%/}-share.zip"
"${PYTHON_BIN}" - "${OUT_DIR}" "${FULL_QC}" "${PROMINENCE}" "${SHARE_ZIP}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

out_dir = Path(sys.argv[1]).resolve()
full_qc = Path(sys.argv[2]).resolve()
prominence = Path(sys.argv[3]).resolve()
share_zip = Path(sys.argv[4]).resolve()

include = [
    full_qc,
    prominence,
    out_dir / "threshold-waveform-review-v1.json",
    out_dir / "exact-global" / "threshold-qc-v1.json",
    out_dir / "code-ref.txt",
]
include.extend(sorted((out_dir / "waveforms").glob("*.png")))
include.extend(sorted((out_dir / "exact-global" / "plots").glob("*-raw-windows.png")))

missing = [str(path) for path in include if not path.is_file()]
if missing:
    raise SystemExit(f"Refusing incomplete high-review bundle; missing: {missing}")

share_zip.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(share_zip, "w", compression=ZIP_DEFLATED) as archive:
    for path in include:
        if path == full_qc:
            arcname = "first-light/threshold-qc-v1.json"
        elif path == prominence:
            arcname = "first-light/prominence-audit-v1.json"
        else:
            arcname = str(path.relative_to(out_dir))
        archive.write(path, arcname=arcname)

print(share_zip)
PY

echo
echo "High-threshold neural-only review complete."
echo "Upload this bundle back to ChatGPT:"
echo "  ${SHARE_ZIP}"
echo
echo "Do NOT upload the raw MAT files."
echo "Do NOT inspect yaw, Figure 3C fit, or navigation while thresholds remain unfrozen."
