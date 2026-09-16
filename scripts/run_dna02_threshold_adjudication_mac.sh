#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:-data/raw/dna02}"
OUT_DIR="${2:-data/cache/dna02-threshold-adjudication-v1}"
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

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

"${PYTHON_BIN}" -m pip install -e '.[dev,source]'

mkdir -p "${OUT_DIR}"
PROMINENCE="${OUT_DIR}/prominence-audit-v1.json"

fly-sniff-dna02-prominence-audit \
  "${FILES[@]}" \
  --out "${PROMINENCE}"

fly-sniff-dna02-threshold-adjudicate \
  "${FILES[@]}" \
  --prominence-audit "${PROMINENCE}" \
  --out-dir "${OUT_DIR}"

git rev-parse HEAD > "${OUT_DIR}/code-ref.txt"

SHARE_ZIP="${OUT_DIR%/}-share.zip"
"${PYTHON_BIN}" - "${OUT_DIR}" "${SHARE_ZIP}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

out_dir = Path(sys.argv[1]).resolve()
share_zip = Path(sys.argv[2]).resolve()
include = [
    out_dir / "prominence-audit-v1.json",
    out_dir / "threshold-qc-v1.json",
    out_dir / "threshold-decisions-template.json",
    out_dir / "code-ref.txt",
]
include.extend(sorted((out_dir / "plots").glob("*.png")))
missing = [str(path) for path in include if not path.is_file()]
if missing:
    raise SystemExit(f"Refusing incomplete share bundle; missing: {missing}")

share_zip.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(share_zip, "w", compression=ZIP_DEFLATED) as archive:
    for path in include:
        archive.write(path, arcname=path.relative_to(out_dir))

print(share_zip)
PY

echo
echo "Neural-only threshold QC complete."
echo "Upload this small bundle back to ChatGPT:"
echo "  ${SHARE_ZIP}"
echo
echo "Do NOT upload the raw MAT files."
echo "Do NOT open yaw/behavior to choose thresholds."
