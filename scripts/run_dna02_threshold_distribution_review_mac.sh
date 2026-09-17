#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:-data/raw/dna02}"
AUDIT="${2:-data/cache/dna02-threshold-adjudication-v1/prominence-audit-v1.json}"
OUT_DIR="${3:-data/cache/dna02-threshold-distribution-review-v1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

FILES=(
  "${RAW_DIR}/180410_gfp_3G_ss730_dual_08_data_for_SH_with_lat_vel.mat"
  "${RAW_DIR}/180430_gfp_3G_ss730_dual_12_data_for_SH_with_lat_vel.mat"
  "${RAW_DIR}/180501_gfp_3G_ss730_dual_13_data_for_SH_with_lat_vel.mat"
  "${RAW_DIR}/180517_gfp_3G_ss730_dual_14_data_for_SH_with_lat_vel.mat"
)

for file in "${FILES[@]}"; do
  if [[ ! -f "${file}" ]]; then
    echo "Missing frozen DNa02 source: ${file}" >&2
    exit 2
  fi
done
if [[ ! -f "${AUDIT}" ]]; then
  echo "Missing prominence audit: ${AUDIT}" >&2
  exit 3
fi

"${PYTHON_BIN}" -m pip install -e '.[dev,source]'
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

fly-sniff-dna02-threshold-distribution-review \
  "${FILES[@]}" \
  --prominence-audit "${AUDIT}" \
  --out-dir "${OUT_DIR}"

git rev-parse HEAD > "${OUT_DIR}/code-ref.txt"

SHARE_DIR="${OUT_DIR}-share"
SHARE_ZIP="${OUT_DIR}-share.zip"
rm -rf "${SHARE_DIR}" "${SHARE_ZIP}"
mkdir -p "${SHARE_DIR}"
cp "${OUT_DIR}/threshold-distribution-review-v1.json" "${SHARE_DIR}/"
cp "${OUT_DIR}/code-ref.txt" "${SHARE_DIR}/"
cp -R "${OUT_DIR}/plots" "${SHARE_DIR}/plots"

(
  cd "$(dirname "${SHARE_DIR}")"
  zip -qr "$(basename "${SHARE_ZIP}")" "$(basename "${SHARE_DIR}")"
)

echo
echo "Created shareable neural-only bundle:"
echo "  ${SHARE_ZIP}"
echo "Upload that ZIP only. Do not upload the raw MAT files."
