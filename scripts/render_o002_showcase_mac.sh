#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${ROOT}" ]]; then
  echo "error: run this inside the fly-sniff checkout" >&2
  exit 2
fi
cd "${ROOT}"

DATA_ROOT="${FLY_SNIFF_DATA_DIR:-${HOME}/fly-sniff-data}"
DOOR_REF="db323a496577"
SOURCE_REF="${FLY_SNIFF_O002_SOURCE_REF:-4d2bdd13bf06}"
RENDER_REF="$(git rev-parse --short=12 HEAD)"

V1="${FLY_SNIFF_O002_DIR:-${DATA_ROOT}/artifacts/o002-dev-${DOOR_REF}-${SOURCE_REF}}"
V2="${FLY_SNIFF_O002_V2_DIR:-${DATA_ROOT}/artifacts/o002-robustness-${DOOR_REF}-${SOURCE_REF}}"
V3="${FLY_SNIFF_O002_V3_DIR:-${DATA_ROOT}/artifacts/o002-stability-${DOOR_REF}-${SOURCE_REF}}"
OUT="${FLY_SNIFF_O002_SHOWCASE_DIR:-${DATA_ROOT}/artifacts/o002-showcase-${SOURCE_REF}-render-${RENDER_REF}}"

VENV="${FLY_SNIFF_VENV:-${ROOT}/.venv-olfactory}"
if [[ ! -x "${VENV}/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  "${PYTHON_BIN}" -m venv "${VENV}"
fi
source "${VENV}/bin/activate"
python -m pip install -q -e '.[dev]'

printf '\n== O002 visual showcase preflight ==\n'
echo "source_ref: ${SOURCE_REF}"
echo "renderer_ref: ${RENDER_REF}"
echo "v1: ${V1}"
echo "v2: ${V2}"
echo "v3: ${V3}"
echo "output: ${OUT}"

required=(
  "${V1}/o002-development-receipt.json"
  "${V2}/o002-robustness-receipt.json"
  "${V3}/o002-stability-receipt.json"
)

for path in "${required[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "error: missing required frozen O002 artifact: ${path}" >&2
    echo "Run ./scripts/run_olfactory_development_mac.sh first, or set FLY_SNIFF_O002_SOURCE_REF." >&2
    exit 3
  fi
done

VIDEO_ARGS=()
if command -v ffmpeg >/dev/null 2>&1; then
  VIDEO_ARGS=(--video)
  echo "ffmpeg: found; 4:5 social MP4 will be rendered"
else
  echo "ffmpeg: not found; static scientific artifacts will still be rendered"
  echo "Install with: brew install ffmpeg"
fi

FORCE_ARGS=()
if [[ "${FLY_SNIFF_FORCE_SHOWCASE:-0}" == "1" ]]; then
  FORCE_ARGS=(--force)
fi

printf '\n== Receipt-driven visualization render ==\n'
fly-sniff-o002-visual \
  "${V1}" \
  "${V2}" \
  "${V3}" \
  --output "${OUT}" \
  "${VIDEO_ARGS[@]}" \
  "${FORCE_ARGS[@]}"

printf '\n== Output integrity ==\n'
for path in \
  "${OUT}/o002-hero-4x5.png" \
  "${OUT}/o002-scientific-deep-dive.png" \
  "${OUT}/o002-class-recall.png" \
  "${OUT}/o002-next-stage-roadmap.png" \
  "${OUT}/o002-visual-receipt.json" \
  "${OUT}/SUMMARY.txt" \
  "${OUT}/o002-showcase.zip"
do
  test -s "${path}"
  shasum -a 256 "${path}"
done

if [[ -f "${OUT}/o002-social.mp4" ]]; then
  shasum -a 256 "${OUT}/o002-social.mp4"
fi

printf '\n== Summary ==\n'
cat "${OUT}/SUMMARY.txt"

cat <<EOF

SHOWCASE COMPLETE
-----------------
Primary shareable:
  ${OUT}/o002-hero-4x5.png

Scientific deep dive:
  ${OUT}/o002-scientific-deep-dive.png

Motion explainer (when ffmpeg is available):
  ${OUT}/o002-social.mp4

Review bundle:
  ${OUT}/o002-showcase.zip

SCIENTIFIC BOUNDARY
-------------------
This renderer consumes already-frozen O002 development receipts and CSVs.
It does not recompute scientific metrics, alter experiment selection, qualify E001/E002,
or authorize O003/O004.
EOF
