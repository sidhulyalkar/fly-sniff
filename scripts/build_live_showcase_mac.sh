#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${ROOT}" ]]; then
  echo "error: run this inside the fly-sniff checkout" >&2
  exit 2
fi
cd "${ROOT}"

DATA_ROOT="${FLY_SNIFF_DATA_DIR:-${HOME}/fly-sniff-data}"
SOURCE_REF="${FLY_SNIFF_O002_SOURCE_REF:-4d2bdd13bf06}"
RENDER_REF="$(git rev-parse --short=12 HEAD)"
OUT="${FLY_SNIFF_LIVE_SHOWCASE_DIR:-${DATA_ROOT}/artifacts/live-showcase-${RENDER_REF}}"
SITE="${OUT}/site"
GRAPH="${FLY_SNIFF_SHOWCASE_GRAPH:-}"
O2_DIR="${FLY_SNIFF_O002_DIR:-${DATA_ROOT}/artifacts/o002-dev-db323a496577-${SOURCE_REF}}"
DEFAULT_ANNOTATIONS="${DATA_ROOT}/sources/malecns/body-annotations-male-cns-v1.0-minconf-0.5.feather"
ANNOTATIONS="${FLY_SNIFF_MALECNS_ANNOTATIONS:-}"
if [[ -z "${ANNOTATIONS}" && -s "${DEFAULT_ANNOTATIONS}" ]]; then
  ANNOTATIONS="${DEFAULT_ANNOTATIONS}"
fi
FETCH_SKELETONS="${FLY_SNIFF_FETCH_SKELETONS:-0}"

VENV="${FLY_SNIFF_VENV:-${ROOT}/.venv-olfactory}"
if [[ ! -x "${VENV}/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  "${PYTHON_BIN}" -m venv "${VENV}"
fi
source "${VENV}/bin/activate"
if [[ "${FLY_SNIFF_SKIP_INSTALL:-0}" != "1" ]]; then
  python -m pip install -q -e '.[dev]'
else
  echo "Skipping editable reinstall (FLY_SNIFF_SKIP_INSTALL=1)"
fi

rm -rf "${SITE}"
mkdir -p "${SITE}/data"
cp web/showcase/index.html "${SITE}/index.html"
cp web/showcase/styles.css "${SITE}/styles.css"
cp web/showcase/app.js "${SITE}/app.js"

EXPORT_CMD=(
  fly-sniff-live-showcase
  --output "${SITE}/data/showcase.json"
  --seed "${FLY_SNIFF_SHOWCASE_SEED:-13013}"
  --seconds "${FLY_SNIFF_SHOWCASE_SECONDS:-18}"
  --sample-hz "${FLY_SNIFF_SHOWCASE_HZ:-10}"
)

if [[ -n "${GRAPH}" ]]; then
  EXPORT_CMD+=(--graph "${GRAPH}")
  if [[ "${FLY_SNIFF_SHOWCASE_ALLOW_CANDIDATE:-0}" == "1" ]]; then
    EXPORT_CMD+=(--allow-candidate)
  fi
fi

printf '\n== Build live showcase data ==\n'
"${EXPORT_CMD[@]}"

if [[ -n "${ANNOTATIONS}" ]]; then
  printf '\n== Add measured whole-connectome soma context ==\n'
  echo "annotations: ${ANNOTATIONS}"
  fly-sniff-morphology-context soma \
    "${ANNOTATIONS}" \
    --output "${SITE}/data/connectome-soma.json" \
    --max-points "${FLY_SNIFF_SHOWCASE_SOMA_POINTS:-12000}"
fi

if [[ -n "${GRAPH}" && "${FETCH_SKELETONS}" == "1" ]]; then
  printf '\n== Add exact selected-circuit SWC morphology ==\n'
  fly-sniff-morphology-context skeletons \
    "${GRAPH}" \
    --output "${SITE}/data/selected-skeletons.json" \
    --cache-dir "${DATA_ROOT}/cache/malecns-v1.0-swc" \
    --max-neurons "${FLY_SNIFF_SHOWCASE_MAX_SKELETONS:-128}" \
    --max-segments-per-neuron "${FLY_SNIFF_SHOWCASE_MAX_SEGMENTS:-6000}"
fi

if [[ -f "${O2_DIR}/o002-development-receipt.json" ]]; then
  printf '\n== Add measured named-odor response explorer ==\n'
  fly-sniff-odor-explorer \
    "${O2_DIR}" \
    --output "${SITE}/data/o002-odor-explorer.json"
else
  echo "O002 v1 directory not found at ${O2_DIR}; named-odor explorer will be omitted"
fi

# Pull the latest O002 truth-card into the site when available.
O002_GLOB="${DATA_ROOT}/artifacts/o002-showcase-${SOURCE_REF}-render-*/o002-hero-4x5.png"
O002_RECEIPT_GLOB="${DATA_ROOT}/artifacts/o002-showcase-${SOURCE_REF}-render-*/o002-visual-receipt.json"
O002_HERO="$(ls -t ${O002_GLOB} 2>/dev/null | head -n 1 || true)"
O002_RECEIPT="$(ls -t ${O002_RECEIPT_GLOB} 2>/dev/null | head -n 1 || true)"

if [[ -n "${O002_HERO}" ]]; then
  cp "${O002_HERO}" "${SITE}/data/o002-hero-4x5.png"
  echo "included O002 hero: ${O002_HERO}"
else
  echo "O002 hero not found; frontend will show its honest fallback state"
fi

if [[ -n "${O002_RECEIPT}" ]]; then
  cp "${O002_RECEIPT}" "${SITE}/data/o002-visual-receipt.json"
fi

cat > "${OUT}/README.txt" <<EOF
FLY-SNIFF LIVE SHOWCASE

Site directory:
  ${SITE}

Preview locally:
  cd "${SITE}"
  python3 -m http.server 8080
  open http://localhost:8080

Default build:
  development-proxy movement only

Claim-bearing graph build:
  FLY_SNIFF_SHOWCASE_GRAPH=/path/to/odor-plume-qualified/graph ./scripts/build_live_showcase_mac.sh

Measured whole-connectome soma context:
  Automatically included when present at:
    ${DEFAULT_ANNOTATIONS}

  Override with:
    FLY_SNIFF_MALECNS_ANNOTATIONS=/path/to/body-annotations-male-cns-v1.0-minconf-0.5.feather \
      ./scripts/build_live_showcase_mac.sh

Fast repeat builds after the environment is already installed:
  FLY_SNIFF_SKIP_INSTALL=1 ./scripts/build_live_showcase_mac.sh

Exact selected-circuit skeletons:
  FLY_SNIFF_SHOWCASE_GRAPH=/path/to/graph \
  FLY_SNIFF_FETCH_SKELETONS=1 \
    ./scripts/build_live_showcase_mac.sh

The browser replays generated JSON. It does not run the scientific simulation.
EOF

printf '\n== Live showcase built ==\n'
echo "${SITE}"
find "${SITE}" -maxdepth 2 -type f -print

if [[ "${FLY_SNIFF_SERVE_SHOWCASE:-0}" == "1" ]]; then
  PORT="${FLY_SNIFF_SHOWCASE_PORT:-8080}"
  echo
  echo "Serving http://localhost:${PORT}"
  cd "${SITE}"
  python3 -m http.server "${PORT}"
fi
