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

VENV="${FLY_SNIFF_VENV:-${ROOT}/.venv-olfactory}"
if [[ ! -x "${VENV}/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  "${PYTHON_BIN}" -m venv "${VENV}"
fi
source "${VENV}/bin/activate"
python -m pip install -q -e '.[dev]'

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
  FLY_SNIFF_SHOWCASE_GRAPH=/path/to/qualified/graph ./scripts/build_live_showcase_mac.sh

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
