#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${ROOT}" ]]; then
  echo "error: run inside fly-sniff checkout" >&2
  exit 2
fi
cd "${ROOT}"

DATA_ROOT="${FLY_SNIFF_DATA_DIR:-${HOME}/fly-sniff-data}"
CODE_REF="$(git rev-parse HEAD)"
OUT_ROOT="${FLY_SNIFF_AUTHORITY_UNLOCK_DIR:-${DATA_ROOT}/artifacts/olfactory-authority-unlock-${CODE_REF:0:12}}"
E002_OUT="${OUT_ROOT}/e002"
E001_OUT="${OUT_ROOT}/e001"
E001_PDF="${FLY_SNIFF_E001_PDF:-${DATA_ROOT}/sources/stensmyr-2012-cell.pdf}"

VENV="${FLY_SNIFF_VENV:-${ROOT}/.venv-olfactory}"
if [[ ! -x "${VENV}/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  "${PYTHON_BIN}" -m venv "${VENV}"
fi
source "${VENV}/bin/activate"
python -m pip install -q -e '.[dev]'

mkdir -p "${OUT_ROOT}" "$(dirname "${E001_PDF}")"

printf '\n== E002 pinned FlyWire candidate discovery ==\n'
python -m fly_sniff.olfactory_authority_unlock scan-e002 --out "${E002_OUT}"

printf '\n== E001 primary-paper byte freeze ==\n'
if [[ -f "${E001_PDF}" ]]; then
  python -m fly_sniff.olfactory_authority_unlock freeze-e001-paper \
    --paper "${E001_PDF}" \
    --out "${E001_OUT}"
else
  cat <<EOF
E001 paper not found at:
  ${E001_PDF}

Fetch the open-archive Cell PDF into that exact path with:

  curl -L --fail --retry 3 \
    -A 'Mozilla/5.0' \
    'https://www.cell.com/article/S0092867412013578/pdf' \
    -o '${E001_PDF}'

Then verify:
  file '${E001_PDF}'
  head -c 5 '${E001_PDF}'

The first bytes must be %PDF-. If the endpoint returns HTML, do not rename it as a PDF.
Download the paper through the publisher/open-archive browser UI instead, save it to the same path,
and rerun this script.
EOF
fi

printf '\n== Authority-unlock outputs ==\n'
echo "root: ${OUT_ROOT}"
echo "E002: ${E002_OUT}/e002-flywire-v783-discovery.json"
if [[ -f "${E001_OUT}/e001-primary-paper-byte-freeze.json" ]]; then
  echo "E001: ${E001_OUT}/e001-primary-paper-byte-freeze.json"
else
  echo "E001: pending local primary-paper PDF"
fi

cat <<'EOF'

SCIENTIFIC BOUNDARY
-------------------
This runner performs source ingress and identity adjudication only.
It does not qualify E001 or E002 automatically.
It does not infer missing Or56a root IDs.
It does not coerce side='na'.
It does not authorize O003/O004.
EOF
