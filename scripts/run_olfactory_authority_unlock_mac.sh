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
if [[ ! -f "${E001_PDF}" ]]; then
  TMP_PDF="${E001_PDF}.download"
  rm -f "${TMP_PDF}"
  echo "Attempting bounded open-archive publisher download..."
  if curl -L --fail --retry 3 --connect-timeout 20 --max-time 120 \
    -A 'Mozilla/5.0' \
    'https://www.cell.com/article/S0092867412013578/pdf' \
    -o "${TMP_PDF}"; then
    if [[ "$(head -c 5 "${TMP_PDF}" 2>/dev/null || true)" == "%PDF-" ]]; then
      mv "${TMP_PDF}" "${E001_PDF}"
      echo "Verified PDF magic and promoted download to ${E001_PDF}"
    else
      echo "Publisher endpoint did not return PDF bytes; deleting temporary response." >&2
      rm -f "${TMP_PDF}"
    fi
  else
    rm -f "${TMP_PDF}"
  fi
fi

if [[ -f "${E001_PDF}" ]]; then
  python -m fly_sniff.olfactory_authority_unlock freeze-e001-paper \
    --paper "${E001_PDF}" \
    --out "${E001_OUT}"
else
  cat <<EOF
E001 publisher auto-fetch did not yield a PDF.

Use the Cell open-archive browser page for DOI 10.1016/j.cell.2012.09.046, download the article PDF,
and save it exactly here:
  ${E001_PDF}

Then rerun this script. The freezer will reject HTML or other non-PDF bytes.
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
