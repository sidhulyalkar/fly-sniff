#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${ROOT}" ]]; then
  echo "error: run this inside the fly-sniff git checkout" >&2
  exit 2
fi
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${FLY_SNIFF_VENV:-${ROOT}/.venv-olfactory}"
DATA_ROOT="${FLY_SNIFF_DATA_DIR:-${HOME}/fly-sniff-data}"
CODE_REF="$(git rev-parse HEAD)"
DOOR_COMMIT="db323a496577c4b4a72b5c2fcd1859e07521ffb5"
E006_ROOT="${FLY_SNIFF_E006_DIR:-${DATA_ROOT}/artifacts/e006-door-${DOOR_COMMIT:0:12}}"
E001_ROOT="${FLY_SNIFF_E001_ARCHIVE_DIR:-${DATA_ROOT}/authorities/e001-stensmyr2012}"
E001_DOOR_ROOT="${FLY_SNIFF_E001_DOOR_DIR:-${DATA_ROOT}/artifacts/e001-door-crosscheck-${DOOR_COMMIT:0:12}-${CODE_REF:0:12}}"
REPORT_ROOT="${FLY_SNIFF_AUTHORITY_REPORT_DIR:-${DATA_ROOT}/artifacts/olfactory-authority-acquisition-${CODE_REF:0:12}}"
BUNDLE="${REPORT_ROOT}.zip"

if ! "${PYTHON_BIN}" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
then
  echo "error: Python >=3.11 is required. Set PYTHON_BIN to a suitable interpreter." >&2
  exit 2
fi

if [[ ! -d "${VENV}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV}"
  source "${VENV}/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install -e '.[dev]'
else
  source "${VENV}/bin/activate"
  python -m pip install -e '.[dev]' >/dev/null
fi

printf '\n== Authority scientific-regression tests ==\n'
python -m pytest -q \
  tests/test_olfactory_e001_source.py \
  tests/test_olfactory_e001_crosscheck.py \
  tests/test_olfactory_geosmin.py \
  tests/test_olfactory_structure.py

mkdir -p "${DATA_ROOT}/authorities"

printf '\n== E001 freeze/verify Stensmyr 2012 institutional archive ==\n'
if [[ -f "${E001_ROOT}/e001-source-receipt.json" ]] && \
   [[ -f "${E001_ROOT}/stensmyr2012-cell-with-supplement.pdf" ]]; then
  python -m fly_sniff.olfactory_e001_source \
    authority/geosmin-e001-source-v1.json \
    --output "${E001_ROOT}" \
    --verify-existing >/tmp/fly-sniff-e001-source.json
else
  if [[ -e "${E001_ROOT}" ]] && [[ -n "$(find "${E001_ROOT}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "error: E001 archive directory exists but is incomplete/nonempty: ${E001_ROOT}" >&2
    exit 3
  fi
  python -m fly_sniff.olfactory_e001_source \
    authority/geosmin-e001-source-v1.json \
    --output "${E001_ROOT}" >/tmp/fly-sniff-e001-source.json
fi
cat "${E001_ROOT}/SUMMARY.txt"

printf '\n== E001 published-claim guardrails ==\n'
python -m fly_sniff.olfactory_geosmin \
  authority/geosmin-e001-evidence-v0.json

printf '\n== E001 cross-check frozen Stensmyr DoOR transcription ==\n'
if [[ ! -f "${E006_ROOT}/door-e006-receipt.json" ]] || \
   [[ ! -f "${E006_ROOT}/door-responses-long.csv" ]]; then
  echo "error: immutable E006 ingestion is missing: ${E006_ROOT}" >&2
  echo "Run scripts/run_olfactory_first_light_mac.sh first." >&2
  exit 4
fi
if [[ -f "${E001_DOOR_ROOT}/e001-door-crosscheck-receipt.json" ]]; then
  python -m fly_sniff.olfactory_e001_crosscheck \
    "${E006_ROOT}" \
    --output "${E001_DOOR_ROOT}" \
    --verify-existing >/tmp/fly-sniff-e001-door.json
else
  if [[ -e "${E001_DOOR_ROOT}" ]] && [[ -n "$(find "${E001_DOOR_ROOT}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "error: E001 DoOR cross-check directory exists but is incomplete/nonempty: ${E001_DOOR_ROOT}" >&2
    exit 5
  fi
  python -m fly_sniff.olfactory_e001_crosscheck \
    "${E006_ROOT}" \
    --output "${E001_DOOR_ROOT}" >/tmp/fly-sniff-e001-door.json
fi
cat "${E001_DOOR_ROOT}/SUMMARY.txt"

printf '\n== E002 exact FlyWire v783 identity reconciliation ==\n'
python -m fly_sniff.olfactory_structure \
  authority/flywire-da2-e002-v0.json \
  | tee /tmp/fly-sniff-e002-identity.json

printf '\n== Build compact authority review bundle ==\n'
rm -rf "${REPORT_ROOT}"
mkdir -p "${REPORT_ROOT}"
cp "${E001_ROOT}/e001-source-receipt.json" "${REPORT_ROOT}/"
cp "${E001_ROOT}/SUMMARY.txt" "${REPORT_ROOT}/E001-SOURCE-SUMMARY.txt"
cp "${E001_DOOR_ROOT}/e001-door-crosscheck-receipt.json" "${REPORT_ROOT}/"
cp "${E001_DOOR_ROOT}/SUMMARY.txt" "${REPORT_ROOT}/E001-DOOR-SUMMARY.txt"
cp authority/geosmin-e001-source-v1.json "${REPORT_ROOT}/"
cp authority/geosmin-e001-evidence-v0.json "${REPORT_ROOT}/"
cp authority/flywire-da2-e002-v0.json "${REPORT_ROOT}/"
cp /tmp/fly-sniff-e002-identity.json "${REPORT_ROOT}/e002-validation.json"

python - "${REPORT_ROOT}" "${CODE_REF}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
code_ref = sys.argv[2]
e001 = json.loads((root / "e001-source-receipt.json").read_text())
e001_door = json.loads((root / "e001-door-crosscheck-receipt.json").read_text())
e002 = json.loads((root / "e002-validation.json").read_text())

lines = [
    "OLFACTORY AUTHORITY ACQUISITION V1",
    f"repository_code_ref: {code_ref}",
    "",
    "E001",
    f"status: {e001['status']}",
    f"pdf_bytes: {e001['acquisition']['pdf_bytes']}",
    f"pdf_sha256: {e001['acquisition']['pdf_sha256']}",
    f"receipt_sha256: {e001['receipt_sha256']}",
    f"door_crosscheck_status: {e001_door['status']}",
    f"door_ab4B_geosmin_raw_response: {e001_door['geosmin']['ab4B_raw_response']}",
    f"door_crosscheck_receipt_sha256: {e001_door['receipt_sha256']}",
    "",
    "E002",
    f"status: {e002['status']}",
    f"osn_body_records: {e002['osn_body_records']}",
    f"da2_lpn_body_records: {e002['da2_lpn_body_records']}",
    f"unresolved_osn_side_records: {e002['unresolved_osn_side_records']}",
    f"unresolved_discrepancies: {e002['unresolved_discrepancies']}",
    f"confirmatory_usable: {e002['confirmatory_usable']}",
    "",
    "NEXT SCIENTIFIC BLOCKERS",
    "- E001: institutional source provenance and frozen DoOR transcription are now checked; numerical-calibration metadata remains unresolved.",
    "- E002: recover the publication's missing annotated ORN_DA2 body, the manually added DA2 afferent root ID, and resolve fw044213 hemisphere.",
    "- O003/O004 remain blocked.",
]
(root / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
PY

rm -f "${BUNDLE}"
(
  cd "$(dirname "${REPORT_ROOT}")"
  zip -qr "$(basename "${BUNDLE}")" "$(basename "${REPORT_ROOT}")"
)

printf '\n== Final authority summary ==\n'
cat "${REPORT_ROOT}/SUMMARY.txt"
echo
echo "Full E001 PDF retained locally: ${E001_ROOT}/stensmyr2012-cell-with-supplement.pdf"
echo "Compact review bundle: ${BUNDLE}"

cat <<'EOF'

SCIENTIFIC BOUNDARY
-------------------
This runner acquires and verifies source provenance and identity authority only.
It does not:
  * treat published figures as raw trial data;
  * fit O001 numeric parameters from figure pixels or the DoOR 146.4 transcription sentinel;
  * guess unresolved Or56a hemisphere/body identities;
  * use connectivity or downstream model performance to expand the cohort;
  * authorize O003/O004 confirmatory execution.
EOF
