#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${ROOT}" ]]; then
  echo "error: run this inside the fly-sniff checkout" >&2
  exit 2
fi
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${FLY_SNIFF_VENV:-${ROOT}/.venv-olfactory}"
DATA_ROOT="${FLY_SNIFF_DATA_DIR:-${HOME}/fly-sniff-data}"
CODE_REF="$(git rev-parse HEAD)"
OUTPUT="${FLY_SNIFF_UNLOCK_DIR:-${DATA_ROOT}/artifacts/olfactory-unlock-${CODE_REF:0:12}}"
BUNDLE="${OUTPUT}-share.zip"

if [[ ! -x "${VENV}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV}"
  "${VENV}/bin/python" -m pip install --upgrade pip
  "${VENV}/bin/python" -m pip install -e '.[dev]'
elif [[ "${FLY_SNIFF_REFRESH_ENV:-0}" == "1" ]] || [[ ! -x "${VENV}/bin/fly-sniff-olfactory" ]]; then
  "${VENV}/bin/python" -m pip install -e '.[dev]'
fi

source "${VENV}/bin/activate"

printf '\n== Qualification regression tests ==\n'
pytest -q \
  tests/test_olfactory_e001_qualified.py \
  tests/test_olfactory_e002_adjudication.py \
  tests/test_olfactory_o002_freeze.py \
  tests/test_olfactory_cli.py \
  tests/test_olfactory_geosmin.py \
  tests/test_olfactory_structure.py

rm -rf "${OUTPUT}"
mkdir -p "${OUTPUT}"

printf '\n== E001 qualified qualitative physiology ==\n'
fly-sniff-olfactory --root "${ROOT}" validate-e001 | tee "${OUTPUT}/e001-qualified.json"

printf '\n== E002 original completeness gate ==\n'
fly-sniff-olfactory --root "${ROOT}" validate-e002 | tee "${OUTPUT}/e002-baseline.json"

printf '\n== E002 exact partial cohort adjudication ==\n'
fly-sniff-olfactory --root "${ROOT}" validate-e002-adjudication | tee "${OUTPUT}/e002-adjudication.json"

printf '\n== O002 development freeze ==\n'
fly-sniff-olfactory --root "${ROOT}" validate-o002-freeze | tee "${OUTPUT}/o002-freeze.json"

printf '\n== Current study gate ==\n'
fly-sniff-olfactory --root "${ROOT}" status | tee "${OUTPUT}/study-status.json"

python - "${OUTPUT}" "${ROOT}" "${CODE_REF}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
root = Path(sys.argv[2])
head = sys.argv[3]

def load(name: str) -> dict:
    return json.loads((out / name).read_text())

e001 = load("e001-qualified.json")
e002 = load("e002-adjudication.json")
o002 = load("o002-freeze.json")
status = load("study-status.json")

lines = [
    "OLFACTORY UNLOCK STATUS V1",
    f"repository: {root}",
    f"code_head: {head}",
    "",
    "O002",
    f"status: {o002['status']}",
    f"same_table_model_search_closed: {o002['same_table_model_search_closed']}",
    "",
    "E001",
    f"status: {e001['status']}",
    f"qualitative_usable: {e001['qualitative_usable']}",
    f"numeric_parameterization_usable: {e001['numeric_parameterization_usable']}",
    "",
    "E002",
    f"status: {e002['status']}",
    f"table_resolved_osn_body_records: {e002['table_resolved_osn_body_records']}",
    f"publication_total_osns: {e002['publication_total_osns']}",
    f"unresolved_osn_body_records: {e002['unresolved_osn_body_records']}",
    f"unresolved_osn_side_records: {e002['unresolved_osn_side_records']}",
    f"da2_lpn_body_records: {e002['da2_lpn_body_records']}",
    "",
    "GLOBAL GATE",
    f"study_status: {status['study_status']}",
    f"unresolved_count_registry: {status['unresolved_count']}",
    "confirmatory_O003_allowed: false",
    "",
    "NEXT SCIENTIFIC ACTION",
    "Reconcile the two publication-vs-annotation Or56a/DA2 OSN root IDs and the one source-side 'na' row.",
    "Do not reopen O002 model search while that structural identity work is unresolved.",
]
(out / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
PY

cp authority/geosmin-e001-qualified-v1.json "${OUTPUT}/"
cp authority/flywire-da2-e002-adjudication-v1.json "${OUTPUT}/"
cp authority/o002-development-freeze-v1.json "${OUTPUT}/"

rm -f "${BUNDLE}"
(
  cd "$(dirname "${OUTPUT}")"
  zip -qr "$(basename "${BUNDLE}")" "$(basename "${OUTPUT}")"
)

printf '\n== Final unlock summary ==\n'
cat "${OUTPUT}/SUMMARY.txt"

echo
echo "Review bundle: ${BUNDLE}"
