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
DOOR_ROOT="${FLY_SNIFF_DOOR_DIR:-${DATA_ROOT}/DoOR.data}"
DOOR_COMMIT="db323a496577c4b4a72b5c2fcd1859e07521ffb5"
OUT_ROOT="${FLY_SNIFF_OUTPUT_DIR:-${DATA_ROOT}/artifacts/e006-door-${DOOR_COMMIT:0:12}}"

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
fi
source "${VENV}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

printf '\n== Focused scientific-regression tests ==\n'
python -m pytest -q \
  tests/test_olfactory_program.py \
  tests/test_olfactory_geosmin.py \
  tests/test_olfactory_structure.py \
  tests/test_olfactory_door.py

printf '\n== Study preflight ==\n'
fly-sniff-olfactory --root "${ROOT}" status

printf '\n== E001 published geosmin evidence ==\n'
fly-sniff-olfactory --root "${ROOT}" validate-e001

printf '\n== E002 FlyWire Or56a/DA2 structural authority ==\n'
fly-sniff-olfactory --root "${ROOT}" validate-e002

mkdir -p "${DATA_ROOT}"
if [[ ! -d "${DOOR_ROOT}/.git" ]]; then
  printf '\n== Clone DoOR ==\n'
  git clone https://github.com/ropensci/DoOR.data.git "${DOOR_ROOT}"
fi

if [[ -n "$(git -C "${DOOR_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "error: DoOR checkout is dirty: ${DOOR_ROOT}" >&2
  echo "Keep the source checkout immutable; copy your own analyses elsewhere." >&2
  exit 3
fi

git -C "${DOOR_ROOT}" fetch origin "${DOOR_COMMIT}" --quiet || git -C "${DOOR_ROOT}" fetch --all --tags --quiet
git -C "${DOOR_ROOT}" checkout --detach "${DOOR_COMMIT}" --quiet

OBSERVED_COMMIT="$(git -C "${DOOR_ROOT}" rev-parse 'HEAD^{commit}')"
if [[ "${OBSERVED_COMMIT}" != "${DOOR_COMMIT}" ]]; then
  echo "error: DoOR checkout does not match frozen commit" >&2
  exit 4
fi

printf '\n== E006 source-resolved DoOR ingestion ==\n'
if [[ -d "${OUT_ROOT}" ]] && [[ -n "$(find "${OUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Existing non-empty output found: ${OUT_ROOT}"
  echo "Not overwriting it. Set FLY_SNIFF_OUTPUT_DIR to a new directory to create another receipt."
else
  fly-sniff-olfactory --root "${ROOT}" ingest-door \
    "${DOOR_ROOT}" \
    --output "${OUT_ROOT}"
fi

printf '\n== Artifact summary ==\n'
echo "Repository: ${ROOT}"
echo "DoOR source: ${DOOR_ROOT} @ ${DOOR_COMMIT}"
echo "E006 output: ${OUT_ROOT}"
if [[ -f "${OUT_ROOT}/door-e006-receipt.json" ]]; then
  OUT_ROOT_PY="${OUT_ROOT}" python - <<'PY'
import json
import os
from pathlib import Path
p = Path(os.environ["OUT_ROOT_PY"]) / "door-e006-receipt.json"
r = json.loads(p.read_text())
for key in (
    "responding_unit_count",
    "study_column_count",
    "response_cells",
    "observed_response_cells",
    "missing_response_cells",
    "geosmin_observed_cells",
    "receipt_sha256",
):
    print(f"{key}: {r[key]}")
PY
fi

cat <<'EOF'

FIRST-LIGHT INTERPRETATION
--------------------------
Passing this script means:
  * the preregistered study contracts are internally valid;
  * E001 and E002 have not been silently promoted beyond their evidence state;
  * the exact frozen DoOR source was ingested without normalization/aggregation;
  * an E006 content-addressed receipt exists.

It does NOT mean:
  * E001/E002/E006 are scientifically qualified;
  * O001 calibration has passed;
  * O003 may be run confirmatorily;
  * biological topology has shown any advantage.

The next scientific action is evidence adjudication, then a separately frozen O003 development lock.
EOF
