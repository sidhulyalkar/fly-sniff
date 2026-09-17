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
CODE_REF="$(git rev-parse HEAD)"
AUDIT_ROOT="${FLY_SNIFF_AUDIT_DIR:-${DATA_ROOT}/artifacts/e006-audit-${DOOR_COMMIT:0:12}-${CODE_REF:0:12}}"

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
  if [[ "${FLY_SNIFF_REFRESH_ENV:-0}" == "1" ]] || ! command -v fly-sniff-olfactory >/dev/null 2>&1; then
    python -m pip install -e '.[dev]'
  fi
fi

printf '\n== Focused scientific-regression tests ==\n'
python -m pytest -q \
  tests/test_olfactory_program.py \
  tests/test_olfactory_geosmin.py \
  tests/test_olfactory_structure.py \
  tests/test_olfactory_door.py \
  tests/test_olfactory_e006_audit.py \
  tests/test_olfactory_cli.py

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
  echo "Existing immutable ingestion found: ${OUT_ROOT}"
  echo "The audit will verify its receipt and content hashes before using it."
else
  fly-sniff-olfactory --root "${ROOT}" ingest-door \
    "${DOOR_ROOT}" \
    --output "${OUT_ROOT}"
fi

printf '\n== E006 fail-closed adjudication audit ==\n'
rm -rf "${AUDIT_ROOT}"
fly-sniff-olfactory --root "${ROOT}" audit-e006 \
  "${OUT_ROOT}" \
  --door-checkout "${DOOR_ROOT}" \
  --output "${AUDIT_ROOT}"

printf '\n== Build compact review bundle ==\n'
SHARE_ZIP="${AUDIT_ROOT}-share.zip"
rm -f "${SHARE_ZIP}"
(
  cd "$(dirname "${AUDIT_ROOT}")"
  zip -qr "$(basename "${SHARE_ZIP}")" "$(basename "${AUDIT_ROOT}")"
)

printf '\n== Final summary ==\n'
cat "${AUDIT_ROOT}/SUMMARY.txt"
echo
echo "Repository: ${ROOT} @ ${CODE_REF}"
echo "DoOR source: ${DOOR_ROOT} @ ${DOOR_COMMIT}"
echo "E006 immutable ingestion: ${OUT_ROOT}"
echo "E006 audit: ${AUDIT_ROOT}"
echo "Share bundle: ${SHARE_ZIP}"

cat <<'EOF'

INTERPRETATION
--------------
A successful shell exit means the source and derived-artifact integrity checks ran successfully.
The audit may still report BLOCKED_METADATA_ADJUDICATION. That is a scientific state, not a
software failure.

The command intentionally does not:
  * normalize or aggregate raw DoOR studies;
  * resolve one-to-many receptor identities from downstream performance;
  * fill missing responses with zero;
  * authorize O003/O004 confirmatory execution.

For a claim-bearing qualification run, use a clean isolated worktree. Untracked files under
authority/, src/, tests/, scripts/, .github/, or pyproject.toml are reported as a scientific
worktree blocker.
EOF
