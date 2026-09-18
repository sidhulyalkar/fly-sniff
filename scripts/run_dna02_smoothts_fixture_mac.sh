#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-data/cache/dna02-smoothts-fixture-v1}"
OUT_JSON="${OUT_DIR}/smoothts-fixture-v1.json"
SHARE_ZIP="${OUT_DIR}-share.zip"

if ! command -v matlab >/dev/null 2>&1; then
  cat >&2 <<'EOF'
MATLAB was not found on PATH.

This probe needs a MATLAB installation that still provides Financial Toolbox
`smoothts`. If MATLAB is installed but not on PATH, invoke its full executable
path or run this from MATLAB manually:

  dna02_smoothts_fixture('data/cache/dna02-smoothts-fixture-v1/smoothts-fixture-v1.json')

No fly data are read by this fixture.
EOF
  exit 2
fi

rm -rf "${OUT_DIR}" "${SHARE_ZIP}"
mkdir -p "${OUT_DIR}"

MATLAB_OUT="${OUT_JSON}" matlab -batch \
  "addpath('scripts'); dna02_smoothts_fixture(getenv('MATLAB_OUT'));"

git rev-parse HEAD > "${OUT_DIR}/code-ref.txt"
shasum -a 256 "${OUT_JSON}" | awk '{print $1}' > "${OUT_DIR}/smoothts-fixture-v1.file-sha256"

python3 - "${OUT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
required_false = [
    "behavior_data_loaded",
    "fly_data_loaded",
    "navigation_performance_used",
]
for key in required_false:
    if payload.get(key) is not False:
        raise SystemExit(f"fixture boundary violation: {key}={payload.get(key)!r}")
if payload.get("schema") != "fly-sniff-dna02-smoothts-fixture-v1":
    raise SystemExit("unexpected fixture schema")
if payload.get("period_length") != 3 or payload.get("alpha") != 0.5:
    raise SystemExit("unexpected smoothing parameter fixture")
if len(payload.get("fixtures", [])) < 6:
    raise SystemExit("synthetic fixture set is incomplete")
print("Synthetic smoothts fixture boundary checks passed.")
PY

SHARE_DIR="${OUT_DIR}-share"
rm -rf "${SHARE_DIR}"
mkdir -p "${SHARE_DIR}"
cp "${OUT_JSON}" "${SHARE_DIR}/"
cp "${OUT_DIR}/code-ref.txt" "${SHARE_DIR}/"
cp "${OUT_DIR}/smoothts-fixture-v1.file-sha256" "${SHARE_DIR}/"

(
  cd "$(dirname "${SHARE_DIR}")"
  zip -qr "$(basename "${SHARE_ZIP}")" "$(basename "${SHARE_DIR}")"
)

echo
echo "Created MATLAB-only conformance bundle:"
echo "  ${SHARE_ZIP}"
echo "It contains synthetic fixtures only and no fly data."
