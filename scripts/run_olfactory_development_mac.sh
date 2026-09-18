#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${ROOT}" ]]; then
  echo "error: run this inside the fly-sniff checkout" >&2
  exit 2
fi
cd "${ROOT}"

DATA_ROOT="${FLY_SNIFF_DATA_DIR:-${HOME}/fly-sniff-data}"
DOOR_COMMIT="db323a496577c4b4a72b5c2fcd1859e07521ffb5"
CODE_REF="$(git rev-parse HEAD)"
E006_ROOT="${FLY_SNIFF_OUTPUT_DIR:-${DATA_ROOT}/artifacts/e006-door-${DOOR_COMMIT:0:12}}"
AUDIT_ROOT="${FLY_SNIFF_AUDIT_DIR:-${DATA_ROOT}/artifacts/e006-audit-${DOOR_COMMIT:0:12}-${CODE_REF:0:12}}"
O002_ROOT="${FLY_SNIFF_O002_DIR:-${DATA_ROOT}/artifacts/o002-dev-${DOOR_COMMIT:0:12}-${CODE_REF:0:12}}"
O002_V2_ROOT="${FLY_SNIFF_O002_V2_DIR:-${DATA_ROOT}/artifacts/o002-robustness-${DOOR_COMMIT:0:12}-${CODE_REF:0:12}}"
O002_V3_ROOT="${FLY_SNIFF_O002_V3_DIR:-${DATA_ROOT}/artifacts/o002-stability-${DOOR_COMMIT:0:12}-${CODE_REF:0:12}}"
BUNDLE="${DATA_ROOT}/artifacts/olfactory-dev-cycle-${DOOR_COMMIT:0:12}-${CODE_REF:0:12}.zip"

printf '\n== Phase 1: source/evidence first light ==\n'
FLY_SNIFF_AUDIT_DIR="${AUDIT_ROOT}" \
  ./scripts/run_olfactory_first_light_mac.sh

source "${FLY_SNIFF_VENV:-${ROOT}/.venv-olfactory}/bin/activate"

printf '\n== Phase 2: O002 frozen within-study development ==\n'
if [[ -d "${O002_ROOT}" ]] && [[ -n "$(find "${O002_ROOT}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Existing O002 development artifact found: ${O002_ROOT}"
  echo "Not overwriting deterministic output for the same code/source identity."
else
  fly-sniff-olfactory --root "${ROOT}" run-o002-dev \
    "${E006_ROOT}" \
    --audit "${AUDIT_ROOT}/e006-audit.json" \
    --output "${O002_ROOT}"
fi

printf '\n== Phase 3: O002 frozen coding robustness decomposition ==\n'
if [[ -d "${O002_V2_ROOT}" ]] && [[ -n "$(find "${O002_V2_ROOT}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Existing O002 v2 robustness artifact found: ${O002_V2_ROOT}"
  echo "Not overwriting deterministic output for the same code/source identity."
else
  fly-sniff-olfactory --root "${ROOT}" run-o002-robustness \
    "${O002_ROOT}" \
    --output "${O002_V2_ROOT}"
fi

printf '\n== Phase 4: O002 frozen subspace and resampling stability ==\n'
if [[ -d "${O002_V3_ROOT}" ]] && [[ -n "$(find "${O002_V3_ROOT}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Existing O002 v3 stability artifact found: ${O002_V3_ROOT}"
  echo "Not overwriting deterministic output for the same code/source identity."
else
  fly-sniff-olfactory --root "${ROOT}" run-o002-stability \
    "${O002_ROOT}" \
    "${O002_V2_ROOT}" \
    --output "${O002_V3_ROOT}"
fi

printf '\n== Phase 5: compact combined review bundle ==\n'
rm -f "${BUNDLE}"
(
  cd "${DATA_ROOT}/artifacts"
  zip -qr "$(basename "${BUNDLE}")" \
    "$(basename "${AUDIT_ROOT}")" \
    "$(basename "${O002_ROOT}")" \
    "$(basename "${O002_V2_ROOT}")" \
    "$(basename "${O002_V3_ROOT}")"
)

printf '\n== O002 v1 summary ==\n'
cat "${O002_ROOT}/SUMMARY.txt"

printf '\n== O002 v2 robustness summary ==\n'
cat "${O002_V2_ROOT}/SUMMARY.txt"

printf '\n== O002 v3 stability summary ==\n'
cat "${O002_V3_ROOT}/SUMMARY.txt"

echo
echo "Combined review bundle: ${BUNDLE}"

cat <<'EOF'

SCIENTIFIC STATUS
-----------------
This command accelerates development by running only experiments already permitted by the frozen evidence
state. O002 remains exploratory/development-only. It does not alter E001/E002/E006 qualification state and
does not authorize O003/O004 confirmatory execution.

The O002 development cycle:
  * uses one source study selected before representation metrics are computed;
  * keeps source responding-unit identities rather than guessing receptor identities;
  * excludes SFR as a non-odor baseline row;
  * never zero-fills missing responses;
  * uses only complete odor rows for multivariate geometry;
  * runs a frozen v2 decomposition of magnitude-only, direction-only, identity-erased, channel-shuffle,
    leave-one-unit, and fixed feature-subset controls;
  * runs a frozen v3 class-wise, paired balanced-holdout, and low-rank subspace stability analysis;
  * treats repeated holdouts as stability diagnostics rather than independent biological replicates;
  * explicitly reports unsupported identity, valence, concentration-generalization, and topology claims.
EOF
