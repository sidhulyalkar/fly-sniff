#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXPECTED_BRANCH="${EXPECTED_BRANCH:-feat/e001-mac-ship-v1}"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" && "${ALLOW_OTHER_BRANCH:-0}" != "1" ]]; then
  echo "ERROR: expected branch '$EXPECTED_BRANCH', found '$CURRENT_BRANCH'." >&2
  echo "Switch first with: git switch $EXPECTED_BRANCH" >&2
  echo "Set ALLOW_OTHER_BRANCH=1 only for an intentional descendant/integration checkout." >&2
  exit 2
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif command -v python3.11 >/dev/null 2>&1; then
  PY="$(command -v python3.11)"
elif [[ -x /opt/homebrew/bin/python3.11 ]]; then
  PY="/opt/homebrew/bin/python3.11"
elif [[ -x /usr/local/bin/python3.11 ]]; then
  PY="/usr/local/bin/python3.11"
else
  echo "ERROR: Python 3.11 was not found." >&2
  echo "Install it with: brew install python@3.11" >&2
  exit 3
fi

echo "===== CHECKOUT ====="
echo "repository: $ROOT"
echo "branch: $CURRENT_BRANCH"
echo "sha: $(git rev-parse HEAD)"
echo "python: $PY"
"$PY" --version

if [[ "${RECREATE_VENV:-0}" == "1" ]]; then
  echo "[bootstrap] removing existing .venv because RECREATE_VENV=1"
  rm -rf .venv
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "[bootstrap] creating .venv"
  "$PY" -m venv .venv
fi

VPY="$ROOT/.venv/bin/python"

echo "[bootstrap] upgrading packaging tools"
"$VPY" -m pip install --upgrade pip setuptools wheel

# Keep a conservative NumPy/SciPy ABI on macOS. This avoids accidentally mixing
# global NumPy 2.x state with wheels compiled against NumPy 1.x while still
# satisfying fly-sniff's declared dependency floor.
echo "[bootstrap] installing stable scientific ABI"
"$VPY" -m pip install --upgrade --force-reinstall \
  'numpy>=1.26,<2' \
  'scipy>=1.12,<2'

echo "[bootstrap] installing fly-sniff and development tools"
"$VPY" -m pip install --upgrade -e '.[dev]'

echo "[bootstrap] validating scientific imports"
"$VPY" - <<'PY'
import sys

import numpy
import pandas
import pyarrow
import scipy

print("python:", sys.executable)
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("pandas:", pandas.__version__)
print("pyarrow:", pyarrow.__version__)
PY

echo "[bootstrap] checking required installed commands"
for name in \
  fly-sniff-doctor \
  fly-sniff-download-annotations \
  fly-sniff-download-weights \
  fly-sniff-staged-trace \
  fly-sniff-review-route \
  fly-sniff-e001-evidence; do
  path="$ROOT/.venv/bin/$name"
  if [[ ! -x "$path" ]]; then
    echo "ERROR: expected console script missing: $path" >&2
    exit 4
  fi
  echo "  $name -> $path"
done

"$ROOT/.venv/bin/fly-sniff-doctor"
"$ROOT/.venv/bin/ruff" check src tests
"$VPY" -m pytest -q

echo
echo "Mac E001 environment is healthy."
echo "Next command:"
echo "  bash scripts/run_e001_mac.sh"
