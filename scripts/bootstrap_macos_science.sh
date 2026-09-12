#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif command -v python3.11 >/dev/null 2>&1; then
  PY="$(command -v python3.11)"
elif [[ -x /opt/homebrew/bin/python3.11 ]]; then
  PY="/opt/homebrew/bin/python3.11"
elif [[ -x /usr/local/bin/python3.11 ]]; then
  PY="/usr/local/bin/python3.11"
else
  echo "ERROR: Python 3.11 was not found. Install it with: brew install python@3.11" >&2
  exit 2
fi

echo "[bootstrap] repository: $ROOT"
echo "[bootstrap] base Python: $PY"
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

# Force the science-v1 ABI lane before editable installation. This repairs the
# common macOS failure where a global SciPy compiled against NumPy 1.x is used
# alongside NumPy 2.x.
echo "[bootstrap] installing stable scientific ABI"
"$VPY" -m pip install --upgrade --force-reinstall \
  'numpy>=1.26,<2' \
  'scipy>=1.12,<2'

echo "[bootstrap] installing fly-sniff editable environment"
"$VPY" -m pip install --upgrade -e '.[malecns,dev]'

echo "[bootstrap] validating import ABI"
"$VPY" - <<'PY'
import sys
import numpy
import scipy
import pandas
import pyarrow

print("python:", sys.executable)
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("pandas:", pandas.__version__)
print("pyarrow:", pyarrow.__version__)
PY

echo "[bootstrap] checking console scripts"
for name in fly-sniff-doctor fly-sniff-trace fly-sniff-audit-trace fly-sniff-science-handoff; do
  path="$ROOT/.venv/bin/$name"
  if [[ ! -x "$path" ]]; then
    echo "ERROR: expected console script missing: $path" >&2
    exit 3
  fi
  echo "  $name -> $path"
done

echo "[bootstrap] doctor"
"$ROOT/.venv/bin/fly-sniff-doctor"

echo "[bootstrap] ruff"
"$ROOT/.venv/bin/ruff" check src tests

echo "[bootstrap] pytest"
"$VPY" -m pytest -q

echo
echo "Science environment is healthy."
echo "Use explicit .venv paths to avoid shell PATH ambiguity:"
echo "  .venv/bin/fly-sniff-trace ..."
echo "  .venv/bin/fly-sniff-audit-trace ..."
echo "  .venv/bin/fly-sniff-science-handoff ..."
