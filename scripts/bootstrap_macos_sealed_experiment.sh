#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXPECTED_BRANCH="${EXPECTED_BRANCH:-feat/sealed-graph-experiment-v1}"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" && "${ALLOW_OTHER_BRANCH:-0}" != "1" ]]; then
  echo "ERROR: expected branch '$EXPECTED_BRANCH', found '$CURRENT_BRANCH'." >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" && "${ALLOW_DIRTY:-0}" != "1" ]]; then
  echo "ERROR: working tree is dirty; bootstrap the exact checkout that will be sealed." >&2
  exit 3
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
  echo "ERROR: Python 3.11 was not found. Install it with: brew install python@3.11" >&2
  exit 4
fi

if [[ "${RECREATE_VENV:-0}" == "1" ]]; then
  rm -rf .venv
fi
if [[ ! -x .venv/bin/python ]]; then
  "$PY" -m venv .venv
fi

VPY="$ROOT/.venv/bin/python"

echo "===== SEALED EXPERIMENT BOOTSTRAP ====="
echo "repository: $ROOT"
echo "branch: $CURRENT_BRANCH"
echo "sha: $(git rev-parse HEAD)"
echo "python: $VPY"
"$VPY" --version

# Preserve the same conservative macOS scientific ABI used by the E001 lane.
"$VPY" -m pip install --upgrade pip setuptools wheel
"$VPY" -m pip install --upgrade --force-reinstall \
  'numpy>=1.26,<2' \
  'scipy>=1.12,<2'

# Editable install is intentional: console entry points are refreshed now, while
# imports during the later three experiment phases resolve to this exact checkout.
"$VPY" -m pip install -e '.[dev]'

REQUIRED_COMMANDS=(
  fly-sniff-e001-evidence
  fly-sniff-sign-authority
  fly-sniff-seed-plan
  fly-sniff-seal-candidate
  fly-sniff-verify-candidate
  fly-sniff-sealed-train
  fly-sniff-sealed-qualify
  fly-sniff-sealed-freeze-final
  fly-sniff-sealed-final
  fly-sniff-redteam-budget
  fly-sniff-redteam-cem-replay
  fly-sniff-redteam-rewires
  fly-sniff-redteam-training
)

for name in "${REQUIRED_COMMANDS[@]}"; do
  path="$ROOT/.venv/bin/$name"
  if [[ ! -x "$path" ]]; then
    echo "ERROR: expected sealed-experiment command missing after editable install: $path" >&2
    exit 5
  fi
  "$path" --help >/dev/null
  echo "  ok: $name"
done

"$VPY" - "$ROOT" <<'PY'
import pathlib
import sys

import fly_sniff

root = pathlib.Path(sys.argv[1]).resolve()
package = pathlib.Path(fly_sniff.__file__).resolve()
expected = root / "src" / "fly_sniff"
if expected not in package.parents and package.parent != expected:
    raise SystemExit(
        f"fly_sniff import is not bound to this checkout: package={package}, expected under={expected}"
    )
print("editable import:", package)
PY

bash -n scripts/run_sealed_experiment_mac.sh
"$ROOT/.venv/bin/ruff" check src tests
"$VPY" -m pytest -q

echo
echo "Sealed experiment environment is healthy."
echo "No navigation performance has been created."
echo "Next: run Phase 1 (seal only) with:"
echo "  BUNDLE=/path/to/sealed-graphbundle E001_RUN=/path/to/passing-e001-run bash scripts/run_sealed_experiment_mac.sh"
