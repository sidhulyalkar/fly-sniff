#!/usr/bin/env bash
# Resolve the Python interpreter consistently for local science/demo scripts.
# Prefer an explicitly supplied interpreter, then the active virtualenv,
# then the repository-local .venv, followed by system python3/python.

if [[ -n "${PYTHON_BIN:-}" ]]; then
  if [[ "$PYTHON_BIN" == */* ]]; then
    [[ -x "$PYTHON_BIN" ]] || {
      echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
      return 127 2>/dev/null || exit 127
    }
  elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "PYTHON_BIN is not available on PATH: $PYTHON_BIN" >&2
    return 127 2>/dev/null || exit 127
  fi
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "No usable Python interpreter found. Activate the virtualenv or set PYTHON_BIN." >&2
  return 127 2>/dev/null || exit 127
fi

export PYTHON_BIN
