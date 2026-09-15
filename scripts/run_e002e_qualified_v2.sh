#!/usr/bin/env bash
set -euo pipefail

source scripts/python_env.sh

BUNDLE=${1:-data/cache/steering-scaffold-v1}
RESULT=${2:-results/e002/pfl3-descending-steering-v2.json}
QUALIFIED=${3:-results/e002/pfl3-descending-steering-qualified-v1.json}

if [[ -n "$(git status --porcelain)" ]]; then
  echo "E002e qualification requires a clean git worktree" >&2
  git status --short >&2
  exit 2
fi

START_SHA="$(git rev-parse HEAD)"
echo "E002e v2 qualification runtime: branch=$(git branch --show-current) sha=$START_SHA"

bash scripts/run_e002e_confirmation_v2.sh "$BUNDLE" "$RESULT"

if [[ "$(git rev-parse HEAD)" != "$START_SHA" ]]; then
  echo "git SHA changed during E002e confirmation" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "git worktree changed during E002e confirmation" >&2
  git status --short >&2
  exit 2
fi

"$PYTHON_BIN" -m fly_sniff.e002e_qualification \
  "$RESULT" \
  --bundle "$BUNDLE" \
  --expected-git-sha "$START_SHA" \
  --output "$QUALIFIED"

echo "E002e qualified seal: $QUALIFIED"
"$PYTHON_BIN" -c 'import hashlib,sys; p=sys.argv[1]; print("sha256", hashlib.sha256(open(p,"rb").read()).hexdigest())' "$QUALIFIED"
