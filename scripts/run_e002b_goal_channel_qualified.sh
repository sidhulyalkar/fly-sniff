#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
RESULT="${RESULT:-results/e002/goal-channel-v1.json}"
QUALIFIED="${QUALIFIED:-results/e002/goal-channel-qualified-v1.json}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "E002b qualification requires a clean git worktree before the run" >&2
  git status --short >&2
  exit 2
fi

START_SHA="$(git rev-parse HEAD)"
START_BRANCH="$(git branch --show-current)"
echo "E002b qualification runtime: branch=$START_BRANCH sha=$START_SHA"

OUTPUT="$RESULT" bash scripts/run_e002b_goal_channel_probe.sh

if [[ "$(git rev-parse HEAD)" != "$START_SHA" ]]; then
  echo "git SHA changed during E002b run" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "git worktree changed during E002b run" >&2
  git status --short >&2
  exit 2
fi

$PYTHON -m fly_sniff.e002b_qualification \
  "$RESULT" \
  --expected-git-sha "$START_SHA" \
  --output "$QUALIFIED"

echo "E002b qualified seal: $QUALIFIED"
$PYTHON -c 'import hashlib,sys; p=sys.argv[1]; print("sha256", hashlib.sha256(open(p,"rb").read()).hexdigest())' "$QUALIFIED"
