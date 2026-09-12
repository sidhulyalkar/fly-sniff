#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXPECTED_BRANCH="${EXPECTED_BRANCH:-feat/e001-mac-ship-v1}"
CURRENT_BRANCH="$(git branch --show-current)"
GIT_SHA="$(git rev-parse HEAD)"
SHORT_SHA="$(git rev-parse --short=10 HEAD)"

if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" && "${ALLOW_OTHER_BRANCH:-0}" != "1" ]]; then
  echo "ERROR: expected branch '$EXPECTED_BRANCH', found '$CURRENT_BRANCH'." >&2
  echo "Switch first with: git switch $EXPECTED_BRANCH" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain)" && "${ALLOW_DIRTY:-0}" != "1" ]]; then
  echo "ERROR: working tree is not clean." >&2
  echo "Commit/stash source changes before producing an evidence bundle." >&2
  echo "Set ALLOW_DIRTY=1 only for explicitly exploratory runs." >&2
  git status --short >&2
  exit 3
fi

VPY="$ROOT/.venv/bin/python"
ANNOTATIONS="${ANNOTATIONS:-data/raw/body-annotations-male-cns-v1.0.feather}"
WEIGHTS="${WEIGHTS:-data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather}"
STAGED_CONFIG="${STAGED_CONFIG:-configs/staged_route_v1.json}"
ROLE_POLICY="${ROLE_POLICY:-configs/role_review_v1.json}"
LITERATURE_AUTHORITY="${LITERATURE_AUTHORITY:-authority/olfactory-navigation-literature-v1.json}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-$SHORT_SHA}"
RUN_DIR="${RUN_DIR:-results/e001/$RUN_TAG}"
STAGED_ROOT="$RUN_DIR/staged-route-v1"
ROLE_REVIEW_ROOT="$RUN_DIR/role-review-v1"
EVIDENCE_ROOT="$RUN_DIR/evidence-pack"

for executable in \
  "$VPY" \
  "$ROOT/.venv/bin/fly-sniff-download-annotations" \
  "$ROOT/.venv/bin/fly-sniff-download-weights" \
  "$ROOT/.venv/bin/fly-sniff-staged-trace" \
  "$ROOT/.venv/bin/fly-sniff-review-route" \
  "$ROOT/.venv/bin/fly-sniff-e001-evidence"; do
  if [[ ! -x "$executable" ]]; then
    echo "ERROR: missing environment executable: $executable" >&2
    echo "Run: RECREATE_VENV=1 bash scripts/bootstrap_macos_e001.sh" >&2
    exit 4
  fi
done

if [[ -e "$RUN_DIR" ]]; then
  echo "ERROR: run directory already exists: $RUN_DIR" >&2
  echo "Choose a new RUN_TAG so evidence from separate runs is never overwritten." >&2
  exit 5
fi
mkdir -p "$RUN_DIR"

if [[ ! -f "$ANNOTATIONS" ]]; then
  echo "===== DOWNLOAD MALECNS ANNOTATIONS ====="
  "$ROOT/.venv/bin/fly-sniff-download-annotations" --output "$ANNOTATIONS" \
    | tee "$RUN_DIR/download-annotations.json"
fi
if [[ ! -f "$WEIGHTS" ]]; then
  echo "===== DOWNLOAD MALECNS CONNECTION WEIGHTS (~1.1 GB) ====="
  "$ROOT/.venv/bin/fly-sniff-download-weights" \
    --output "$WEIGHTS" \
    --yes-large-download \
    | tee "$RUN_DIR/download-weights.json"
fi

for path in "$ANNOTATIONS" "$WEIGHTS" "$STAGED_CONFIG" "$ROLE_POLICY" "$LITERATURE_AUTHORITY"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required input is missing: $path" >&2
    exit 6
  fi
done

mkdir -p "$RUN_DIR/frozen-inputs"
cp "$STAGED_CONFIG" "$RUN_DIR/frozen-inputs/staged_route_v1.json"
cp "$ROLE_POLICY" "$RUN_DIR/frozen-inputs/role_review_v1.json"
cp "$LITERATURE_AUTHORITY" "$RUN_DIR/frozen-inputs/olfactory-navigation-literature-v1.json"

export E001_ROOT="$ROOT"
export E001_RUN_DIR="$RUN_DIR"
export E001_BRANCH="$CURRENT_BRANCH"
export E001_GIT_SHA="$GIT_SHA"
export E001_ANNOTATIONS="$ANNOTATIONS"
export E001_WEIGHTS="$WEIGHTS"
export E001_STAGED_CONFIG="$STAGED_CONFIG"
export E001_ROLE_POLICY="$ROLE_POLICY"
export E001_LITERATURE_AUTHORITY="$LITERATURE_AUTHORITY"

"$VPY" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

paths = {
    "annotations": os.environ["E001_ANNOTATIONS"],
    "weights": os.environ["E001_WEIGHTS"],
    "staged_config": os.environ["E001_STAGED_CONFIG"],
    "role_policy": os.environ["E001_ROLE_POLICY"],
    "literature_authority": os.environ["E001_LITERATURE_AUTHORITY"],
}
manifest = {
    "protocol": "e001-mac-run-manifest-v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "branch": os.environ["E001_BRANCH"],
    "git_sha": os.environ["E001_GIT_SHA"],
    "inputs": {
        key: {
            "path": value,
            "size_bytes": Path(value).stat().st_size,
            "sha256": sha256(value),
        }
        for key, value in paths.items()
    },
}
out = Path(os.environ["E001_RUN_DIR"]) / "run-manifest.json"
out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

echo "===== E001 PREREGISTERED STAGED STRUCTURAL TRACE ====="
set +e
"$ROOT/.venv/bin/fly-sniff-staged-trace" \
  "$ANNOTATIONS" \
  "$WEIGHTS" \
  --config "$STAGED_CONFIG" \
  --output "$STAGED_ROOT" \
  --strict \
  > >(tee "$RUN_DIR/staged-trace.stdout.json") \
  2> >(tee "$RUN_DIR/staged-trace.stderr.log" >&2)
STAGED_EXIT=$?
set -e
printf '%s\n' "$STAGED_EXIT" > "$RUN_DIR/staged-trace.exit-code.txt"

if [[ "$STAGED_EXIT" != "0" ]]; then
  cat > "$RUN_DIR/SHIP_STATUS.md" <<EOF
# E001 ship status

**Claim level:** DEVELOPMENT_ONLY

The preregistered structural gate exited with code \`$STAGED_EXIT\`.
This run is preserved as a negative or execution result. Do not tune the
frozen route thresholds in place and do not proceed to role promotion,
task optimization, or a public MaleCNS functional claim from this run.
EOF
  echo
  echo "E001 structural gate did not pass. Artifacts preserved at: $RUN_DIR" >&2
  echo "This is not automatically an engineering failure; inspect staged_trace_report.json and stderr." >&2
  exit "$STAGED_EXIT"
fi

echo "===== EXACT BODY-ID ROLE REVIEW ====="
"$ROOT/.venv/bin/fly-sniff-review-route" \
  "$STAGED_ROOT" \
  --policy "$ROLE_POLICY" \
  --output "$ROLE_REVIEW_ROOT" \
  > >(tee "$RUN_DIR/role-review.stdout.json") \
  2> >(tee "$RUN_DIR/role-review.stderr.log" >&2)

echo "===== CLAIM-BOUNDARY EVIDENCE PACK ====="
"$ROOT/.venv/bin/fly-sniff-e001-evidence" \
  "$STAGED_ROOT" \
  "$ROLE_REVIEW_ROOT" \
  --policy "$ROLE_POLICY" \
  --authority "$LITERATURE_AUTHORITY" \
  --output "$EVIDENCE_ROOT" \
  > >(tee "$RUN_DIR/evidence-pack.stdout.json") \
  2> >(tee "$RUN_DIR/evidence-pack.stderr.log" >&2)

"$VPY" - <<PY
import json
from pathlib import Path

run_dir = Path("$RUN_DIR")
pack = json.loads((run_dir / "evidence-pack" / "e001_evidence.json").read_text())
gate = pack["ship_gate"]
lines = [
    "# E001 ship status",
    "",
    f"**Claim level:** {pack['claim_level']}",
    f"**Structural evidence pack ready:** {gate['structural_evidence_pack_ready']}",
    f"**Training promotion ready:** {gate['training_promotion_ready']}",
    f"**Public functional result ready:** {gate['public_functional_result_ready']}",
    "",
    "## Maximum allowed wording",
    "",
    f"> {gate['max_public_wording']}",
    "",
    "## Next blockers",
    "",
]
lines.extend(f"- {item}" for item in gate["remaining_blockers"])
(run_dir / "SHIP_STATUS.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

mkdir -p results/e001
printf '%s\n' "$RUN_DIR" > results/e001/LATEST

echo
echo "===== COMPLETE ====="
echo "Run directory: $RUN_DIR"
echo "Read first:     $RUN_DIR/SHIP_STATUS.md"
echo "Body-ID audit:  $EVIDENCE_ROOT/E001_BODY_ID_AUDIT.md"
echo "Machine pack:   $EVIDENCE_ROOT/e001_evidence.json"
echo "Role review:    $ROLE_REVIEW_ROOT/role_review.json"
echo
echo "Do not start task optimization merely because this script exits 0."
echo "A zero exit establishes a provenance-checked structural candidate pack, not a functional circuit."
