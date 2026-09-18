# Olfactory computation MacBook runbook

This runbook is for local development and evidence acquisition on macOS. It does not authorize confirmatory O003/O004 execution.

## Fast path: one command

The normal local workflow is now one command:

```bash
./scripts/run_olfactory_first_light_mac.sh
```

That command:

1. verifies Python >=3.11 and reuses the existing editable virtual environment when possible;
2. runs the focused scientific regression suite;
3. validates the study, E001, and E002 without promoting blocked evidence;
4. verifies or clones the exact pinned DoOR source;
5. reuses an existing immutable E006 ingestion only after its receipt/artifact hashes are checked;
6. runs the complete E006 provenance, missingness, scale, metadata, geosmin, and mapping audit;
7. identifies performance-blind within-study development subsets;
8. emits one compact ZIP for review.

A shell exit of zero means the integrity checks executed successfully. The scientific audit can still return
`BLOCKED_METADATA_ADJUDICATION`; that is a valid scientific state.

## 1. Checkout

For ordinary work after this lane lands:

```bash
git fetch origin
git switch research/olfactory-computation-v0
git pull --ff-only
```

During review of the one-command audit branch:

```bash
git fetch origin
git switch feat/e006-one-command-audit-v1
git pull --ff-only origin feat/e006-one-command-audit-v1
```

For claim-bearing work, prefer an isolated clean worktree. The E006 audit records tracked modifications and
fails the scientific cleanliness gate when untracked files exist under `authority/`, `src/`, `tests/`,
`scripts/`, `.github/`, or `pyproject.toml`.

## 2. Python

Python 3.11+ is required. Apple Silicon example:

```bash
brew install python@3.12
PYTHON_BIN="$(brew --prefix python@3.12)/bin/python3.12" \
  ./scripts/run_olfactory_first_light_mac.sh
```

The first run creates `.venv-olfactory` and installs development dependencies. Later runs reuse the editable
environment for speed. Force an environment refresh only when dependencies change:

```bash
FLY_SNIFF_REFRESH_ENV=1 ./scripts/run_olfactory_first_light_mac.sh
```

## 3. Data locations

By default:

```text
~/fly-sniff-data/DoOR.data
~/fly-sniff-data/artifacts/e006-door-db323a496577/
~/fly-sniff-data/artifacts/e006-audit-db323a496577-<code-ref>/
~/fly-sniff-data/artifacts/e006-audit-db323a496577-<code-ref>-share.zip
```

Use another disk with:

```bash
FLY_SNIFF_DATA_DIR=/Volumes/YourDrive/fly-sniff-data \
  ./scripts/run_olfactory_first_light_mac.sh
```

## 4. What the E006 audit checks

The audit validates the canonical E006 receipt hash separately from the SHA-256 of the finished JSON file,
then verifies the content hashes of the long-form response table and mapping table.

It produces:

```text
e006-audit.json
SUMMARY.txt
study-coverage.csv
responding-unit-coverage.csv
odor-coverage.csv
study-response-scales.csv
study-metadata-joined.csv
geosmin-observations.csv
ambiguous-unit-mappings.json
candidate-development-subsets.json
```

The source metadata are read directly from the exact pinned DoOR tree at
`data/door_dataset_info.csv`. The audit does not infer that two studies are commensurable merely because
their metadata look similar.

## 5. Development subset policy

To accelerate development without cross-study normalization, v1 identifies within-study candidates using a
frozen performance-blind rule:

- electrophysiology;
- spike response data;
- at least 500 observed cells;
- at least 20 responding units;
- at least 25 odor names;
- non-empty concentration metadata.

Candidates are selected using source coverage and metadata only. No decoding score, model output, navigation
metric, or behavioral outcome is inspected.

A selected development subset is not an E006 global qualification and cannot be promoted into confirmatory
O003 evidence.

## 6. Scientific guardrails

The audit never authorizes:

- averaging raw responses across studies without an assay-comparability authority;
- treating source missingness as measured zero response;
- selecting among one-to-many receptor mappings using model performance;
- converting DoOR source metadata into undocumented dose-response curves;
- O003/O004 confirmatory execution.

E001, E002, and E006 may remain blocked while development diagnostics proceed.

## 7. One-command O002 development

After the E006 audit path is green, run the complete permitted development cycle with:

```bash
./scripts/run_olfactory_development_mac.sh
```

This reruns the fail-closed E006 audit at the current code identity and then executes only the frozen
performance-blind within-study O002 subset. It creates one combined bundle:

```text
~/fly-sniff-data/artifacts/olfactory-dev-cycle-<source>-<code>.zip
```

The O002 v1 experiment keeps the original DoOR responding-unit IDs as features. It does not collapse
ambiguous units to receptor identities. It excludes the SFR pseudo-odor, never imputes missing response
cells, and restricts multivariate geometry to odor rows observed for every selected responding unit.

Because the selected source contains one aggregate vector per odor, v1 does not claim held-out odor-identity
decoding. It reports:

- PCA explained variance and effective rank;
- leave-one-responding-unit-out stability of pairwise odor geometry;
- an explicitly exploratory chemical-class nearest-centroid probe with a fixed permutation null;
- unsupported analyses, including concentration generalization, valence decoding, cross-study
  generalization, receptor-specific interpretation, and topology effects, as blocked/not estimated.

These metrics are development diagnostics. They may inform the design of a later frozen experiment, but they
cannot be promoted directly into O003 confirmatory evidence.

## 8. Full qualification check

Before a claim-bearing artifact or merge:

```bash
source .venv-olfactory/bin/activate
python -m ruff check .
python -m pytest -q
git rev-parse HEAD
git status --short
```

A negative or blocked scientific result must not be turned green by loosening an evidence threshold.

## 9. Manual CLI use

The integrated command can also be run directly against an existing immutable ingestion:

```bash
source .venv-olfactory/bin/activate

fly-sniff-olfactory audit-e006 \
  "$HOME/fly-sniff-data/artifacts/e006-door-db323a496577" \
  --door-checkout "$HOME/fly-sniff-data/DoOR.data" \
  --output "$HOME/fly-sniff-data/artifacts/e006-audit-manual"
```

Use this only when debugging the audit. The Mac runner is the preferred routine workflow.

## 10. What to send back

Normally send only the generated:

```text
e006-audit-...-share.zip
```

If the command fails before creating the bundle, send the complete terminal traceback.

The ZIP contains enough information to distinguish source/provenance failures, metadata limitations, mapping
ambiguity, sparse coverage, and a legitimate scientific block without uploading the full DoOR tables.

## 11. Current claim boundary

Passing the command means the preregistration and source/audit machinery worked at that exact code/source
state. It does not mean E001/E002/E006 are qualified, O001 calibration passed, biological topology has an
advantage, or O003 may be run confirmatorily.


## 12. Receipt-driven O002 showcase

Once O002 v1/v2/v3 development artifacts exist, render the presentation layer without rerunning science:

    ./scripts/render_o002_showcase_mac.sh

By default the runner consumes the frozen `4d2bdd13bf06` O002 development artifacts. Override only when a new frozen O002 source identity has been reviewed:

    FLY_SNIFF_O002_SOURCE_REF=<12-char-source-ref> ./scripts/render_o002_showcase_mac.sh

When `ffmpeg` is installed, the runner also creates a 1080×1350 4:5 social MP4. Without `ffmpeg`, every static artifact and the deterministic review ZIP are still produced.

Outputs include:

- `o002-hero-4x5.png` — primary shareable result card;
- `o002-scientific-deep-dive.png` — full scientific evidence plate;
- `o002-class-recall.png` — class-wise recall view;
- `o002-next-stage-roadmap.png` — measured result → authority gates → O003/O004 roadmap;
- `o002-social.mp4` — optional 16 s social explainer;
- `o002-visual-receipt.json` — verified input lineage and output hashes;
- `o002-showcase.zip` — deterministic review bundle.

The visualization lane is downstream only. It verifies frozen receipt lineage and CSV hashes, and it never recomputes or selects scientific outcomes. Regenerating presentation outputs with `FLY_SNIFF_FORCE_SHOWCASE=1` may change layout/rendering bytes but must not change the scientific inputs or claim boundary.
