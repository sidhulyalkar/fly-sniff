# Olfactory computation MacBook runbook

This runbook is for local development and evidence acquisition on macOS. It does not authorize confirmatory O003/O004 execution.

## 1. Clone and enter the study branch

```bash
git clone https://github.com/sidhulyalkar/fly-sniff.git
cd fly-sniff
git fetch origin research/olfactory-computation-v0
git switch research/olfactory-computation-v0
```

If you already have the repository:

```bash
cd /path/to/fly-sniff
git fetch origin
git switch research/olfactory-computation-v0
git pull --ff-only origin research/olfactory-computation-v0
```

## 2. Python environment

Python 3.11+ is required. On Apple Silicon, Homebrew Python is a simple option:

```bash
brew install python@3.12
python3.12 -m venv .venv-olfactory
source .venv-olfactory/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Verify:

```bash
python --version
python -m pytest -q tests/test_olfactory_program.py tests/test_olfactory_geosmin.py tests/test_olfactory_structure.py tests/test_olfactory_door.py tests/test_olfactory_cli.py
python -m ruff check src/fly_sniff/olfactory_*.py tests/test_olfactory_*.py
```

## 3. Inspect current scientific readiness

```bash
fly-sniff-olfactory status
```

Expected at the current v0 stage: the command succeeds but reports the study as `blocked`, because unresolved evidence is an expected scientific state rather than a software failure.

Inspect E001 and E002 explicitly:

```bash
fly-sniff-olfactory validate-e001
fly-sniff-olfactory validate-e002
```

The expected v0 states are:

- E001: `development_evidence_not_qualified`
- E002: `blocked_incomplete_body_id_adjudication`

Do not manually edit these statuses merely to make the study appear ready.

## 4. One-command first light

The canonical Mac runner installs the editable package, runs focused tests, validates the study, clones the exact pinned DoOR source, and creates an E006 ingestion receipt.

```bash
chmod +x scripts/run_olfactory_first_light_mac.sh
./scripts/run_olfactory_first_light_mac.sh
```

By default it stores external data outside the repository:

```text
~/fly-sniff-data/DoOR.data
~/fly-sniff-data/artifacts/e006-door-db323a496577/
```

To keep data on another disk:

```bash
export FLY_SNIFF_DATA_DIR=/Volumes/YourDrive/fly-sniff-data
./scripts/run_olfactory_first_light_mac.sh
```

To choose a Python interpreter:

```bash
PYTHON_BIN=python3.12 ./scripts/run_olfactory_first_light_mac.sh
```

## 5. Manual E006 DoOR ingestion

If you want to run the steps yourself:

```bash
mkdir -p ~/fly-sniff-data
cd ~/fly-sniff-data
git clone https://github.com/ropensci/DoOR.data.git
cd DoOR.data
git checkout --detach db323a496577c4b4a72b5c2fcd1859e07521ffb5
git status --short
```

`git status --short` must be empty. Then return to fly-sniff and ingest:

```bash
cd /path/to/fly-sniff
source .venv-olfactory/bin/activate

fly-sniff-olfactory ingest-door \
  ~/fly-sniff-data/DoOR.data \
  --output ~/fly-sniff-data/artifacts/e006-door-first-light
```

The output directory contains:

```text
door-responses-long.csv
door-unit-mappings.json
door-e006-receipt.json
```

The receipt should report 78 responding units and preserves every study-specific observed/missing response without normalization or aggregation.

## 6. Inspect the geosmin evidence in the real E006 artifact

After ingestion:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

p = Path.home() / "fly-sniff-data/artifacts/e006-door-first-light/door-responses-long.csv"
df = pd.read_csv(p)
geosmin = df[(df.odor_name == "geosmin") & (df.response_status == "observed")]
print(geosmin[["responding_unit", "study_id", "raw_response"]].to_string(index=False))
PY
```

This is descriptive source inspection only. Do not average study columns or interpret their raw scales as directly comparable unless the later E006 adjudication explicitly allows it.

## 7. Evaluate source missingness and study coverage

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

p = Path.home() / "fly-sniff-data/artifacts/e006-door-first-light/door-responses-long.csv"
df = pd.read_csv(p)

print("responding units:", df.responding_unit.nunique())
print("study columns:", df.study_id.nunique())
print("odor names:", df.odor_name.nunique(dropna=True))
print("observed cells:", (df.response_status == "observed").sum())
print("missing cells:", (df.response_status == "missing").sum())
print("missing fraction:", (df.response_status == "missing").mean())

coverage = (
    df.assign(observed=df.response_status.eq("observed"))
      .groupby("study_id")["observed"]
      .agg(["sum", "count", "mean"])
      .sort_values("sum", ascending=False)
)
print(coverage.head(30).to_string())
PY
```

This is a useful E006 adjudication diagnostic because sparse assay coverage can otherwise create misleading cross-odor comparisons.

## 8. Run the entire repository test suite

Before trusting any scientific artifact produced from a new branch head:

```bash
python -m ruff check .
python -m pytest -q
```

Also record the exact source commit:

```bash
git rev-parse HEAD
git status --short
```

A claim-bearing artifact should be produced from a clean checkout.

## 9. What you should not run yet

Do not treat these as authorized flagship experiments yet:

- O003 confirmatory odor-conflict evaluation;
- O004 confirmatory plume/navigation evaluation;
- a final intact-vs-rewire p-value;
- an engineered-sensor result presented as validation of the fly mechanism.

They remain blocked until the evidence registry and a later frozen ExperimentSpec/Lock allow them.

## 10. What to send back after first light

The most useful artifacts/logs to share are:

```bash
fly-sniff-olfactory status
cat ~/fly-sniff-data/artifacts/e006-door-db323a496577/door-e006-receipt.json
git rev-parse HEAD
```

and, if anything fails, the complete traceback plus:

```bash
python --version
uname -m
git status --short
git -C ~/fly-sniff-data/DoOR.data rev-parse HEAD
```

Those outputs are enough to distinguish environment problems, provenance failures, parser failures, and real evidence-quality blockers.
