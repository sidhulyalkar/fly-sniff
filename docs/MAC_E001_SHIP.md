# Canonical Mac E001 evidence lane

This is the fastest scientifically defensible path from a clean checkout to an exact-body-ID MaleCNS evidence bundle.

The lane is intentionally conservative:

- it uses the preregistered staged E001 route rather than a giant ORN-to-DNa02 search;
- it seals raw-data and authority/config hashes before tracing;
- it preserves a failed structural gate as a result rather than tuning thresholds in place;
- it derives exact role IDs only after E001 passes;
- it binds the generated role review to the current role policy and literature authority;
- it emits a human-readable claim table and a machine-readable ship gate;
- it never promotes E001 structure into a functional-navigation claim.

## Branch

Run this lane from:

```bash
git fetch origin
git switch feat/e001-mac-ship-v1
git pull --ff-only origin feat/e001-mac-ship-v1
```

Do not run the canonical bundle from `feat/science-rigor-v1`. That branch contains useful earlier audit machinery, but the current integration branch starts from the newer task-optimized lane and includes the corrected hDeltaC literature boundary, PFN soma-side priority, PFL3 laterality caveat, restricted sensory-interface claim, and authority hashing.

## 1. Bootstrap the Mac environment

Python 3.11 is the supported local lane. The bootstrap script creates `.venv`, installs a conservative NumPy/SciPy ABI, installs the package in editable mode, checks all required console commands, runs Ruff, and runs the full test suite.

```bash
RECREATE_VENV=1 bash scripts/bootstrap_macos_e001.sh
```

If Python 3.11 is missing:

```bash
brew install python@3.11
```

You normally only need `RECREATE_VENV=1` on the first run or after a broken environment.

## 2. Run the complete E001 evidence pipeline

```bash
bash scripts/run_e001_mac.sh
```

The first run downloads the public MaleCNS v1.0 annotation table and the roughly 1.1 GB connection-weight table if they are missing. Later runs reuse those immutable local inputs and reseal their hashes.

Each run gets a unique directory:

```text
results/e001/<UTC timestamp>-<git sha>/
```

The script refuses to overwrite an earlier run.

## 3. Read these files in this order

### `SHIP_STATUS.md`

This is the first file to inspect. It says the maximum claim level this run supports.

### `evidence-pack/E001_BODY_ID_AUDIT.md`

This is the body-ID forensic review. For each selected role it reports the exact body ID, type, side metadata/evidence, structural-stage membership, local retained-edge context, evidence class, functional uncertainty, allowed wording, and forbidden wording.

### `evidence-pack/e001_evidence.json`

Machine-readable equivalent of the audit. Downstream tooling should read this rather than infer claim status from filenames or successful exit codes.

### `role-review-v1/role_review.json`

Exact candidate role IDs and side-evidence sources generated from the passing staged route.

### `staged-route-v1/staged_trace_report.json`

Per-stage structural results and preregistered primary gate status.

### `staged-route-v1/handoff_audit.json`

Exact body-ID continuity across FB5AB, hDeltaC, and PFL3 handoffs.

### `run-manifest.json`

Git SHA plus SHA-256 hashes and sizes of the raw connectome files, route config, role policy, and literature authority used by that run.

## What a zero exit code means

A successful run means only:

> A provenance-checked, exact-body-ID MaleCNS structural candidate passed the preregistered E001 structural/handoff gates and the role draft is internally consistent with the frozen policy.

It does **not** mean:

- the selected MaleCNS neurons were physiologically recorded performing the modeled computation;
- hDeltaC is experimentally established as the odor-gated wind integrator;
- the connectome model navigates the plume;
- the intact graph beats rewired controls;
- the final public claim is qualified.

`fly-sniff-e001-evidence` therefore keeps `training_promotion_ready=false` and `public_functional_result_ready=false` at E001 even when all structural checks pass.

## What happens if E001 fails

The run stops after the staged structural trace and writes a `SHIP_STATUS.md` with `DEVELOPMENT_ONLY` claim level. The partial/negative structural artifacts are kept.

Do **not** change `max_hops`, `min_weight`, fanout, role regexes, or handoff requirements merely to turn that run green. Any new structural hypothesis should be versioned as a new protocol/config and justified independently.

## Fast shipping sequence

The project should ship through five explicit gates.

| gate | evidence required | maximum public statement |
|---|---|---|
| E001 structure | exact MaleCNS body IDs, edges, hashes, staged/handoff audit, body-ID review | “We extracted a body-ID-resolved structural candidate.” |
| Candidate graph | human-reviewed node/edge inclusion + sign coverage + frozen roles | “We built an explicit connectome-constrained model candidate.” |
| Matched development | same optimization budget for intact, rewires, and lesion on development seeds | “The intact candidate did/did not outperform matched controls during development.” |
| Frozen final | immutable code/config/parameters/seeds + one-way held-out/OOD evaluation | “Under the preregistered benchmark, intact wiring performed X vs controls.” |
| Social render | video generated only from matching frozen recording/evaluation receipts | visual headline may summarize the measured result, with audit link |

The quickest reputable route is therefore **not** to spend more time polishing the social clip before E001 is regenerated. The visual renderer is already adequate to carry the result once the evidence exists.

## Recommended public framing before final qualification

Safe:

> We are testing whether a body-ID-resolved circuit extracted from the MaleCNS connectome provides useful structure for odor navigation compared with matched rewired controls.

Unsafe before the final gate:

> The fly connectome found the smell source.

The distinction is small on screen and enormous scientifically.
