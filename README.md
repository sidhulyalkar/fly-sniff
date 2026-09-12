# fly-sniff 🪰

**Can a fruit-fly connectome provide a useful algorithm for odor-source navigation?**

`fly-sniff` is a reproducible NeuroAI benchmark built around the 2026 MaleCNS whole-central-nervous-system connectome. The project tests a narrow, falsifiable claim: whether a connectome-constrained Drosophila navigation circuit can find the source of an intermittent turbulent odor plume, and whether its biological topology provides measurable benefit over matched rewired controls.

> **Important:** this project does not claim to simulate a complete biological fly. MaleCNS provides structural connectivity and annotations, not complete membrane dynamics, receptor kinetics, synaptic physiology, neuromodulatory state, or plasticity rules. Every modeled assumption is explicit and every public visualization distinguishes measured connectome structure from simulated neural dynamics.

## Try the public development showcase

The ridiculous wrapper is **WHO FARTED?** Six suspects, one simulated odor source, a visible turbulent plume, and two search trajectories running from the same starting state.

The showcase now follows a strict **record first, replay second** contract. The simulation writes an immutable, SHA-256-identified episode containing plume state, antenna signals, body-frame airflow, controller commands, trajectories, and outcome. The video only replays that record.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
fly-sniff-doctor

fly-sniff-record \
  --output artifacts/showcase/who-farted-run.json \
  --seed 13013 \
  --sim-seconds 45

fly-sniff-replay-showcase \
  artifacts/showcase/who-farted-run.json \
  --output artifacts/showcase/who-farted-4x5.mp4 \
  --seconds 15 \
  --fps 30
```

The social canvas is exactly **1080×1350 (4:5)**. The current public clip is permanently labelled **DEVELOPMENT PROXY • NOT A MALECNS RESULT**. The blue controller is a transparent biology-inspired proxy while the actual MaleCNS circuit is being qualified.

The fixed development seed currently produces a useful demo sanity check: the proxy reaches the source at **15.4 s**, while the random controller does not reach the source during the 45 s episode. That is a **single development episode**, not a cohort result and not MaleCNS evidence.

See [`docs/SHOWCASE_QUICKSTART.md`](docs/SHOWCASE_QUICKSTART.md) for the Mac-first setup, [`docs/RECORDED_SHOWCASE.md`](docs/RECORDED_SHOWCASE.md) for the replay contract, and [`docs/CIRCUIT_SHOWCASE_STORYBOARD.md`](docs/CIRCUIT_SHOWCASE_STORYBOARD.md) for the evidence-gated neural visualization plan.

## Public result we are building toward

> **We gave the newly mapped fruit-fly connectome a smell to follow. Then we scrambled its wiring. Can the real brain still find the source?**

The final shareable comparison will replay the *same frozen plume* for:

1. **MaleCNS topology** — reviewed sensory/navigation/descending subgraph.
2. **Degree-preserving rewire** — same nodes and exact directed in/out degrees, topology disrupted.
3. **Classical cast-and-surge** — transparent engineering baseline.

The controller cannot see the culprit or source coordinates. It receives only the same bilateral odor/wind observations used by the benchmark. A top-down room makes the causal comparison obvious. The sensory HUD exposes the modeled left/right antenna values, body-frame airflow, and exact steering command. Proxy renders are permanently watermarked and cannot be used as MaleCNS evidence.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

For live neuPrint extraction:

```bash
pip install -e '.[malecns,dev]'
export NEUPRINT_TOKEN='...'
```

## Fast development path

Before asking the connectome to cross a turbulent room, run the deliberately tiny E002A left/right steering test:

```bash
fly-sniff-choice --controller proxy --trials 100
fly-sniff-choice --controller random --trials 100
```

This is not a separate throwaway game. It uses the production `Controller.act(observation)` interface and asks one question: when odor is stronger on one antenna, does the controller commit a turn toward that side? Chance accuracy is 50%.

Then sanity-check the full development benchmark:

```bash
fly-sniff-benchmark --episodes 64
```

The older direct renderers remain available for debugging, but the recorded replay path above is the canonical public showcase.

## Start from the real MaleCNS resource

A small-first, token-free path downloads Janelia's public v1.0 annotation table:

```bash
fly-sniff-download-annotations
fly-sniff-offline-discover \
  data/raw/body-annotations-male-cns-v1.0.feather \
  --patterns configs/circuit_discovery_v0.json
```

Then, with a neuPrint token, discover and extract candidate connectivity directly from `male-cns:v1.0`:

```bash
fly-sniff-extract-malecns \
  --patterns configs/circuit_discovery_v0.json \
  --extract-induced
```

For full offline structural tracing, explicitly fetch the roughly 1.1 GB public connection-weight table:

```bash
fly-sniff-download-weights --yes-large-download
```

The preferred biological discovery path is staged rather than one giant ORN→DNa02 query. It asks independently whether the graph contains candidate corridors for olfactory context → goal circuitry, PFN wind input → goal circuitry, goal circuitry → PFL steering outputs, and PFL outputs → DNa02:

```bash
fly-sniff-staged-trace \
  data/raw/body-annotations-male-cns-v1.0.feather \
  data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  --config configs/staged_route_v0.json \
  --output data/cache/staged-route-v0
```

These are structural discovery results only. A candidate corridor is never automatically promoted to `REAL BRAIN WIRING`.

Exact upstream URLs and qualification rules live in [`docs/DATA_AUTHORITY.md`](docs/DATA_AUTHORITY.md).

## Primary benchmark claim

The first publishable success target is preregistered in [`docs/BENCHMARK_CONTRACT.md`](docs/BENCHMARK_CONTRACT.md):

- at least **70% source-finding success** across 1,000 held-out turbulent-plume episodes;
- `SPL(MaleCNS) - SPL(degree-preserving rewire) >= 0.10`;
- paired bootstrap 95% CI for the SPL difference excludes zero;
- at least **60% success** on a frozen out-of-distribution plume regime.

If the biological graph does not beat the controls, that is a valid result. We do not move the goalposts after seeing the frozen test.

## Research gates

- **E001 Data authority** — real v1.0 annotations/body IDs and reproducible connectivity extraction.
- **E002 Circuit sanity** — sensory injection propagates and bilateral steering readouts behave coherently.
- **E002A Sniff choice** — smallest controller-level left/right odor steering assay.
- **E003 Easy plume** — qualified graph exceeds random navigation.
- **E004 Turbulent plume** — source localization under intermittent evidence.
- **E005 Causal topology controls** — MaleCNS vs exact degree-preserving rewires.
- **E006 Freeze** — seal dynamics, roles, seeds, plume distributions, nulls.
- **E007 Final** — run once, issue receipt, render from that receipt.

See [`docs/CIRCUIT_CANDIDATES.md`](docs/CIRCUIT_CANDIDATES.md), [`docs/BENCHMARK_CONTRACT.md`](docs/BENCHMARK_CONTRACT.md), and [`docs/VISUALIZATION_GOAL.md`](docs/VISUALIZATION_GOAL.md).

## Reuse policy

We keep the social path deliberately light:

- **PomPy (MIT)** is a useful reference for puff-plume mathematics and API design.
- **FlyGym 2.x** is the future embodiment target once the circuit itself earns a result.
- Published navigation-model repositories are scientific references unless their licenses are compatible with this MIT project; GPL code is not copied into `fly-sniff`.
- No full game engine is required for social v0. The first MP4 is generated from recorded benchmark state with Matplotlib. A browser renderer can consume the same immutable recording later without reimplementing controller dynamics.

## Reproducibility rule

Development seeds are disposable. Public showcase videos replay hashed episode recordings. The final benchmark manifest is immutable and hashed before final evaluation, and a final renderer may only display cohort metrics produced by a matching evaluation receipt.

## License

Code: MIT. MaleCNS data are external and retain their upstream license/citation requirements.
