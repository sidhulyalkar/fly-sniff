# fly-sniff 🪰

**Can a fruit-fly connectome provide a useful algorithm for odor-source navigation?**

`fly-sniff` is a reproducible NeuroAI benchmark built around the 2026 MaleCNS whole-central-nervous-system connectome. The project tests a narrow, falsifiable claim: whether a connectome-constrained Drosophila navigation circuit can find the source of an intermittent turbulent odor plume, and whether its biological topology provides measurable benefit over matched rewired controls.

> **Important:** this project does not claim to simulate a complete biological fly. MaleCNS provides structural connectivity and annotations, not complete membrane dynamics, receptor kinetics, synaptic physiology, neuromodulatory state, or plasticity rules. Every modeled assumption is explicit and every public visualization distinguishes measured connectome structure from simulated neural dynamics.

## Public result we are building toward

> **We gave the newly mapped fruit-fly connectome a smell to follow. Then we scrambled its wiring. Can the real brain still find the source?**

The fast social wrapper is intentionally ridiculous: **Who Farted?** Six people stand in a room, one emits the hidden odor source, and the viewer can see the simulated smell plume. The controller cannot see the culprit or source coordinates. It receives only the same bilateral odor/wind observations used by the benchmark.

The final shareable comparison will replay the *same frozen plume* for:

1. **MaleCNS topology** — reviewed sensory/navigation/descending subgraph.
2. **Degree-preserving rewire** — same nodes and exact directed in/out degrees, topology disrupted.
3. **Classical cast-and-surge** — transparent engineering baseline.

A top-down room makes the causal comparison obvious, while a small fly-centered inset shows the plume in body coordinates with odor made visible for the audience. Proxy renders are permanently watermarked and cannot be used as MaleCNS evidence.

## Install

```bash
python -m venv .venv
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

Then render the room concept using development-only controllers:

```bash
fly-sniff-party --output artifacts/who-farted-proxy.mp4
```

The video is hard-labelled **DEVELOPMENT PROXY • NOT A MALECNS RESULT**. Its only purpose is to tune scene readability while the real E001/E002 MaleCNS circuit is qualified.

The original scientific development renderer and benchmark remain available:

```bash
fly-sniff-benchmark --episodes 64
fly-sniff-demo --output artifacts/fly-sniff-proxy.mp4
```

No final connectome claim is allowed from these runs.

## Start from the real MaleCNS resource

A small-first, token-free path downloads Janelia's public ~13 MB v1.0 annotation table:

```bash
fly-sniff-download-annotations
fly-sniff-offline-discover data/raw/body-annotations-male-cns-v1.0.feather
```

Then, with a neuPrint token, discover and extract candidate connectivity directly from `male-cns:v1.0`:

```bash
fly-sniff-extract-malecns \
  --patterns configs/circuit_discovery_v0.json \
  --extract-induced
```

The full public connection-weight table is roughly 1.1 GB, so the project does not download it by default. Exact upstream URLs and qualification rules live in [`docs/DATA_AUTHORITY.md`](docs/DATA_AUTHORITY.md).

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
- No full game engine is required for social v0. The first MP4 is generated directly from benchmark state with Matplotlib. A browser renderer can reuse the same scene contract later.

## Reproducibility rule

Development seeds are disposable. The final benchmark manifest is immutable and hashed before final evaluation. A final renderer may only display cohort metrics produced by a matching evaluation receipt.

## License

Code: MIT. MaleCNS data are external and retain their upstream license/citation requirements.
