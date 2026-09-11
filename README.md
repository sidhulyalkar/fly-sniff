# fly-sniff 🪰

**Can a fruit-fly connectome provide a useful algorithm for odor-source navigation?**

`fly-sniff` is a reproducible NeuroAI benchmark built around the 2026 MaleCNS whole-central-nervous-system connectome. The project tests a narrow, falsifiable claim: whether a connectome-constrained Drosophila navigation circuit can find the source of an intermittent turbulent odor plume, and whether its biological topology provides measurable benefit over matched rewired controls.

> **Important:** this project does not claim to simulate a complete biological fly. MaleCNS provides structural connectivity and annotations, not complete membrane dynamics, receptor kinetics, synaptic physiology, neuromodulatory state, or plasticity rules. Every modeled assumption is explicit and every public visualization distinguishes measured connectome structure from simulated neural dynamics.

## Public result we are building toward

**FlyBrain Plume Hunt**

A single 24–30 second split-screen video shows the *same frozen turbulent plume* driving matched controllers:

1. **MaleCNS topology** — reviewed sensory/navigation/descending subgraph.
2. **Degree-preserving rewire** — same nodes and exact directed in/out degrees, topology disrupted.
3. **Classical cast-and-surge** — transparent engineering baseline.
4. **Matched artificial recurrent controller** — added before final freeze.

The viewer sees the odor plume, fly trajectories, left/right antennal activity, selected navigation-population activity, steering output, distance-to-source, path efficiency, and final success/SPL. The final seconds switch from a visually intuitive episode to statistics across the complete held-out cohort.

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

## Run the development system now

The current runnable controller is intentionally labelled as a **biology-inspired proxy**, not MaleCNS:

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
- **E003 Easy plume** — qualified graph exceeds random navigation.
- **E004 Turbulent plume** — source localization under intermittent evidence.
- **E005 Causal topology controls** — MaleCNS vs exact degree-preserving rewires.
- **E006 Freeze** — seal dynamics, roles, seeds, plume distributions, nulls.
- **E007 Final** — run once, issue receipt, render from that receipt.

See [`docs/CIRCUIT_CANDIDATES.md`](docs/CIRCUIT_CANDIDATES.md), [`docs/BENCHMARK_CONTRACT.md`](docs/BENCHMARK_CONTRACT.md), and [`docs/VISUALIZATION_GOAL.md`](docs/VISUALIZATION_GOAL.md).

## Reproducibility rule

Development seeds are disposable. The final benchmark manifest is immutable and hashed before final evaluation. A final renderer may only display cohort metrics produced by a matching evaluation receipt.

## License

Code: MIT. MaleCNS data are external and retain their upstream license/citation requirements.
