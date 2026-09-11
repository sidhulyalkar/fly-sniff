# fly-sniff 🪰

**Can a fruit-fly connectome provide a useful algorithm for odor-source navigation?**

`fly-sniff` is a reproducible NeuroAI benchmark built around the 2026 MaleCNS whole-central-nervous-system connectome. The project tests a narrow, falsifiable claim: whether a connectome-constrained Drosophila navigation circuit can find the source of an intermittent turbulent odor plume, and whether its biological topology provides measurable benefit over matched rewired controls.

> **Important:** this project does not claim to simulate a complete biological fly. MaleCNS provides structural connectivity and annotations, not complete membrane dynamics, receptor kinetics, synaptic physiology, neuromodulatory state, or plasticity rules. Every modeled assumption is explicit and every public visualization distinguishes measured connectome structure from simulated neural dynamics.

## Public result we are building toward

**FlyBrain Plume Hunt**

A single 20–30 second split-screen video shows the *same frozen turbulent plume* driving four controllers:

1. **MaleCNS topology** — extracted sensory/navigation/descending subgraph.
2. **Degree-preserving rewire** — same nodes and edge-count statistics, topology disrupted.
3. **Biologically inspired proxy** — transparent cast-and-surge baseline used while the real graph is being qualified.
4. **Classical plume controller** — non-neural engineering baseline.

The viewer sees the odor plume, fly trajectories, left/right antennal activity, navigation-population activity, steering output, distance-to-source, path efficiency, and final success/SPL. The visualization is not evidence by itself; it is a rendering of a frozen benchmark run.

## Primary benchmark claim

The first publishable success target is preregistered in [`docs/BENCHMARK_CONTRACT.md`](docs/BENCHMARK_CONTRACT.md):

- at least **70% source-finding success** across 1,000 held-out turbulent-plume episodes;
- `SPL(MaleCNS) - SPL(degree-preserving rewire) >= 0.10`;
- paired bootstrap 95% CI for the SPL difference excludes zero;
- at least **60% success** on a frozen out-of-distribution plume regime.

If the biological graph does not beat the controls, that is a valid result. We do not move the goalposts after seeing the frozen test.

## Data authority

The project targets the public Janelia **MaleCNS v1.0** resource (`male-cns:v1.0`). Raw connectome data remain external and are never silently vendored into the repository. See [`docs/DATA_AUTHORITY.md`](docs/DATA_AUTHORITY.md).

## Status

`v0`: establishing the benchmark, plume simulator, controller API, connectome extraction contract, matched rewiring, metrics, tests, and social-ready renderer.

## Reproducibility rule

Development seeds are disposable. The final benchmark manifest is immutable and hashed before final evaluation. A renderer may only display metrics produced by an evaluation receipt; it may not recompute or alter them for presentation.

## License

Code: MIT. MaleCNS data are external and retain their upstream license/citation requirements.
