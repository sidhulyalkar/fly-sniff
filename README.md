# fly-sniff 🪰

**Can a fruit-fly connectome provide useful algorithms, and can we tell when the wiring itself matters?**

`fly-sniff` is a reproducible MaleCNS experimentation and showcase harness. Its primary scientific benchmark asks whether a connectome-constrained Drosophila navigation circuit can find the source of an intermittent turbulent odor plume, and whether its biological topology provides measurable benefit over matched rewired controls. The same execution stack is now being generalized so we can reproduce recognizable connectome demos such as stimulus-driven activity movies, looming reflexes, and closed-loop game control without hiding the engineered sensory and motor interfaces.

> **Important:** this project does not claim to simulate a complete biological fly. MaleCNS provides structural connectivity and annotations, not complete membrane dynamics, receptor kinetics, synaptic physiology, neuromodulatory state, or plasticity rules. Every modeled assumption is explicit and every public visualization distinguishes measured connectome structure from simulated neural dynamics.

## Public result we are building toward

> **We gave the newly mapped fruit-fly connectome a smell to follow. Then we scrambled its wiring. Can the real brain still find the source?**

The fast social wrapper is intentionally ridiculous: **Who Farted?** Six people stand in a room, one emits the hidden odor source, and the viewer can see the simulated smell plume. The controller cannot see the culprit or source coordinates. It receives only the same bilateral odor/wind observations used by the benchmark.

The final shareable comparison will replay the *same frozen plume* for:

1. **MaleCNS topology**: reviewed sensory/navigation/descending subgraph.
2. **Degree-preserving rewire**: same nodes and exact directed in/out degrees, topology disrupted.
3. **Classical cast-and-surge**: transparent engineering baseline.

A top-down room makes the causal comparison obvious, while a small fly-centered inset shows the plume in body coordinates with odor made visible for the audience. Proxy renders are permanently watermarked and cannot be used as MaleCNS evidence.

## One connectome core, many experiments

The repository is no longer restricted to a single odor-specific demo. The reusable architecture is:

```text
WORLD / RECORDED STIMULUS
          |
          v
  sensory adapter manifest
          |
          v
  named GraphBundle roles
          |
          v
  ConnectomeRuntime
          |
          v
  named neural readouts
          |
          v
   motor/output adapter
          |
          v
WORLD / VISUALIZATION
```

`ConnectomeRuntime` is deliberately ignorant of odor, pixels, games, and button meanings. An adapter injects named neural roles and requests named readouts. That separation lets us reproduce several popular MaleCNS experiment archetypes while keeping the biological graph distinct from engineering glue.

List the current showcase contracts:

```bash
fly-sniff-showcase --list
```

Audit whether a GraphBundle contains the roles needed by a specific experiment:

```bash
fly-sniff-showcase \
  --experiment odor-choice \
  --graph path/to/graph
```

The audit reports `interface_ready` separately from `male_cns_claim_allowed`. A candidate graph can be useful for development without being promoted into biological evidence.

### Deterministic connectome replay

Recorded sensory channels can now be replayed through the same modeled dynamics. A JSONL stimulus frame has the form:

```json
{"t": 0.0, "inputs": {"odor_left": 0.8, "odor_right": 0.2}}
```

Replay it with:

```bash
fly-sniff-replay path/to/graph stimulus.jsonl \
  --readout steer_left \
  --readout steer_right \
  --output artifacts/replay-trace.jsonl
```

The command hashes the stimulus and resulting trace and writes a receipt. This is the foundation for movie-driven neural visualizations and deterministic game-frame replays. Activity under a movie is **not** evidence that a fly perceives or understands the movie.

The full replication roadmap lives in [`docs/REPLICATION_LADDER.md`](docs/REPLICATION_LADDER.md). It progresses from stimulus replay to native looming reflexes, connectome cinema, closed-loop arcade control, the odor plume flagship, and finally a cross-world intact-vs-rewired topology tournament.

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

- **E001 Data authority**: real v1.0 annotations/body IDs and reproducible connectivity extraction.
- **E002 Circuit sanity**: sensory injection propagates and bilateral steering readouts behave coherently.
- **E002A Sniff choice**: smallest controller-level left/right odor steering assay.
- **E003 Easy plume**: qualified graph exceeds random navigation.
- **E004 Turbulent plume**: source localization under intermittent evidence.
- **E005 Causal topology controls**: MaleCNS vs exact degree-preserving rewires.
- **E006 Freeze**: seal dynamics, roles, seeds, plume distributions, nulls.
- **E007 Final**: run once, issue receipt, render from that receipt.

See [`docs/CIRCUIT_CANDIDATES.md`](docs/CIRCUIT_CANDIDATES.md), [`docs/BENCHMARK_CONTRACT.md`](docs/BENCHMARK_CONTRACT.md), [`docs/VISUALIZATION_GOAL.md`](docs/VISUALIZATION_GOAL.md), and [`docs/REPLICATION_LADDER.md`](docs/REPLICATION_LADDER.md).

## Reuse policy

We keep the social path deliberately light:

- **PomPy (MIT)** is a useful reference for puff-plume mathematics and API design.
- **FlyGym 2.x** is the future embodiment target once the circuit itself earns a result.
- Published navigation-model repositories are scientific references unless their licenses are compatible with this MIT project; GPL code is not copied into `fly-sniff`.
- Third-party games are adapters, not scientific dependencies. We start with legally clean minimal worlds and never redistribute proprietary assets.
- A browser renderer should replay saved scientific trajectories rather than silently implementing a second neural model in JavaScript.

## Reproducibility rule

Development seeds are disposable. The final benchmark manifest is immutable and hashed before final evaluation. A final renderer may only display cohort metrics produced by a matching evaluation receipt.

## License

Code: MIT. MaleCNS data are external and retain their upstream license/citation requirements.
