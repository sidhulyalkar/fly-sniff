# Showcase quickstart

This is the shortest reproducible path to the public **WHO FARTED?** development demo.

The current showcase is intentionally a **development proxy**, not a qualified MaleCNS result. The recording and renderer permanently display `DEVELOPMENT PROXY • NOT A MALECNS RESULT` so visual iteration can happen in public without smuggling an unfinished biological claim into the video.

## Recommended local setup: macOS

Homebrew keeps Python and ffmpeg isolated from the system Python:

```bash
brew install python@3.11 ffmpeg

git clone https://github.com/sidhulyalkar/fly-sniff.git
cd fly-sniff
git fetch origin
git switch --track origin/feat/recorded-sensory-replay-v1

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
fly-sniff-doctor
```

If the repository already exists locally, replace the clone step with:

```bash
cd /path/to/fly-sniff
git fetch origin
git switch feat/recorded-sensory-replay-v1
git pull --ff-only
```

## Record first, render second

The social renderer does not run the simulation. First create the authoritative episode record:

```bash
fly-sniff-record \
  --output artifacts/showcase/who-farted-run.json \
  --seed 13013 \
  --sim-seconds 45
```

Then render that exact hashed episode:

```bash
fly-sniff-replay-showcase \
  artifacts/showcase/who-farted-run.json \
  --output artifacts/showcase/who-farted-4x5.mp4 \
  --seconds 15 \
  --fps 30

open artifacts/showcase/who-farted-4x5.mp4
```

For a fast preview:

```bash
fly-sniff-replay-showcase \
  artifacts/showcase/who-farted-run.json \
  --output artifacts/showcase/who-farted-4x5.gif \
  --seconds 6 \
  --fps 10
```

The JSON episode contains the exact plume samples, fly pose, modeled left/right antenna signals, body-frame airflow, controller command, diagnostics, success state, and a SHA-256 identity. The replay loader refuses a modified episode whose hash does not match.

## What the viewer is seeing

- six visual suspects occupy the benchmark room;
- exactly one suspect is located at the simulated odor-source coordinate;
- green particles visualize the modeled stochastic odor plume for the audience;
- the controllers do **not** receive culprit identity, source position, distance-to-source, or the viewer-visible plume image;
- both strategies start from the same state and remain synchronized to the same seeded exogenous plume;
- the blue path is the transparent biology-inspired development proxy;
- the pink path is the random control;
- the sensory panel exposes the proxy's actual modeled left/right antenna values, body-frame airflow, search mode, and exact steering command;
- the culprit reveal comes from benchmark ground truth and every displayed trajectory comes from the hashed episode recording.

## Sanity-check the controller plumbing

```bash
fly-sniff-choice --controller proxy --trials 100
fly-sniff-choice --controller random --trials 100
fly-sniff-benchmark --episodes 64
pytest -q
```

The forced-choice assay has a 50% chance baseline and uses the same production `Controller.act(observation)` interface used by the plume benchmark.

## Start the actual MaleCNS lane in parallel

```bash
python -m pip install -e '.[malecns,dev]'
fly-sniff-download-annotations
fly-sniff-offline-discover \
  data/raw/body-annotations-male-cns-v1.0.feather \
  --patterns configs/circuit_discovery_v0.json
```

For live neuPrint extraction, set a token in the shell:

```bash
export NEUPRINT_TOKEN='YOUR_TOKEN'
fly-sniff-extract-malecns \
  --patterns configs/circuit_discovery_v0.json \
  --extract-induced
```

For the complete public connection-weight table:

```bash
fly-sniff-download-weights --yes-large-download
```

Then trace candidate structural corridors with `fly-sniff-trace` and continue through E001/E002 review, bilateral role sealing, sign provenance, qualification, lesion checks, and deterministic replay. A discovery graph is never automatically promoted to `REAL BRAIN WIRING`.

## CI artifact

The `demo-pack` GitHub Actions workflow installs ffmpeg, creates one authoritative JSON recording, and renders both the fast GIF and 15 s 1080×1350 MP4 from that exact recording. The generated artifact is therefore reproducible from a clean checkout rather than being a hand-edited hero animation.

See `docs/RECORDED_SHOWCASE.md` for the recording contract and the sensory-routing claim boundary.
