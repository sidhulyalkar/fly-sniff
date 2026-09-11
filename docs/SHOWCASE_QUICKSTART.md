# Showcase quickstart

This is the shortest reproducible path to the public **WHO FARTED?** development demo.

The current showcase is intentionally a **development proxy**, not a qualified MaleCNS result. The renderer permanently displays `DEVELOPMENT PROXY • NOT A MALECNS RESULT` so visual iteration can happen in public without smuggling an unfinished biological claim into the video.

## Run it

```bash
git clone https://github.com/sidhulyalkar/fly-sniff.git
cd fly-sniff
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
fly-sniff-doctor
fly-sniff-showcase --output artifacts/showcase/who-farted-4x5.mp4 --seconds 15 --fps 30
```

Windows PowerShell activation:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
fly-sniff-doctor
fly-sniff-showcase --output artifacts/showcase/who-farted-4x5.mp4 --seconds 15 --fps 30
```

If `ffmpeg` is not available, render a GIF immediately:

```bash
fly-sniff-showcase --output artifacts/showcase/who-farted-4x5.gif --seconds 6 --fps 10
```

## What the viewer is seeing

- six visual suspects occupy the benchmark room;
- exactly one suspect is located at the simulated odor-source coordinate;
- green particles visualize the modeled stochastic odor plume for the audience;
- the controllers do **not** receive culprit identity, source position, or distance-to-source;
- both strategies start from the same state and run against the same seeded plume;
- the blue path is the transparent biology-inspired development proxy;
- the pink path is the random control;
- the lower panel is a body-centered visualization of modeled odor around the agent, not literal fly vision;
- the culprit reveal is derived from benchmark ground truth and the displayed paths come from the actual run.

## Sanity-check the controller plumbing

```bash
fly-sniff-choice --controller proxy --trials 100
fly-sniff-choice --controller random --trials 100
fly-sniff-benchmark --episodes 64
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

For the complete public connection-weight table:

```bash
fly-sniff-download-weights --yes-large-download
```

Then trace candidate structural corridors with `fly-sniff-trace` and continue through E001/E002 review, bilateral role sealing, sign provenance, qualification, lesion checks, and deterministic replay. A discovery graph is never automatically promoted to `REAL BRAIN WIRING`.

## CI artifact

The `demo-pack` GitHub Actions workflow renders both a fast GIF preview and a 15 s, 1080×1350 MP4. The generated artifact is therefore reproducible from a clean checkout rather than being a hand-edited hero animation.
