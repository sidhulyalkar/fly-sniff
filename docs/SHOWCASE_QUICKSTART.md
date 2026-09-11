# Showcase quickstart

This path gets the development demo running without making any MaleCNS claim.

## 1. Checkout the showcase branch

```bash
git clone https://github.com/sidhulyalkar/fly-sniff.git
cd fly-sniff
git fetch origin
git switch -c local/showcase origin/feat/social-showcase-v1
```

## 2. Create a Python 3.11+ environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[malecns,dev]'
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[malecns,dev]'
```

## 3. Check readiness

```bash
fly-sniff-doctor
```

MP4 output requires `ffmpeg` on PATH. If it is unavailable, use GIF output first.

## 4. Prove the controller plumbing before rendering

```bash
fly-sniff-choice --controller proxy --trials 100
fly-sniff-choice --controller random --trials 100
fly-sniff-benchmark --episodes 64
```

These are development checks only.

## 5. Render the phone-first clip

Production-style local render:

```bash
fly-sniff-showcase \
  --output artifacts/showcase/who-farted-4x5.mp4 \
  --seconds 15 \
  --fps 30
```

Fast GIF smoke render:

```bash
fly-sniff-showcase \
  --output artifacts/showcase/who-farted-4x5.gif \
  --seconds 6 \
  --fps 10
```

The canvas is exactly 1080x1350. The clip is permanently labelled `DEVELOPMENT PROXY • NOT A MALECNS RESULT`.

## 6. Begin MaleCNS discovery in parallel

The small public annotation table is the fastest biological starting point:

```bash
fly-sniff-download-annotations
fly-sniff-offline-discover \
  data/raw/body-annotations-male-cns-v1.0.feather \
  --patterns configs/circuit_discovery_v0.json
```

For live neuPrint extraction, obtain a read token from neuprint.janelia.org, set `NEUPRINT_TOKEN`, then run:

```bash
fly-sniff-extract-malecns \
  --patterns configs/circuit_discovery_v0.json \
  --extract-induced
```

The resulting discovery graph is not benchmark-qualified. Continue through the repository's E001/E002 review, trace, role-sealing, sign-provenance, and qualification gates before any `REAL BRAIN WIRING` public label is allowed.
