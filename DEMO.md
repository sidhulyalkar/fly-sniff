# fly-sniff demo

This is the fastest auditable way to show the project while the MaleCNS circuit is still being qualified.

## Build the development demo

```bash
python -m pip install -e ".[dev]"
fly-sniff-demo-pack \
  --output-dir artifacts/demo \
  --seconds 6 \
  --fps 10 \
  --trials 100 \
  --format gif
```

The pack contains:

- `who-farted-proxy.gif` — 1080×1080 social/demo animation.
- `choice-proxy.json` — deterministic two-choice plumbing sanity check.
- `choice-random.json` — random-control sanity check on the same seeds.
- `demo-manifest.json` — claim status, source-head provenance, render settings, and summary metrics.

The same pack is built from a clean checkout by `.github/workflows/demo.yml` and uploaded as the `who-farted-development-demo` GitHub Actions artifact.

## What this demo does and does not claim

The development animation is intentionally stamped:

> **DEVELOPMENT PROXY • NOT A MALECNS RESULT**

It demonstrates the environment, plume visualization, controller interface, controls, rendering, evaluation plumbing, and final reveal. It does **not** claim that the MaleCNS connectome has localized the source yet.

`WHO FARTED?` is the visual joke. The measured task is simulated odor-source localization. The controller is not chemically identifying a person.

The two-choice development sanity check uses pure left/right forced-choice accuracy, so chance is exactly 50%. Turn magnitude is reported separately as commitment rate.

## Upgrade path to the real result

The public scene should not need to be rebuilt when E001 qualifies. The scientific upgrade is intentionally a controller/data-authority swap:

1. derive and seal exact MaleCNS v1.0 body IDs from pinned upstream evidence;
2. construct and qualify the candidate graph;
3. run the same two-choice interface on the qualified graph;
4. freeze the evaluation manifest and degree-preserving rewire;
5. render with `fly-sniff-party-final`, which requires the sealed circuit, manifest, and evaluation receipt.

Until those gates pass, use only the development demo for sharing.
