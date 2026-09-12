# Auditable social and scientific replay

## What can be shared now

A transparent **development proxy**, with an audience-visible stochastic odor
plume, bilateral modeled antenna inputs, body-frame airflow, actual outgoing
turn commands, and recorded trajectories. This is an engineering demonstration,
not an odor-navigation result from the MaleCNS connectome. People are visual
markers; the room has one odor source, and people do not obstruct flow. This is
not CFD, human odor identification, measured neural activity, or a whole-fly simulation.

## Priority gates for a reputable MaleCNS demonstration

| Priority | Deliverable | Acceptance evidence |
| --- | --- | --- |
| 1 | Establish real-data structural identity | Regenerate E001 stages from exact annotation and weight files; retain checksums, config, body IDs, thresholded edges and per-stage audits. Missing routes are valid findings. |
| 2 | Establish a defensible sensor-to-motor model | Review injection/readout roles, laterality and unresolved signs; pass E002 perturbation, lesion, and deterministic-replay gates. Anatomy alone does not establish odor function. |
| 3 | Test the wiring hypothesis | Freeze the model, seeds, compute budget, baseline selection and comparison before evaluation. Run paired intact/rewired/lesioned models on the same plume. Report failures, effect sizes and uncertainty, not a selected successful seed. |
| 4 | Bind the scientific visualization | Record complete puffs and actual observation→decision pairs. Bind sparse modeled state to the graph fingerprint and source soma coordinates. Preserve missing-geometry counts and expose edge/display subsampling. |
| 5 | Publish the short clip and audit | Keep labels legible at phone size; show simulation time and end outcomes. Link to the model assumptions, code, recording digest, structural audit and cohort results. |

The proposed distinctive question is: **does this extracted wiring help this
explicit odor-navigation model, compared with matched wiring controls?** This is
a hypothesis to test, not an established novelty or result. A negative finding
can still make a useful demonstration. Qualification on a different task, such
as looming escape, does not qualify odor navigation.

## Runnable integrated checkout

Python 3.11+ is required. From this branch:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
fly-sniff-doctor
ruff check src tests
pytest -q
fly-sniff-record --output artifacts/showcase/run.json --seed 13013 --sim-seconds 45 --plume-points 600
fly-sniff-cinematic artifacts/showcase/run.json --output artifacts/showcase/scientific.mp4 --seconds 15 --fps 30
fly-sniff-cinematic artifacts/showcase/run.json --output artifacts/showcase/scientific.png
```

Install ffmpeg for MP4 (`brew install ffmpeg` on macOS); PNG and GIF need no
ffmpeg. Reinstall after changing branches because console-script metadata is
created at installation. `python -m fly_sniff.staged_trace --help` distinguishes
a missing installed command from a checkout that does not contain that module.

With the actual local MaleCNS tables:

```bash
fly-sniff-staged-trace data/raw/body-annotations-male-cns-v1.0.feather data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather --config configs/staged_route_v0.json --output data/cache/staged-route-reviewed --strict
```

Each stage produces nodes, edges, path provenance and a report; use
`fly-sniff-audit-trace <stage-directory> --output <audit.json>` to check its
persisted structure. Strict mode fails if any requested corridor is absent; it
does not establish that the candidate stages form one continuous functional route.
Use a fresh output directory when changing the list of stages.

For an explicitly reviewed **candidate** GraphBundle with signed edges, input
and readout roles, source soma positions, and dataset provenance:

```bash
fly-sniff-record --candidate-graph data/cache/reviewed-candidate --controllers random --output artifacts/showcase/candidate-run.json --plume-points 600
fly-sniff-export-neural-scene data/cache/reviewed-candidate --output artifacts/showcase/candidate-scene.json
fly-sniff-cinematic artifacts/showcase/candidate-run.json --neural-scene artifacts/showcase/candidate-scene.json --output artifacts/showcase/candidate.mp4
```

This explicit exploration path is always labeled **not a qualified result**.
Do not fill missing signs, functional roles or anatomical coordinates with guesses
just to make the command run. Without suitable real data, use the proxy's sensory
trace; no anatomical image is required. A checksum detects modification, not
whether the source's provenance or biological interpretation is valid.

## Repairs and visual semantics

- PR #10's inspected CI failed `I001` in `trace.py`; installation had succeeded.
- PR #12's inspected CI failed `B009`/`RUF046` in `neural_scene.py`.
- Those lint failures prevented tests from running. Local execution then exposed
  `make_seed_split` allocating an approximately 16 GB population array. Sampling
  integer indices plus one preserves the sampling procedure without that allocation.
- PR #11 owns the staged CLI; PR #10 alone cannot install it. The integration
  combines the staged and cinematic additions with PR #13's causal science fixes.
- CI builds a wheel and runs commands from outside the source tree. Ruff is pinned
  to the observed CI version rather than disabling its failing rules.
- The green density uses the same Gaussian mixture and cutoff as the simulator,
  evaluated on a raster with interpolation for display. Brightness uses a fixed
  0–10 model-unit scale (clipped). Sampled/incomplete frames are labeled explicitly.
- Sensors continue to sample the advancing plume after a fly stops; outgoing
  decisions become invalid and are omitted from the turn trace. The last frame
  also has no outgoing decision. Repeated render callbacks never execute dynamics.
- A timed reveal says **not found** if neither controller reached the source.
  The final frame is held for readability; simulation seconds remain visible.
- Anatomical projection uses uniform X/Z scaling. Straight edges represent graph
  connectivity, not neurite paths or a transmission movie. Only the strongest
  1,800 drawable edges are shown; displayed/total counts remain visible.
- Glow is sparse top-absolute **modeled state**, with positive/negative state
  colors; it is not recorded spikes, neurotransmitter sign, or silence for omitted
  cells. Scene integrity and graph identity are checked before rendering.

## Limits of this development verification

Synthetic fixtures verify code paths, threshold handling, scene binding and
installed CLIs. They are not MaleCNS evidence. No full real-data odor route or
cohort qualification was produced in this repair. The seed-13013 proxy episode
is a fixed development example, not a success-rate estimate or comparator victory
for the biological connectome.
