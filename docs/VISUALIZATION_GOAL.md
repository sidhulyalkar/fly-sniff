# Visualization goal: make the experiment obvious before the caption

The social artifact should be understandable with the sound off in under three seconds.

## Public question

> **We gave the newly mapped fruit-fly connectome a smell to follow. Then we scrambled its wiring. Can the real brain still find the source?**

The comedic wrapper is **WHO FARTED?** Six stylized people stand in a room; exactly one person occupies the simulated odor-source location. The viewer sees the smell plume. The agent does not receive the culprit identity, source coordinates, or distance-to-source.

Scientific claim boundary: this is **odor-source localization**, not chemical identity recognition or person identification. `WHO FARTED?` is the visual metaphor, not a claim that the model recognizes a human-specific odor signature.

## Fast social cut

**Primary format:** 1080×1350 (4:5), 15–18 s, 30 fps. Also preserve a square-safe crop for X/Threads.

### 0–2 s — hook

Huge text:

> WHO FARTED?

Immediately show six suspects and a visible drifting odor plume.

Small subtitle:

> real fly wiring vs the same brain scrambled

### 2–12 s — the experiment is the joke

Show two synchronized rooms using the **identical frozen plume**:

- `REAL BRAIN WIRING`
- `SCRAMBLED WIRING`

Both start from the same state. Thick trails make the paths readable on a phone. A compact timer and source-distance indicator update live.

The culprit is not highlighted until the reveal. The odor plume remains visible to the audience with an always-legible qualifier:

> smell visible to you • source hidden from the fly

A small lower inset may show:

> FLY POV • ODOR MADE VISIBLE

This is a body-centered visualization of the modeled odor field, **not visual perception and not what a fly literally sees**. Show left/right antenna activity so the audience can connect smell to steering without reading a methods paragraph.

### 12–15 s — culprit reveal

Circle the source person and reveal the outcome generated from the actual trajectory.

Possible labels, selected only from the result:

- `CASE CLOSED 💨` — source reached.
- `FALSE ACCUSATION` — controller reaches a decoy if a future discrete accusation mechanic is added.
- `STILL SNIFFING...` — timeout / no source found.

Show elapsed time and path efficiency. Never invent a successful ending for a failed controller.

### 15–18 s — the scientific punchline

When a qualified MaleCNS result exists, finish with the cohort rather than only the hero clip:

- held-out plume count;
- MaleCNS success rate;
- rewired success rate;
- paired MaleCNS − rewire SPL delta and 95% CI;
- OOD success if space permits.

Positive ending only if supported:

> BIOLOGICAL WIRING HELPED

Null ending:

> SAME NEURONS. DIFFERENT WIRING. NO ADVANTAGE.

A clean null is better than moving the goalposts.

## Development cut

`fly-sniff-party` renders the scene now using development-only controllers. It is permanently labelled:

> DEVELOPMENT PROXY • NOT A MALECNS RESULT

This lets us tune composition, plume visibility, mobile readability, comedy, and animation while E001/E002 circuit qualification proceeds.

The final renderer must refuse unqualified graph bundles and must be receipt-driven exactly like the existing scientific renderer.

## No fake room physics

The current plume is a 2-D stochastic puff benchmark, not room CFD. Therefore social v0 should use an open room with decorative humans and no furniture/walls that appear to deflect odor. If future visuals include airflow-blocking obstacles, the plume model must first be upgraded to model those interactions.

## Why top-down first

A top-down comparison wins for v0 because the viewer can simultaneously see:

1. where the odor actually goes;
2. where both agents go;
3. that the plume is identical;
4. who the hidden source is after reveal;
5. whether real and scrambled wiring diverge.

A full first-person 3-D game adds camera, geometry, collision, asset, and renderer work while making the causal comparison less obvious. The fly-centered inset gives us the fun POV without paying that complexity tax.

## Renderer architecture

Scientific simulation remains authoritative Python code. Presentation is downstream:

```text
MaleCNS / rewire controller
        ↓
FlySniffEnv + TurbulentPlume
        ↓
recorded episode state / receipt
        ↓
┌──────────────────────┬────────────────────────┐
│ Matplotlib social v0 │ browser replay later   │
│ fastest MP4          │ Canvas / Phaser/WebGL  │
└──────────────────────┴────────────────────────┘
```

The browser renderer, if built, should replay saved scientific trajectories rather than reimplement controller dynamics in JavaScript.

## Scientific long-form cut

Keep the existing 24-second square result available for researchers. It can show `MALECNS CONNECTOME`, `DEGREE-PRESERVING REWIRE`, and `CLASSICAL PLUME SEARCH`, modeled ORN/descending channels, followed by the complete 1,000-episode cohort. The funny 4:5 cut earns attention; the square scientific cut earns scrutiny.


## O002 measured-data showcase

O002 now has a separate receipt-driven visualization lane. This is the primary public artifact while E001/E002 remain authority-gated.

The visual hierarchy is deliberately:

1. **hero 4:5 result card** — one-screen interpretation of measured DoOR response geometry;
2. **scientific deep dive** — geometry, confusion, class recall, coding decomposition, paired holdouts, and low-rank retention;
3. **class-recall panel** — compact answer to which source chemical classes are easier or harder to separate;
4. **motion explainer** — a 16 s social cut rendered from the same frozen outputs;
5. **next-stage roadmap** — explicitly separates demonstrated sensory structure from still-gated mechanism and navigation work.

Run the complete presentation pass with:

    ./scripts/render_o002_showcase_mac.sh

The renderer consumes the existing O002 v1/v2/v3 receipts and verifies their lineage plus the hashes of every CSV it plots. It does not recompute PCA, classification, holdout, or subspace metrics. It may change composition, typography, annotations, and animation only.

The primary O002 story is:

> Measured fly olfactory responses contain stable within-study chemical-class geometry. In this development analysis, direction-normalized population responses retain more class information than simple response-magnitude summaries, while erasing stable responding-unit identity weakens the signal.

This remains a **development result**. The visual system must never convert it into a receptor-specific, behavioral-valence, cross-study, or connectome-topology claim.

### Visual separation of evidence levels

The measured O002 artifact and the future connectome command-center view serve different purposes.

- O002 is the current **truth card**: measured sensory data and frozen development analyses.
- The dark command-center language is a **mechanism interface**: morphology, replay, and independent probes may be shown there only with explicit provenance and clock labels.
- An integrated behavior + morphology + probe display must never imply that visually synchronized panels are causal measurements of one another unless a qualified receipt actually establishes that relationship.
- Development composites must retain a large `DEVELOPMENT PROXY` or equivalent label.
- Claim-bearing MaleCNS vs rewire visuals remain blocked until their own structural and physiological gates pass.

The next-stage visual ladder is therefore:

    measured sensory geometry
            ↓
    E001 / E002 authority resolution
            ↓
    O003 odor-conflict + temporal mechanism
            ↓
    O004 plume evidence → action
            ↓
    qualified intact-topology vs matched-null replay

Presentation may preview later boxes. Evidence may not skip them.
