# Visualization goal: the 24-second result

The social artifact should make the experiment understandable with the sound off in under five seconds.

## Final composition

**Format:** 1080×1080, 24–30 s, 30 fps. Designed to survive X/Threads compression.

### 0–3 s — hook

Large text:

> CAN A FRUIT-FLY CONNECTOME FIND AN INVISIBLE ODOR SOURCE?

Immediately show a turbulent plume drifting downwind. The source is marked for the viewer but hidden from the controllers.

### 3–16 s — paired experiment

Show four synchronized arenas using the **identical plume seed**:

- `MALECNS CONNECTOME`
- `DEGREE-PRESERVING REWIRE`
- `CLASSICAL CONTROLLER`
- `MATCHED ARTIFICIAL RNN` (after implemented)

Every pane shows the current position and path trace. Use one shared plume visualization so the causal comparison is visually obvious.

Along the bottom, show only interpretable channels:

- left ORN activity;
- right ORN activity;
- selected central-complex population activity;
- left descending steering activity;
- right descending steering activity.

Never call modeled activity a recording.

### 16–22 s — cohort result

Transition from the pretty episode to the actual experiment:

- `1,000 HELD-OUT PLUMES`
- success rate for each controller;
- mean SPL for each controller;
- paired MaleCNS − rewire SPL delta and 95% CI;
- OOD success.

A single episode earns attention. The cohort earns credibility.

### 22–24 s — claim

The final card is generated from the frozen receipt. Example *only if supported*:

> BIOLOGICAL WIRING IMPROVED PATH EFFICIENCY BY +0.14 SPL
> 95% paired bootstrap CI [+0.09, +0.19]

Below it:

`github.com/sidhulyalkar/fly-sniff`

If the primary result is negative, say so clearly and show the failure. A surprising null is better than a decorative overclaim.

## Development visualization

`fly-sniff-demo` currently renders a deliberately watermarked **biology-inspired proxy** alongside classical and random baselines. This exists to validate the entire visual pipeline before MaleCNS circuit qualification. It is forbidden to crop away or relabel the proxy watermark.

## Why this should travel better than “fly plays game X”

The viewer simultaneously sees:

1. an intuitive challenge;
2. an invisible/noisy sensory signal;
3. biological neural activity proxies/outputs;
4. a causal topology control;
5. a quantitative result.

The spectacle and the scientific control are the same picture.
