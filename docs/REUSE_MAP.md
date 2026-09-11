# Reuse map: copy plumbing, not claims

The priority is to ship a falsifiable MaleCNS result quickly without rebuilding solved simulation infrastructure or importing incompatible licensing/assumptions.

## Use now / safe reference

### PomPy — `InsectRobotics/pompy`

- Purpose: NumPy puff-based odor plume simulation.
- License: MIT.
- Useful pieces: puff advection/diffusion mathematics, concentration-field processors, Matplotlib animation patterns.
- Decision: **safe to adapt with attribution**, but do not replace the current vectorized `TurbulentPlume` merely for provenance theatre. Compare equations/behavior and borrow only pieces that materially improve fidelity or visualization.

### I2Bot — `XuelongSun/I2Bot`

- Purpose: embodied multimodal insect navigation in Webots, including odor-plume tracking.
- License: MIT.
- Useful pieces: odor-puff world visualization, robot/plume integration patterns, future sensorimotor embodiment.
- Decision: **future embodiment reference**, not required for the first social result.

### InsectNavigationToolkitModelling — `XuelongSun/InsectNavigationToolkitModelling`

- Purpose: central-complex / mushroom-body inspired navigation toolkit.
- License: MIT.
- Useful pieces: navigation model organization, path-integration/reference controllers.
- Decision: eligible for attribution-preserving reuse where it supplies a baseline or utility. It is not MaleCNS evidence and must never be blended into the connectome controller without explicit labeling.

## Use later

### FlyGym 2.x — `NeLy-EPFL/flygym`

- Purpose: NeuroMechFly v2 embodied fly with vision, olfaction, walking, interactive scenes.
- License: Apache-2.0.
- Current integration cost: FlyGym 2.x uses a rewritten 2026 API and currently targets Python 3.12+, while `fly-sniff` targets Python 3.11.
- Decision: **do not block social v0 on FlyGym**. Integrate after the topology experiment works, when a legged embodied demonstration adds scientific value.

## Scientific reference only unless licensing changes

### Kathman et al. — `nagellab/Kathmanetal2025`

- Purpose: turbulent-plume Drosophila navigation with baseline / goal-directed / search states and odor-memory dynamics.
- License: GNU.
- Decision: use as a behavioral/scientific reference. **Do not copy GPL/GNU implementation into this MIT repository.** Reimplement ideas independently from the paper if needed.

### insectNavigationCX — `XuelongSun/insectNavigationCX`

- Purpose: copy-and-shift, ring-attractor and steering building blocks for multimodal insect navigation.
- Repository search does not expose a clear license in the project root.
- Decision: use the publication/model as conceptual prior; **do not copy source until the license is verified**.

### `NathanMULLER/flygym-game`

- Purpose: a game-style wrapper around FlyGym/NeuroMechFly.
- Decision: useful UX inspiration only. A game engine is not on the v0 critical path, and source should not be copied until its exact license/provenance is verified.

## Fast architecture decision

```text
TODAY
MaleCNS graph + our controller API
        ↓
FlySniffEnv + our vectorized plume
        ↓
E002A two-choice sniff
        ↓
easy plume
        ↓
Who Farted? Matplotlib MP4

LATER, only if useful
        ↓
saved trace/receipt
        ├── browser replay (Canvas / Phaser / WebGL)
        └── FlyGym / NeuroMechFly embodiment
```

The presentation layer should replay authoritative Python episode state. It must not become a second simulation implementation with subtly different controller logic.
