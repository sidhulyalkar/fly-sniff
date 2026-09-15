# Program A physical sensory contract v1

Program A cannot ask whether biological wiring helps odor navigation if the sensory geometry is an arbitrary simulator convention.

This contract therefore separates two independent questions:

1. **Where is concentration measured in the archived plume, in physical coordinates?**
2. **Where are the fly's left and right olfactory sampling landmarks, in the fly body frame?**

Only after both are independently qualified may they be composed into controller inputs.

## Smooth plume evidence already supported

The Alvarez-Salvado et al. eLife methods for the public PLIF plume report:

- the imaging region extends up to **30 cm downwind** from the odor source;
- the measured field extends up to **8 cm to either side** of the centerline;
- acetone fluorescence was acquired at **15 Hz**;
- the camera was binned to a nominal **0.74 mm/pixel** before plume post-processing.

The public Dryad file contract used elsewhere in fly-sniff is:

- DOI `10.5061/dryad.g27mq71`;
- file `10302017_10cms_bounded_2.h5`;
- dataset `/dataset2`;
- expected shape `(3600, 406, 216)`.

These facts are useful but insufficient for a PASS physical plume artifact. In particular, `(406, 216)` must not be interpreted by shape alone as `(downwind, crosswind)`, and the odor-source origin must not be guessed from an array boundary.

The committed policy therefore remains `BLOCKED_SOURCE_BYTES_UNVERIFIED` until the actual source file is hash-verified, then remains `BLOCKED_ARRAY_ORIENTATION_UNVERIFIED` until exact axes, signs, source origin and archive spatial transform are resolved.

## Bilateral olfactory geometry is a separate evidence problem

The production simulator currently contains an `antenna_separation` value. That value is a simulation parameter, not anatomical evidence, and is explicitly forbidden as Program A geometry authority.

The v1 anatomy contract prespecifies a candidate sampling landmark:

> centroid of the olfactory third antennal segment / funiculus on each side, measured in a scale-calibrated frontal or body-centered projection.

The 2024 wild-type Drosophila SEM atlas (Jurgens, Drechsler & Paululat; DOI `10.1093/genetics/iyae129`) is a candidate image authority because it provides high-resolution external anatomy including frontal head and antenna views. It does **not**, by citation alone, establish the bilateral sampling offsets required by the model.

A real PASS geometry artifact therefore needs a reproducible measurement receipt containing:

- exact image or source-data identity;
- source byte hash when materialized;
- scale authority;
- species/sex/preparation context;
- landmark definition;
- projection convention;
- independent left and right body-frame offsets;
- measurement uncertainty or biological variability.

If those data are not adequate, the correct state is `BLOCKED_REQUIRES_ANATOMY_MEASUREMENT`.

## Why subpixel sampling is mandatory

The plume measurement scale is approximately millimetric per image pixel, while individual fly head and antennal structures are sub-millimetric. Program A therefore cannot assume that left and right antennae correspond to two neighboring integer pixels.

`physical_sensory.py` keeps sensor locations continuous in physical millimeters and freezes **bilinear** spatial interpolation. Rounding or nearest-pixel sampling would create an artificial sensory baseline determined by the image raster rather than anatomy.

## Coordinate conventions

The v1 implementation defines:

- world coordinate 1: `downwind_mm`;
- world coordinate 2: `crosswind_mm`;
- body offset 1: `forward_mm`;
- body offset 2: `lateral_mm`, positive to the fly's left;
- heading: radians from positive downwind, counter-clockwise toward positive crosswind.

The physical plume artifact maps world coordinates to archived array axes only after exact axis roles, signs, millimeters-per-pixel and source-index origin are explicitly frozen.

Left and right anatomy offsets are represented separately. Exact symmetry is not assumed.

## Temporal boundary

For the smooth source, measurements are native 15 Hz observations. Program A v1 freezes `causal_native_frame_hold`:

- a controller may consume the current or most recent measured frame;
- it may not interpolate using a future frame;
- publication upsampling cannot be described as new measurements.

This keeps visual or reproduction interpolation from quietly increasing biological sensory bandwidth.

## Transform assembly

A `PhysicalSensoryTransform` can be constructed only when:

- `PhysicalPlumeCalibration.status == PASS_PHYSICAL_PLUME`; and
- `BilateralSensorGeometry.status == PASS_BILATERAL_SENSOR_GEOMETRY`.

The transform hash binds both exact upstream hashes. Any change to source origin, axis orientation, anatomical offsets, uncertainty, spatial interpolation or temporal policy therefore changes the scientific identity of the sensory transform.

## Current status

As committed in `configs/program_a_physical_sensory_policy_v1.json`:

- plume publication scale evidence: **available**;
- exact plume source SHA-256: **unresolved**;
- archived array orientation/origin: **unresolved**;
- bilateral anatomical sampling offsets: **unresolved**;
- physical sensory transform: **BLOCKED**.

This is progress, not failure. It narrows the remaining work to measurements whose absence would otherwise be easy to hide inside simulator defaults.

No navigation performance is part of this contract.
