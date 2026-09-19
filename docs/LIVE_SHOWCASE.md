# Live showcase: experiment theater

The live webpage is a presentation layer over frozen scientific outputs. It is not a second simulator and it must not silently turn development evidence into a biological claim.

## Public question

> How does intermittent odor evidence become a steering decision, and what changes under controlled interventions?

The page answers that question through three deliberately separate modes:

- **WATCH** — one selected controller followed through plume → sensory evidence → controller/circuit state → steering;
- **COMPARE** — paired counterfactual conditions in the same plume, start state, clock, and paired RNG;
- **PROVE** — measured O002 data, provenance, claim boundaries, anatomy/source receipts, and the evidence ladder.

The visual hierarchy is intentional: cinematic presentation may make an experiment legible, but it may never upgrade the evidence class of the underlying artifact.

## End-state experience

A visitor should be able to:

1. follow one selected controller through the **same frozen plume** with ghost counterfactual trajectories in WATCH mode;
2. jump to automatically derived replay events such as first plume contact, maximum bilateral asymmetry, peak turn, and closest source approach;
3. toggle plume and hidden source as viewer-only overlays;
4. inspect left/right sensory inputs, bilateral difference, steering command, and controller/circuit readouts at the exact same replay frame;
5. compare all paired interventions at the same replay time in COMPARE mode;
6. switch between development-controller state, topology, and independently sourced anatomy without conflating them;
7. inspect measured O002 named-odor fingerprints natively in the browser rather than depending on a dense static figure;
8. inspect provenance and claim boundaries in PROVE mode without opening developer tools.

## Presentation modes

### WATCH

WATCH is the default public entry point. Only the selected controller is drawn as the primary fly. Other conditions remain visible as low-emphasis trajectory counterfactuals. The plume renderer may use short histories of already exported frames to communicate intermittent filament structure, but it must never synthesize new scientific state.

The event strip is derived exclusively from frozen replay fields. Event labels are descriptive replay landmarks, not inferred biological events.

### COMPARE

COMPARE renders all conditions with equal visual status and adds same-time metrics such as distance to source, maximum current odor input, and absolute turn command. Comparison cards are presentation summaries of the frozen replay and must not be promoted to statistical claims.

### PROVE

PROVE contains the material that justifies or limits the public story:

- measured O002 response vectors and source receipt;
- current circuit qualification status;
- morphology/topology provenance;
- paired-control contract;
- claim boundaries and the unresolved evidence ladder.

The archived O002 figure may remain available as a secondary artifact, but the primary page should use native responsive views of the frozen data.

## Evidence layers

The live site deliberately separates three evidence classes.

### A. Measured sensory evidence

O002 is the current truth card. It is based on frozen measured DoOR physiology and may describe within-study chemical-class geometry, coding decompositions, resampling stability, and low-dimensional retention.

It does **not** establish that a specific MaleCNS circuit uses that structure for navigation.

### B. Modeled circuit computation

A graph-backed replay may show modeled dynamics over a MaleCNS GraphBundle. Those values are simulated rate-model states over structural connectivity, not neural recordings.

Claim-bearing graph mode must refuse an unqualified bundle by default. Generic graph qualification is not enough: the manifest must explicitly list `odor-plume` in `qualified_experiments`, so qualification earned by another experiment such as looming cannot leak into navigation claims.

### C. Behavioral presentation

The browser replays precomputed simulator frames. It never reimplements the controller or plume in JavaScript.

That keeps one scientific implementation authoritative:

```text
Python simulation / frozen receipts
            ↓
    replay JSON export
            ↓
browser presentation only
```

## Named odors without overclaiming

The site includes a measured named-odor explorer sourced directly from the frozen O002 v1 matrix. A visitor can select a source odor and inspect its measured 24-channel response fingerprint.

That selector is intentionally **not wired into the movement replay yet**. The current navigation plume carries generic scalar concentration. A named odor may drive movement only after a frozen odor-to-sensory mapping has been independently qualified. This prevents the visual interface from implying a geosmin-, ester-, or acid-specific behavioral result before the experiment actually supports one.

The intended progression is:

```text
named O002 odor
   ↓ measured response fingerprint (available now)
qualified receptor/sensory mapping
   ↓
MaleCNS odor-navigation circuit
   ↓
movement under the same paired-control protocol
```

## Multi-condition arena

The development page currently supports:

- **Bilateral proxy**: biology-inspired development controller;
- **Left antenna off**: explicit one-channel sensory ablation;
- **Odor blind**: both odor channels zeroed while wind remains available.

These are development controls only. They are not described as impaired biological flies.

When a qualified odor-navigation GraphBundle is supplied, the exporter instead produces:

- **MaleCNS topology**;
- **degree-preserving rewire**;
- **MaleCNS odor blind**.

Every condition receives the same plume seed, initial arena state, simulation clock, and paired controller RNG seed. The declared intervention is therefore the intended difference.

## Connectome visualization

There are two very different things we may visualize.

### Topology view

The graph-backed frontend can display nodes, edges, role labels, and activity-linked highlighting from a GraphBundle. When no GraphBundle is present, the right-hand panel must render an explicitly labelled **development controller state** diagram from replayed controller inputs/outputs instead of decorative neurons or an empty faux-anatomy view.

This view is labelled:

> TOPOLOGY • NOT MORPHOLOGY

A deterministic browser layout is only a visualization of graph relationships.

### Anatomical view

The live page now supports real MaleCNS spatial context assets built downstream of the official v1.0 data:

- a low-opacity soma-location point cloud for broad CNS context;
- exact public SWC centerline skeletons for the experimentally relevant circuit;
- active circuit neurons highlighted over the dim whole-brain context;
- neutral geometric camera presets (front / oblique / side);
- a strict source receipt for every XYZ/skeleton asset.

The page must not generate decorative neuron shapes and call them anatomy.

For performance, the whole-connectome context is deterministically bounded by `bodyId` for the current Canvas implementation, while the selected circuit uses higher-resolution skeletons. The next renderer upgrade should move the soma cloud to WebGL so all available measured soma positions can be shown on desktop while mobile uses a lower level of detail. The scientific graph and activity traces remain unchanged.

Build the measured context layer from the local official annotation feather:

```bash
fly-sniff-morphology-context soma \
  data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather \
  --output artifacts/connectome-soma.json
```

With a bounded GraphBundle, fetch exact public SWC skeletons for those body IDs:

```bash
fly-sniff-morphology-context skeletons path/to/graph \
  --output artifacts/selected-skeletons.json \
  --cache-dir ~/fly-sniff-data/cache/malecns-v1.0-swc
```

The live builder can include both layers automatically through `FLY_SNIFF_MALECNS_ANNOTATIONS` and `FLY_SNIFF_FETCH_SKELETONS=1`.

## Critique-resistant design rules

The live page should survive the following questions visibly, without relying on a caption hidden elsewhere:

### “Are the conditions seeing the same odor?”
Yes. The replay JSON contains one shared plume snapshot per time point and paired condition states.

### “Can the controller see the source?”
No. Source coordinates are viewer-only. The UI labels the overlay as such.

### “Did you change more than the wiring?”
The page exposes the intervention for each condition. The degree-preserving rewire uses the same nodes and directed degree structure with topology disrupted.

### “Is this neural activity recorded?”
No. Modeled readouts are labelled modeled dynamics.

### “Is that the actual anatomy?”
Only when an independently sourced XYZ/skeleton asset is loaded. Otherwise the panel says topology, not morphology.

### “Is the development proxy a MaleCNS result?”
No. Development mode carries a permanent DEVELOPMENT PROXY badge and claim_allowed=false.

### “Are repeated browser runs changing the science?”
No. JavaScript only replays frozen JSON.

## Frontend structure

The first integrated page lives in:

```text
web/showcase/
  index.html
  styles.css
  app.js
```

The replay exporter is:

```text
fly-sniff-live-showcase
```

The one-command local site builder is:

```bash
./scripts/build_live_showcase_mac.sh
```

It writes a self-contained static site under:

```text
~/fly-sniff-data/artifacts/live-showcase-<renderer-ref>/site/
```

Preview it locally with:

```bash
cd ~/fly-sniff-data/artifacts/live-showcase-<renderer-ref>/site
python3 -m http.server 8080
open http://localhost:8080
```

To build from a qualified graph:

```bash
FLY_SNIFF_SHOWCASE_GRAPH=/path/to/qualified/graph \
  ./scripts/build_live_showcase_mac.sh
```

Candidate graphs are rejected unless the operator explicitly opts into visibly labelled development output:

```bash
FLY_SNIFF_SHOWCASE_GRAPH=/path/to/candidate/graph \
FLY_SNIFF_SHOWCASE_ALLOW_CANDIDATE=1 \
  ./scripts/build_live_showcase_mac.sh
```

## Roadmap

### Stage 1 — integrated development theater

- cinematic WATCH mode with one selected fly and ghost counterfactuals;
- event-indexed frozen replay;
- COMPARE mode with same-time paired condition summaries;
- explicit development-controller state when no graph is loaded;
- sensory telemetry and explicit interventions;
- topology/anatomy panel with fail-honest labels;
- native O002 measured-data explorer;
- PROVE mode with evidence rail and claim boundary.

### Stage 2 — qualified odor-navigation circuit

- replace development controllers with a qualified graph-backed intact/rewire comparison;
- add frozen behavioral cohort statistics;
- export modeled node-level activity for the selected circuit;
- retain sensory-ablation controls.

### Stage 3 — anatomical MaleCNS context

- ingest independently sourced whole-connectome coordinates/skeletons;
- render a dim whole-brain context plus high-resolution selected circuit;
- synchronize active circuit highlighting to the replay clock;
- retain a topology-only fallback.

### Stage 4 — public live study page

The final public page should let a visitor move between:

1. **Measured sensory structure** — O002;
2. **Circuit mechanism** — qualified intact/rewire/lesion comparisons;
3. **Behavior** — paired source-finding trajectories;
4. **Anatomy** — real whole-connectome context and highlighted pathway;
5. **Evidence** — receipts, hashes, source identities, claim boundaries.

The visual goal is not simply “a fly moving with a glowing brain.” It is a compact causal argument where every visual transition corresponds to a documented experimental relationship.
