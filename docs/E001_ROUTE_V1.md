# E001 route v1: preregistered structural test

This experiment is the next evidence gate for the public `fly-sniff` claim. It asks a narrower question than "can we draw a path through the connectome?":

> **Does the exact MaleCNS v1.0 release contain a reviewable, thresholded structural corridor linking literature-anchored odor and wind candidates to a bilateral steering readout under one frozen search plan?**

A positive E001 result is structural evidence only. It does not establish neural dynamics, odor tuning, synaptic sign, or successful navigation.

## Frozen primary stages

`configs/staged_route_v1.json` preregisters five required structural questions:

1. odor-value candidates (`MBON12`, `MBON13`, `MBON19`, `AD1b2`) → `FB5AB`;
2. `FB5AB` → `hDeltaC`;
3. wind-direction `PFNa` / `PFNp` / `PFNm` candidates → `hDeltaC`;
4. `hDeltaC` → `PFL3`, allowing explicit intermediate populations rather than assuming a direct edge;
5. `PFL3` → bilateral `DNa02`.

The labels are cross-dataset literature priors, not MaleCNS body-ID assertions. If a label is absent, a bounded corridor is absent, or the persisted corridor fails audit, that stage fails. Do not broaden regexes or increase hop/fanout/weight settings after seeing the failure and then call the rescued route preregistered.

## Exact body-ID handoff gate

Independent stage success is not enough. Five local corridors could use different members of the same named population and never form the biological hypothesis we intended to test. The v1 config therefore preregisters three required handoffs:

1. at least one exact retained `FB5AB` body ID must be shared between `odor_value_to_fb5ab` and `fb5ab_to_hDeltaC`;
2. at least one exact retained `hDeltaC` body ID must be shared by the odor-input stage, the wind-input stage, and the outgoing `hDeltaC_to_pfl3` stage;
3. at least one exact retained `PFL3` body ID must be shared between `hDeltaC_to_pfl3` and `pfl3_to_dna02`.

The second check is deliberately strong: the primary integration hypothesis should not pass merely because odor reaches one hDeltaC neuron, wind reaches another, and a third projects toward steering. Exact body-ID overlap is still only structural evidence, but it makes the staged result a coherent structural hypothesis rather than a collage of unrelated paths.

## Separate memory hypothesis

The same config contains two **nonblocking** exploratory stages for `hDeltaK ↔ PFG` recurrence. They represent the odor-blank persistence hypothesis motivated by 2026 functional work. They are intentionally excluded from the primary steering gate. A memory corridor cannot rescue a failed odor/wind-to-steering route, and a steering corridor cannot establish working memory. Optional hDeltaK and PFG handoff checks report whether the same exact body IDs participate in both traced directions.

## Run on the exact local MaleCNS files

```bash
fly-sniff-staged-trace \
  data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather \
  data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  --config configs/staged_route_v1.json \
  --output data/cache/staged-route-v1 \
  --strict
```

`--strict` succeeds only when every **primary required** stage produces a candidate corridor, every candidate passes its structural audit, and every required exact-body-ID handoff passes. Optional memory stages and handoffs are still executed and reported but cannot make the primary gate pass or fail.

The same command is encoded in `.github/workflows/e001-realdata.yml`. That workflow downloads the pinned public MaleCNS v1.0 flat-connectome files, records their SHA-256 digests, runs the frozen v1 plan, and uploads the full evidence bundle before enforcing the scientific pass/fail result. A failed biological hypothesis therefore remains inspectable instead of disappearing behind a red CI badge.

## Required artifacts

The top-level `staged_trace_report.json` records SHA-256 digests and sizes for the annotation table, weight table, and config supplied to the CLI. `handoff_audit.json` records the exact shared body IDs across stage boundaries. Each stage writes:

- `source_seeds.csv`: every exact body ID matched by the source regexes plus available labels/side;
- `target_seeds.csv`: every exact body ID matched by the target regexes plus available labels/side;
- `nodes.parquet`: retained corridor nodes;
- `edges.parquet`: retained thresholded edges;
- `path_provenance.csv`: forward/reverse search depths and retained seed flags;
- `trace_report.json`: stage settings, counts, hypothesis class, and evidence scope;
- `structural_audit.json`: closure, threshold, bounded-path, geometry, and report-consistency checks for candidate corridors.

These artifacts are the review surface. A regex count or a type-level path is not enough to qualify a population.

## Review before E002

For every primary stage, inspect the exact seed tables, retained intermediate types, and required handoff body IDs. Reject obviously over-broad matches, cross-side mistakes, anonymous/uninterpretable route composition, or dataset-schema surprises before building a GraphBundle. Preserve a failing E001 run as a negative result rather than tuning the final benchmark around it.

Only after the route is reviewed should we construct the signed candidate GraphBundle and run E002 mirrored perturbation, laterality, lesion, and deterministic replay checks. The final navigation-control design is already versioned in code, but its manifest must not be sealed until it contains the digest of that reviewed and E002-qualified GraphBundle.

`fly-sniff-final-v2` uses the same held-out/OOD plume seeds for intact, lesion, classical, and eight independently seeded directed degree-preserving rewires. The original designated rewire remains available for backwards-comparable reporting, while the wiring claim additionally requires intact SPL superiority over the per-environment mean of the sealed rewire ensemble.

## Claim ladder

- **E001 passes:** "The MaleCNS release contains this audited, body-ID-continuous structural hypothesis under the preregistered search plan."
- **E002 passes:** "Under our explicit rate-model assumptions, the reviewed corridor propagates mirrored inputs to the expected bilateral steering sign and survives deterministic replay checks."
- **Frozen navigation comparison passes:** report the measured intact-vs-control and intact-vs-null-ensemble effects with uncertainty.
- **Only then:** a social headline may describe what the connectome-derived model did in the odor-navigation task. It must still identify the activity as modeled rather than recorded physiology.

See `authority/olfactory-navigation-literature-v1.json` for the literature priors and explicit forbidden inferences.
