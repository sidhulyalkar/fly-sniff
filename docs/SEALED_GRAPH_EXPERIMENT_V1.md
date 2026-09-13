# Sealed GraphBundle experiment v1

This protocol exists to make the navigation result scientifically interpretable even if the result is negative. It freezes the candidate graph, sign authority, sensory normalization, optimizer budget, development seeds, and final-test policy before navigation performance is inspected.

The public claim boundary remains narrow: this is a fixed model built on MaleCNS structural connectivity plus explicit literature and transmitter priors. Passing navigation does not establish measured physiology, a complete fly sensory algorithm, or that the biological connectome literally performed the simulation.

## 1. E001 evidence is counted by biological edge

A biological connection is identified by the ordered pair `source_body_id -> target_body_id`.

If the same MaleCNS edge appears in more than one staged-route artifact, it is counted once. The evidence pack preserves every stage in `stages_seen`. If two stage artifacts report different structural weights for the same ordered body-ID pair, evidence generation fails instead of choosing one value.

This prevents staged-route overlap from inflating edge counts or total structural weight.

## 2. Transmitter and sign authority

Every unique presynaptic body ID in the candidate GraphBundle receives an auditable record:

`source_body_id -> type -> transmitter -> confidence -> evidence_level -> authority -> model_sign`

Resolution order is frozen before performance:

1. independently reviewed exact-body authority;
2. exact MaleCNS body annotation from the sealed `nodes.parquet`;
3. type-level transmitter evidence applied by the sealed type annotation;
4. unknown.

The exact `nodes.parquet` SHA-256 is bound into the sign report and candidate manifest. A sign report produced from another nodes file is rejected even if graph topology is unchanged.

Unknown transmitter evidence remains `transmitter=null`, `confidence=null`, and `sign=0`. A nonzero edge sign without matching authority causes the sign audit to fail. Glutamatergic sign remains unresolved under the current v1 sign convention.

A nonzero sign is still a modeling convention derived from predicted presynaptic transmitter. It is not a measurement of receptor-specific postsynaptic effect.

The existing minimum signed-edge fraction is **0.60**. This threshold must not be relaxed because the candidate fails it.

## 3. Candidate graph inclusion rule

The candidate graph is the exact pre-performance GraphBundle supplied to the sealer. The optimizer cannot choose graph membership.

Every sealed node must occur in at least one required E001 primary-stage node artifact. Every sealed edge must occur, with the same structural weight, in the unique union of the required primary E001 stages:

- `odor_value_to_fb5ab`
- `fb5ab_to_hDeltaC`
- `wind_to_hDeltaC`
- `hDeltaC_to_pfl3`
- `pfl3_to_dna02`

The large `hDeltaC_to_pfl3` corridor is therefore handled without a post-hoc type whitelist. Exact intermediate membership is whatever was already present in the supplied pre-performance GraphBundle, subject to the required-stage provenance check.

After sealing, v1 forbids changing node membership, edge membership, structural weights, model-role membership, signs, normalization, or scientific authority. Any such change requires a new protocol version and a new final seed draw.

## 4. PFN and sensory normalization

Task-optimized sensory roles use `role_total_l1` normalization.

For each sensory role, the modeled role-level value is divided equally across the exact sealed body IDs in that role. Therefore the total absolute direct drive assigned to the role equals the absolute role-level value regardless of population size.

This applies to:

- `odor_context_left`
- `odor_context_right`
- `wind_basis_left`
- `wind_basis_right`

The exact current PFN body IDs do not change. This rule prevents a larger left or right PFN population from receiving a larger total external drive merely because it contains more cells.

## 5. Immutable candidate manifest

`fly-sniff-seal-candidate` creates `fly-sniff-sealed-candidate-v1` only after checking:

- passing E001 primary structure;
- passing required body-ID handoffs;
- exact role membership against the E001 role review;
- node and edge provenance against required stages;
- unique source-target graph edges;
- structural-weight agreement;
- source-body sign authority and nodes-file identity;
- frozen sensory normalization;
- the unchanged signed-edge coverage gate.

The manifest binds the full node-ID set, canonical edge identity, role membership, GraphBundle file hashes, sign-authority report, inclusion policy, task config, E001 evidence, and source artifacts.

Real matched optimization is not allowed unless `training_ready=true` and the manifest verifies against the supplied GraphBundle and task config.

## 6. Matched development optimization

The real experiment lane uses the sealed candidate only. It does not expose an exploratory flag.

The matched cohort is:

- intact sealed topology;
- eight independently seeded directed degree-preserving rewires;
- the prespecified bilateral steering-input lesion.

Every topology receives the same frozen train/validation split, trainable-parameter bounds, CEM population, generation count, episodes per candidate, objective, optimizer seed, and total represented episode budget. Each topology is optimized separately. Comparing a trained intact graph against untrained rewires is forbidden.

The red-team lane additionally reconstructs optimizer work from persisted receipts, replays deterministic CEM mechanics, checks rewire completion and degree preservation, and runs diagnostic-only shortcut attacks. These diagnostics cannot change v1 graph membership, parameters, thresholds, or final-test policy.

## 7. Final test is one-way

The trained final manifest is frozen only after:

- the sealed candidate verifies;
- matched intact/rewire/lesion training is complete and hash-bound;
- the represented optimizer execution audit verifies equal budgets;
- intact trained E002 passes under the frozen criteria.

The final command requires `--arm-final` and writes a one-way consumption lock before any held-out or OOD episode is evaluated. Presence of that lock forbids another v1 final run even if execution later fails.

Changing the output directory is not a legal way to rerun the final test.

## 8. Mac execution

Development and final execution are intentionally separated.

Development only:

```bash
BUNDLE=/path/to/sealed-graphbundle \
E001_RUN=/path/to/passing-e001-run \
bash scripts/run_sealed_experiment_mac.sh
```

This produces `READY_FOR_FINAL.md` after sealing, matched optimization, red-team audits, trained E002, and final-manifest freeze. It does not evaluate final held-out or OOD episodes.

After reviewing the frozen development artifacts, consume that exact run once:

```bash
RUN_FINAL=1 \
FINAL_RUN_DIR=results/sealed-experiment/<frozen-run-id> \
BUNDLE=/path/to/sealed-graphbundle \
bash scripts/run_sealed_experiment_mac.sh
```

Final mode does not rebuild, retrain, requalify, or refreeze the candidate. It only verifies and consumes the already frozen artifacts.

## Interpretation rule

Whatever the final test reports is the v1 result. A weak or null intact-vs-rewire effect is not a reason to alter graph inclusion, sign rules, PFN normalization, optimizer budgets, or scientific thresholds. The cinematic must be built around the observed result, with measured anatomy, literature priors, modeled activity, and evaluation outcome visually distinguished.
