# MaleCNS integration evidence v1

This document summarizes the reviewed structural and modeled evidence for the candidate odor/wind-to-steering route. It is intentionally narrower than a behavioral claim.

## Evidence authorities

- `authority/malecns-v1.0-goal-relay-evidence.json`
- `authority/malecns-v1.0-integration-route-evidence.json`
- `authority/malecns-v1.0-persistent-goal-structure-evidence.json`
- `authority/malecns-v1.0-steering-probe-evidence-v2.json`
- `authority/malecns-v1.0-integration-transmitter-evidence.json`

Each authority records the source artifact hash and the claim boundary appropriate to that evidence class.

## Candidate structural route

The current discovery-stage goal route is:

```text
FB5AB  -- odor-context prior --\
PFNa   -- wind prior ---------+--> hDeltaC --> hDeltaG --> PFL3 / PFL2
PFNm   -- wind prior ---------+
PFNp   -- wind prior ---------/
```

`hDeltaG` was selected from the preregistered hDeltaA/G/H/M structural comparison before downstream behavioral optimization. `hDeltaM` remains the frozen comparator. Relay identity must not be changed because a navigation, lesion, training, or rewire experiment performs poorly.

This route is only the goal-channel side of the eventual steering computation. PFL3 also receives a heading channel, including EPG/Delta7 input in the published steering architecture. A hDeltaG-to-PFL3 route alone must not be promoted to a complete steering controller.

## Integration-route audit result

The reviewed `malecns-integration-route-audit-v1` artifact was produced from clean git head `913b82de0243761bfb65bde1aa6e94959525c15b`. Its SHA-256 is:

```text
393044451445af3c1ba1dd720adcb633e134b407e37a3dfcd42d17ad7d515a13
```

At the fixed weight >= 5 threshold, complete three-edge routes into PFL3 were observed for every preregistered input family:

| Route | paths | source coverage | hDeltaC | hDeltaG | PFL3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FB5AB -> hDeltaC -> hDeltaG -> PFL3 | 90 | 2/2 | 11/20 | 7/8 | 23/24 |
| PFNa -> hDeltaC -> hDeltaG -> PFL3 | 271 | 44/58 | 11/20 | 7/8 | 23/24 |
| PFNm -> hDeltaC -> hDeltaG -> PFL3 | 69 | 10/47 | 8/20 | 5/8 | 17/24 |
| PFNp -> hDeltaC -> hDeltaG -> PFL3 | 95 | 18/291 | 11/20 | 7/8 | 23/24 |

The weight >= 5 support is distributed. For PFNa, no single hDeltaG body ID accounts for more than about 26% of retained three-edge paths. At weight >= 10, however, the surviving FB5AB/PFNa/PFNm paths collapse onto the same `hDeltaC 14975 -> hDeltaG 48642` bridge. Threshold-10 survival is therefore robustness evidence, not population-wide coverage.

PFNp has no complete PFL3 or PFL2 route at weight >= 10. This negative high-threshold result remains part of the record rather than being repaired by changing selectors or thresholds.

## Persistent-goal structural lane

The original exact `PFG` selector returned zero rows and remains part of audit history. MaleCNS uses `PFGs` for the 18-neuron population, so a separate alias-resolution audit tested `hDeltaK <-> PFGs`.

The result is strongly recurrent structurally:

- hDeltaK -> PFGs: 84 edges survive weight >= 10;
- PFGs -> hDeltaK: 96 edges survive weight >= 10;
- 81 reciprocal neuron pairs survive weight >= 10;
- those reciprocal pairs cover all 31 hDeltaK and all 18 PFGs neurons.

By contrast, direct hDeltaK -> PFL3 and PFGs -> PFL3 support disappears at weight >= 3, as do the tested two-hop routes through the recurrent pair.

Dense recurrence therefore does not imply direct steering access. Structure alone does not establish persistent activity, working memory, attractor dynamics, or behavioral function.

## Steering probe v2

The provenance-hardened `E002a-steering-scaffold-propagation-v2` report passed all seven fixed gates. It records hashes for `nodes.parquet`, `edges.parquet`, `roles.json`, and `manifest.json` plus seed 13013 and 32 model steps.

Observed modeled turns:

- left PFL3 drive: `+0.9716491195410979`;
- right PFL3 drive: `-0.9716533526536367`;
- deterministic replay error: `0.0`;
- complete PFL3-output cut peak turn: `0.0`.

Partial relay-lesion amplitudes remain descriptive because the current rate model normalizes postsynaptic incoming weights.

## Sign audit v1: preserved blocked result

`malecns-integration-sign-audit-v1` correctly blocked on the local public annotation feather because that artifact has no recognized neurotransmitter column. The result is preserved as a provenance finding rather than overwritten.

The local annotation file remains authoritative for the exact body IDs/types used in the structural route, but it is not an adequate source for transmitter identity.

## Sign audit v2: external transmitter-evidence recovery

The recovery lane is additive and independently provenance-sealed.

`authority/malecns-v1.0-integration-transmitter-evidence.json` pins the public Male CNS Cell Type Explorer to commit:

```text
789cc6c105798ce2fd70ba85dab394f90899616b
```

for the same `male-cns:v1.0` catalog. It stores type-level predicted transmitter identity and confidence for every exact subtype represented by the frozen route.

All frozen integration presynaptic types are catalog-predicted ACh, but prediction confidence varies. The lowest-confidence case is `PFNp_b` at 53.2%. It is therefore a mandatory sensitivity case: downstream mechanistic probes must be repeated with `PFNp_b` modeled sign set to zero. This confidence value is not converted into a post-hoc qualification threshold.

`integration_sign_audit_v2.py`:

1. verifies the exact integration-audit SHA-256;
2. verifies the pinned transmitter-authority SHA-256;
3. verifies the raw weight-table SHA-256;
4. reuses exact body-ID/type membership from the already-sealed integration audit;
5. maps exact subtype names through the pinned transmitter authority;
6. preserves transmitter prediction confidence;
7. blocks on a missing subtype or zero modeled sign;
8. records mandatory sensitivity cases and affected body IDs.

Type-level ACh -> modeled `+1` remains a modeling convention, not measured receptor-level physiology.

Run:

```bash
bash scripts/run_integration_sign_audit_v2.sh
```

Expected output:

```text
results/route/integration-sign-audit-v2.json
```

## Descriptive topography audit

The next route constraint is anatomical topology, not task performance.

`integration_topography_audit.py` parses column metadata only where the frozen MaleCNS instance strings actually provide it and reports thresholded source-column -> target-column structural matrices.

It deliberately returns:

```text
directional_role_status: unresolved
```

because anatomical column numbers are not assigned wind/goal angles by this audit. PFL3 instance L/R tokens are reported only as instance metadata and are not promoted to motor side. Candidate steering side remains grounded in exact PFL3 -> DNa02 connectivity from E002a.

Release metadata currently supports column parsing for nearly all PFNa/PFNm cells and all hDeltaC, hDeltaG, PFL3, and PFL2 cells. PFNp lacks usable `_C#` instance labels in this artifact and therefore remains pooled rather than receiving invented column positions.

Run:

```bash
bash scripts/run_integration_topography_audit.sh
```

Expected output:

```text
results/route/integration-topography-audit-v1.json
```

## Heading channel is a separate required branch

PFL3 is a goal/heading comparator, not merely an output of the odor/wind goal route. The heading side must therefore be audited independently before the project claims biologically faithful steering.

`configs/heading_route_audit_v1.json` preregisters fixed 1/3/5/10 structural tests for:

```text
EPG -> PFL3
EPG -> Delta7
Delta7 -> PFL3
EPG -> Delta7 -> PFL3
```

Delta7 is predicted glutamatergic in the current catalog and remains unresolved under the conservative modeled-sign policy until separate receptor/physiology evidence justifies an effective sign.

Run:

```bash
bash scripts/run_heading_route_audit.sh
```

Expected output:

```text
results/route/heading-route-audit-v1.json
```

## E002b and E002c are intentionally separate

`configs/e002b_goal_channel_protocol_v1.json` is preregistered before running the signed goal-channel model.

E002b stops at PFL outputs. It asks whether the frozen signed candidate route propagates controlled upstream perturbations through hDeltaC and the frozen hDeltaG relay reproducibly. It does not claim heading comparison, DNa02 turning, or odor-source navigation.

The primary odor-context abstraction is a gate on PFN-derived hDeltaC wind representation, following the published Matheson model. Treating the predicted cholinergic FB5AB structural input as additive drive is retained only as a descriptive sensitivity comparator. No behavioral score may choose between those semantics.

E002c will be designed only after the heading-route evidence is reviewed. It will add the independently constrained heading channel and then test PFL3 goal/heading comparison before connecting to the already-audited downstream steering scaffold.

## Current claim boundary

Defensible:

> MaleCNS v1.0 contains a body-ID-resolved structural bridge from literature-motivated FB5AB/PFN populations through hDeltaC and a frozen hDeltaG relay into PFL3/PFL2, plus a separately qualified modeled PFL3-to-DNa02 steering-propagation scaffold.

Not yet defensible:

- the complete fly smell-navigation circuit;
- physiological signal transmission through hDeltaC -> hDeltaG -> PFL;
- PFN/FB5AB tuning in this individual MaleCNS connectome;
- PFL3 heading/goal comparison in the current model;
- the complete inhibitory steering see-saw circuit;
- persistent dynamics from hDeltaK/PFGs structure alone;
- modeled rate state as measured firing;
- intact MaleCNS wiring outperforming matched rewiring in odor navigation.

## Current local science handoff

```bash
git pull --ff-only
.venv/bin/ruff check src tests
.venv/bin/python -m pytest -q
bash scripts/run_integration_sign_audit_v2.sh
bash scripts/run_integration_topography_audit.sh
bash scripts/run_heading_route_audit.sh
```

Review all three outputs before implementing/running E002b. Do not select a relay, structural threshold, sign interpretation, or anatomical direction from downstream task performance.
