# MaleCNS steering scaffold v1

This document defines the narrow scientific claim supported by the first body-ID-resolved steering audit. It deliberately separates a well-supported **steering-side structural scaffold** from the still-unresolved **odor-to-goal pathway**.

## Evidence artifact

The authoritative local audit is `results/route/literature-route-audit-v1.json`, protocol `malecns-literature-route-audit-v1`.

Sealed identity:

- audit SHA-256: `250459ad4128ec37b2ecfee45b77fcf9ca4c784f24a949c9744a95206c5323b7`
- generating git SHA: `329f9d016f9dbb5c34876f3410c8e4f0dd0b3fce`
- annotations SHA-256: `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2`
- weights SHA-256: `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1`
- route-audit config SHA-256: `90c4cc59bb702485d3f0eff6b859194bc51e34ce581f7d8b233ec8e2f63ea05b`

The audit was generated from a clean `feat/science-rigor-v1` checkout. Its structural claims are descriptive. They do not establish physiological influence.

## What reproduced strongly in MaleCNS v1.0

The preregistered 1/3/5/10-synapse sweep showed robust steering-side structure:

| Structural prediction | Edge/path support | Weight sum | Support at threshold 10 |
| --- | ---: | ---: | ---: |
| `FB5AB -> hDeltaC` | 40 edges | 1751 | 40 edges |
| `PFL3 -> DNa02` | 24 edges | 736 | 24 edges |
| `PFL3 -> DNa03` | 24 edges | 468 | 22 edges |
| `PFL3 -> LAL010` | 24 edges | 319 | 18 edges |
| `DNa03 -> DNa02` | 2 edges | 552 | 2 edges |
| `LAL010 -> DNa02` | 2 edges | 357 | 2 edges |
| `PFL2 -> DNa03` | 24 edges | 1571 | 24 edges |
| `PFL2 -> LAL010` | 24 edges | 240 | 11 edges |
| `MBON32 -> DNa02` | 2 edges | 104 | 2 edges |
| `PFL3 -> DNa03 -> DNa02` | 24 two-hop paths | n/a | 22 paths |
| `PFL3 -> LAL010 -> DNa02` | 24 two-hop paths | n/a | 18 paths |

All 24 PFL3 neurons have a direct structural edge to one of the two DNa02 neurons in the audited table. The two DNa03 neurons and two LAL010 neurons provide strong additional relay routes to ipsilateral DNa02 targets.

This is sufficient to define a **candidate steering scaffold** for mechanistic model probes. It is not sufficient to call the scaffold a functional neural circuit.

## Direct hDeltaC-to-PFL3 result and relay interpretation

The direct `hDeltaC -> PFL3` population pair is weak in this MaleCNS release:

- 20 observed edge pairs;
- total structural weight 23;
- every observed edge has weight 1 or 2;
- zero edges survive a minimum-weight threshold of 3, 5, or 10.

This result must not be hidden by lowering a threshold after inspection. However, it also must not be misinterpreted as a failed reproduction of the 2022 wind-guided navigation mechanism.

Matheson et al. explicitly state in their model Methods that the biological pathways connecting hDeltaC to the mutual-inhibition circuit, PFL3, and PFL2 **all involve at least one additional cell type**. Their model collapses these pathways into a simplified hDeltaC output. The paper specifically points to downstream local FB types including hDeltaA, hDeltaG, hDeltaH, and hDeltaM as candidate motifs with strong projections toward PFL3.

Therefore the current conclusion is:

- strong `FB5AB -> hDeltaC` structure is reproduced;
- a strong **direct** `hDeltaC -> PFL3` edge family is not reproduced;
- the literature does not require that direct edge family to be strong;
- the correct next experiment is a preregistered multi-cell relay audit.

Do not lower the direct-edge threshold and do not declare the multi-cell mechanism absent from this result alone.

## Next preregistered goal-relay audit

`configs/goal_relay_audit_v1.json` freezes three hypothesis families before inspecting their MaleCNS edge results:

1. **2022 local-relay candidates**: `hDeltaC -> {hDeltaA,hDeltaG,hDeltaH,hDeltaM} -> {PFL3,PFL2}`;
2. **independent goal-interface positive control**: `FC2A/FC2B/FC2C -> PFL3`, motivated by goal-steering work showing strong FC2 input to PFL3;
3. **newer persistent-goal candidates**: `hDeltaK <-> PFG`, with direct access to PFL3 tested as exploratory rather than assumed.

These families must be reported separately. A strong FC2 result cannot be used to rescue a failed hDelta relay, and a strong hDeltaK/PFG result cannot be retroactively called evidence for the 2022 hDeltaC model.

Run:

```bash
bash scripts/run_goal_relay_audit.sh
```

Expected output:

```text
results/route/goal-relay-audit-v1.json
```

## Restricted candidate graph

`configs/steering_scaffold_candidate_v1.json` includes only:

- all 24 PFL3 neurons;
- both DNa03 neurons;
- both LAL010 neurons;
- both DNa02 neurons;
- structural edge families `PFL3->DNa02`, `PFL3->DNa03`, `PFL3->LAL010`, `DNa03->DNa02`, and `LAL010->DNa02`.

It excludes:

- odor sensory roles;
- FB5AB/hDeltaC from the steering model;
- the contralateral inhibitory see-saw arm;
- the PFL2 gain layer;
- any behavioral success claim.

The exact DNa02 output-role candidates are:

- `steer_left`: DNa02_L, body ID `523769`;
- `steer_right`: DNa02_R, body ID `10360`.

The PFL3 cells are grouped for the isolated model probe according to which DNa02 side they directly target. These groups are called `turn_drive_left` and `turn_drive_right`, not odor roles.

## Sign provenance

`authority/malecns-v1.0-steering-sign-evidence.json` records the sign assumptions separately from structural connectivity.

For the restricted excitatory scaffold, the presynaptic types PFL3, DNa03, and LAL010 are treated as `+1` under a type-level predicted-acetylcholine modeling rule. This is an explicit engineering assumption, not measured synaptic physiology.

The builder refuses an unresolved sign rather than silently converting structural synapse counts into excitation. The scaffold config is bound to the reviewed sign-authority path and SHA-256; generated manifests also record the exact authority used.

The contralateral inhibitory arm remains outside this scaffold until its exact MaleCNS body IDs and transmitter/sign provenance are sealed.

## E002a: isolated steering propagation

`fly-sniff-probe-steering` implements `E002a-steering-scaffold-propagation-v1`.

It drives the audited PFL3 role groups directly and asks only whether the explicit rate model propagates the expected left/right steering semantics through the exact MaleCNS steering scaffold.

Core checks are:

1. the graph remains `qualification_status='candidate'`;
2. no PFL3 drive role is aliased to `odor_left` or `odor_right`;
3. all included edges have explicit non-zero modeled sign;
4. left drive produces positive/left modeled turn and right drive negative/right modeled turn;
5. mirrored drive produces opposite signs;
6. exact-seed replay is deterministic;
7. removing all PFL3 outputs abolishes modeled DNa02 turning.

Additional direct-route and relay lesions are descriptive only.

### Why relay-lesion amplitudes are not qualification thresholds

`MaleCNSRateController` uses postsynaptic absolute-row normalization of log-compressed structural weights. Removing one parallel incoming family causes the remaining incoming weights to be renormalized. Consequently, the magnitude of a modeled turn after a DNa03 or LAL010 lesion is not proportional to that route's biological contribution.

The current model can test topology-dependent propagation, sign/laterality semantics, determinism, and complete disconnection. It must **not** be used to infer physiological effect size from partial-lesion amplitude.

No threshold should be invented after seeing these lesion values.

## Running the isolated probe

After generating `results/route/literature-route-audit-v1.json`:

```bash
bash scripts/run_steering_scaffold_probe.sh
```

This builds:

```text
data/cache/steering-scaffold-v1/
```

and writes:

```text
results/e002/steering-scaffold-v1.json
```

The graph builder is bound to the sealed route-audit SHA-256, reviewed sign-authority SHA-256, and exact body-ID sets. Drift causes the build to fail.

## Claim ladder

After this audit, the strongest defensible statements are:

**Supported:**

> MaleCNS v1.0 contains a body-ID-resolved structural steering scaffold consistent with published PFL3-to-DNa02 architecture, including strong DNa03 and LAL010 relay motifs.

> MaleCNS v1.0 contains strong FB5AB-to-hDeltaC structural connectivity in the audited population pair.

**Not yet supported:**

> We found the complete fly smell-navigation circuit.

> A specific hDeltaC-to-PFL relay is qualified in MaleCNS.

> The full steering see-saw circuit is modeled.

> The modeled state is neural firing.

> Intact MaleCNS wiring improves odor-source navigation over matched rewiring.

The last statement requires the complete role/sign contract, perturbation qualification, degree-preserving rewiring, and frozen held-out behavioral evaluation.
