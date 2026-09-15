# E002b goal-channel probe

E002b is the restricted mechanistic propagation assay for the frozen MaleCNS v1.0
odor/wind goal-channel candidate. It deliberately stops at PFL3/PFL2. It does not
include heading comparison, DNa02 turning, or odor-source navigation.

## Sealed real-data inputs

The first real E002b input set is bound by
`authority/malecns-v1.0-e002b-input-seal.json`:

- integration route audit SHA-256
  `393044451445af3c1ba1dd720adcb633e134b407e37a3dfcd42d17ad7d515a13`;
- integration sign-v2 SHA-256
  `99f26acce9c00706bc00b7a47408157f1119f71aff49549e539db41f39925a1f`;
- integration topography SHA-256
  `6b37488bac2adf37165d280064600fb16ea39975d535ff4af6b0f3570d54482c`;
- raw MaleCNS weight table SHA-256
  `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1`.

The input seal was created after the audit artifacts existed but does not modify the
preregistered E002b criteria, route, threshold list, or relay selection.

## Sign readiness

`integration-sign-audit-v2` reports both:

- `ready_for_modeled_sign_probe = true`;
- `ready_with_sensitivity_requirements = true`.

All presynaptic types in the primary route have an explicit modeled `+1` convention
from the pinned type-level transmitter authority. This remains a modeling convention,
not receptor-level physiology.

PFNp_b is the lowest-confidence type-level prediction at 53.2%. Its 136 exact body IDs
are therefore carried into the mandatory sensitivity condition where every PFNp_b
presynaptic edge is assigned modeled sign zero.

The frozen hDeltaM relay comparator has its own authority file rather than modifying
the already-sealed integration transmitter authority. The same pinned Male CNS Cell
Type Explorer release predicts hDeltaM as ACh at 95.9% confidence.

## Frozen runtime

`configs/e002b_probe_runtime_v1.json` freezes the engineering perturbation before the
first official real-data E002b run:

- seed: `22002`;
- modeled steps: `32`;
- pulse steps: `8`;
- abstract drive amplitude: `1.0`;
- structural thresholds: `1, 3, 5, 10`;
- repository default rate-model tau, gain, and dt.

These are not physiological estimates and are not eligible for tuning against E002b
response magnitude or downstream navigation performance.

## Primary semantics

The primary model keeps the literature-constrained odor-context abstraction separate
from the additive transmitter sensitivity model.

```text
PFNa / PFNm / PFNp -> hDeltaC -> hDeltaG -> PFL3 / PFL2
                          ^
                          |
                  FB5AB odor context
```

For the preregistered gate model, odor-context OFF removes PFN-to-hDeltaC transmission.
The separate hDeltaC lesion removes hDeltaC-to-hDeltaG. The separate hDeltaG lesion
removes hDeltaG-to-PFL3/PFL2. Thus the three negative controls test distinct layers.

The additive `FB5AB_ACh` condition directly drives FB5AB together with each PFN family
and is reported only as a mechanistic sensitivity comparator. It cannot replace the
primary gating abstraction because it yields a larger response.

## Real structural support carried into the probe

At structural weight >=5, the sealed route artifact contains complete hDeltaG-mediated
three-edge paths into PFL3 for all preregistered input families:

| input family | complete paths | source coverage | hDeltaC | hDeltaG | PFL3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FB5AB | 90 | 2/2 | 11/20 | 7/8 | 23/24 |
| PFNa | 271 | 44/58 | 11/20 | 7/8 | 23/24 |
| PFNm | 69 | 10/47 | 8/20 | 5/8 | 17/24 |
| PFNp | 95 | 18/291 | 11/20 | 7/8 | 23/24 |

At weight >=10, PFNp has no complete PFL3 or PFL2 route. That remains a negative
high-threshold result and is not repaired by changing threshold or selector.

## Topography policy

The topography audit remains `directional_role_status = unresolved`.

- PFNa: 56/58 cells have parseable anatomical columns;
- PFNm: 47/47;
- PFNp: 0/291, so PFNp remains pooled;
- hDeltaC: 20/20;
- hDeltaG: 8/8;
- PFL3: 24/24;
- PFL2: 12/12.

E002b reports PFNa/PFNm single-column impulses where metadata exists, but column labels
are never converted to a physical wind angle or behavioral direction.

## Independent pre-run falsification check

Before the official local runner was executed, the sealed route/sign/topography JSONs
were independently replayed with the repository rate-model equations. This was a
sanity check, not the authoritative E002b artifact.

Across thresholds 1/3/5/10:

- deterministic replay error was exactly `0`;
- odor-gate OFF produced exactly zero PFL response to PFN probes;
- hDeltaC-output cut produced exactly zero PFL response;
- hDeltaG-output cut produced exactly zero PFL response;
- intact PFNa/PFNm/PFNp responses followed the structural threshold pattern;
- PFNp intact response was exactly zero at threshold 10, matching the absence of a
  complete threshold-10 structural route.

At threshold 5, modeled PFL3 peak magnitudes for the abstract uniform PFN probes were
approximately:

- PFNa: `0.42688`;
- PFNm: `0.10734`;
- PFNp: `0.16822`.

The PFNp_b sign-zero sensitivity changed the PFNp-to-PFL3 peak by about -6.9% at
threshold 1, -1.1% at threshold 3, and 0% at threshold 5. Thus the lowest-confidence
PFNp_b transmitter prediction is not carrying the threshold-5 propagation result in
this restricted model.

These values are modeled response amplitudes under the frozen engineering probe. They
are not firing rates, tuning curves, effect sizes, or qualification thresholds.

## Official run

From a checkout containing the sealed local audit JSONs and raw MaleCNS weights:

```bash
bash scripts/run_e002b_goal_channel_probe.sh
```

Expected output:

```text
results/e002/goal-channel-v1.json
```

The official artifact additionally includes the hDeltaM comparator extracted from the
exact raw weight table and must be reviewed before E002b is described as passed.

## Claim boundary

A passing official E002b result can support only:

> Under an explicit literature-constrained gating abstraction and conservative
> type-level modeled-sign conventions, the restricted body-ID-resolved MaleCNS
> candidate goal channel deterministically propagates abstract PFN perturbations
> through hDeltaC and the frozen hDeltaG relay to PFL outputs, with the preregistered
> causal cuts abolishing that modeled propagation.

It still cannot support:

- physical wind-direction tuning;
- odor tuning of the modeled cells;
- physiological effective connectivity;
- measured neural firing;
- PFL3 goal-versus-heading comparison;
- steering direction or DNa02 motor output;
- odor-source navigation;
- intact-versus-rewired behavioral superiority.
