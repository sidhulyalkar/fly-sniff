# MaleCNS integration evidence v1

This document summarizes the reviewed structural and modeled evidence for the candidate odor/wind-to-steering route. It is intentionally narrower than a behavioral claim.

## Evidence authorities

- `authority/malecns-v1.0-goal-relay-evidence.json`
- `authority/malecns-v1.0-integration-route-evidence.json`
- `authority/malecns-v1.0-persistent-goal-structure-evidence.json`
- `authority/malecns-v1.0-steering-probe-evidence-v2.json`

Each authority records the source artifact hash and the claim boundary appropriate to that evidence class.

## Candidate structural route

The current discovery-stage route is:

```text
FB5AB  -- odor-context prior --\
PFNa   -- wind prior ---------+--> hDeltaC --> hDeltaG --> PFL3 --> steering scaffold
PFNm   -- wind prior ---------+                    \
PFNp   -- wind prior ---------/                     +--> PFL2
```

`hDeltaG` was selected from the preregistered hDeltaA/G/H/M structural comparison before any downstream behavioral optimization. `hDeltaM` remains the frozen comparator. Relay identity must not be changed because a navigation, lesion, training, or rewire experiment performs poorly.

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

PFNp has no complete PFL3 or PFL2 route at weight >= 10. This is retained as a negative high-threshold result rather than repaired by changing selectors or thresholds.

## Persistent-goal structural lane

The original exact `PFG` selector returned zero rows and remains part of audit history. MaleCNS uses `PFGs` for the 18-neuron population, so a separate alias-resolution audit tested `hDeltaK <-> PFGs`.

The result is strongly recurrent structurally:

- hDeltaK -> PFGs: 84 edges survive weight >= 10;
- PFGs -> hDeltaK: 96 edges survive weight >= 10;
- 81 reciprocal neuron pairs survive weight >= 10;
- those reciprocal pairs still cover all 31 hDeltaK and all 18 PFGs neurons.

By contrast, direct hDeltaK -> PFL3 and PFGs -> PFL3 support disappears at weight >= 3, as do the tested two-hop routes through the recurrent pair.

This structural separation is useful. Dense recurrence does not imply direct steering access. It does not establish persistent neural activity, working memory, attractor dynamics, or behavioral function.

## Steering probe v2

The provenance-hardened `E002a-steering-scaffold-propagation-v2` report passed all 7 fixed gates. It records hashes for `nodes.parquet`, `edges.parquet`, `roles.json`, and `manifest.json` plus seed 13013 and 32 model steps.

Observed modeled turns:

- left PFL3 drive: `+0.9716491195410979`;
- right PFL3 drive: `-0.9716533526536367`;
- deterministic replay error: `0.0`;
- complete PFL3-output cut peak turn: `0.0`.

Partial relay lesion amplitudes remain descriptive because the current rate model normalizes postsynaptic incoming weights.

## Next gate: transmitter/sign readiness

Do not connect the integration route to the behavioral controller yet.

`configs/integration_sign_audit_v1.json` and `src/fly_sniff/integration_sign_audit.py` now freeze a conservative presynaptic transmitter audit over the exact body-ID populations above. The audit:

1. verifies the exact integration artifact SHA-256;
2. uses the artifact's frozen body IDs rather than re-selecting populations;
3. discovers a recognized neurotransmitter annotation column from the local MaleCNS data;
4. maps ACh to modeled `+1` and GABA to modeled `-1`;
5. leaves glutamate, modulators, missing values, and unknown values at modeled sign `0`;
6. blocks the next signed mechanistic probe if any required presynaptic population remains unresolved.

This is a modeling-readiness audit. Predicted transmitter identity is not synapse-specific receptor physiology.

Run:

```bash
bash scripts/run_integration_sign_audit.sh
```

Expected output:

```text
results/route/integration-sign-audit-v1.json
```

If the public annotation artifact does not contain a recognized neurotransmitter field, that is a legitimate blocked result. Do not substitute a guessed sign or loosen the rule.

## Current claim boundary

Defensible:

> MaleCNS v1.0 contains a body-ID-resolved structural bridge from literature-motivated FB5AB/PFN populations through hDeltaC and a frozen hDeltaG relay into PFL3/PFL2, plus a separately qualified modeled PFL3-to-DNa02 steering scaffold.

Not yet defensible:

- the complete fly smell-navigation circuit;
- physiological signal transmission through hDeltaC -> hDeltaG -> PFL;
- PFN/FB5AB tuning in this individual MaleCNS connectome;
- the inhibitory steering see-saw circuit;
- persistent dynamics from hDeltaK/PFGs structure alone;
- modeled rate state as measured firing;
- intact MaleCNS wiring outperforming matched rewiring in odor navigation.
