# Statistical evaluation contract

The public animation is a visualization of a scientific benchmark, not the benchmark itself. Claims about biological wiring must come from the sealed held-out evaluation defined here and in the final manifest.

## Paired experimental design

Controllers are compared on the **same episode seeds**. For each seed, the plume realization, starting distribution, arena configuration, and other exogenous benchmark parameters are matched. This makes the biological-versus-rewire comparison paired rather than a comparison of unrelated episode samples.

A hero episode may illustrate behavior but must never substitute for the paired cohort.

## Success

An episode succeeds when the agent enters the circular source region of radius `source_radius` before the episode limit.

The per-controller success rate is the empirical mean of this binary outcome across the sealed seed set.

For MaleCNS versus rewire, the report also contains a paired success comparison:

- MaleCNS success rate;
- rewire success rate;
- paired success-rate delta;
- paired percentile-bootstrap 95% CI for that delta;
- `MaleCNS only` discordant successes;
- `rewire only` discordant successes;
- exact two-sided McNemar/binomial p-value on discordant pairs.

This paired success analysis is additional inference. It does not replace preregistered gold gates after results are observed.

## SPL

Success weighted by Path Length is

\[
\mathrm{SPL}=S\frac{L^*}{\max(L^*,L)},
\]

where

- \(S\in\{0,1\}\) is episode success;
- \(L\) is the realized path length;
- \(L^*\) is the shortest valid path to the **success region**, not the source center.

Because the benchmark arena has no internal obstacles, the shortest path to the circular goal region is

\[
L^*=\max(d_{center}-r_{goal},0).
\]

Using center distance directly would over-credit near-direct trajectories because the agent is allowed to stop at the source radius.

## Paired SPL effect

For each held-out seed \(i\), define

\[
d_i = \mathrm{SPL}_{MaleCNS,i}-\mathrm{SPL}_{rewire,i}.
\]

The reported effect is

\[
\bar d=\frac{1}{N}\sum_i d_i.
\]

A deterministic paired percentile bootstrap resamples the paired differences and reports the 2.5th and 97.5th percentiles as the 95% interval.

The repository's current gold gate requires the preregistered SPL delta threshold and a lower confidence bound above zero. This gate must not be changed after seeing final-test results.

## OOD evaluation

The sealed manifest defines a separate out-of-distribution plume configuration and seed set. OOD performance is reported separately from in-distribution held-out performance.

No OOD setting should be selected after inspecting final results to make one controller look better.

## Rewire control

The rewired graph is generated from the sealed biological graph with the preregistered rewiring seed and settings. Rewiring must preserve the invariants promised by `degree_preserving_rewire` and must be validated before evaluation.

The public phrase `SCRAMBLED WIRING` is acceptable only when the rendered controller actually corresponds to this sealed rewired graph. A random-walk development control must remain labeled as a random/development control.

## Confidence intervals and p-values

Confidence intervals are primary for effect size interpretation. The exact McNemar p-value is included as a paired binary-outcome diagnostic, not as a magic pass/fail oracle.

Do not convert `p < 0.05` into a claim that the model reproduces biological olfactory navigation. The benchmark can only establish performance under its explicit simulation assumptions.

## Multiple comparisons

The final claim is centered on the preregistered MaleCNS-versus-rewire metrics. Additional exploratory comparisons, ablations, cell-family breakdowns, or visualization-selected examples must be labeled exploratory unless they were specified before final evaluation.

## Public reporting rules

A final result card may show:

- held-out episode count;
- MaleCNS success rate;
- rewire success rate;
- paired success-rate delta and CI;
- paired SPL delta and CI;
- OOD success rate;
- exact manifest / receipt identifier.

It must not:

- infer a cohort claim from one video episode;
- hide failed or timeout episodes from the underlying cohort;
- swap to a more favorable seed set after evaluation;
- report a rewired result from a different plume set than the biological graph;
- replace a null result with a different metric after unblinding;
- imply that benchmark success validates real-world odor chemistry, room CFD, or measured neural dynamics.

## Null results

A null result is a valid outcome. If the sealed analysis shows no robust MaleCNS advantage, the correct public conclusion is a null or equivalence-style statement supported by the measured effect and interval. The visualization must not convert a null cohort into a positive claim by choosing an unusually successful hero episode.
