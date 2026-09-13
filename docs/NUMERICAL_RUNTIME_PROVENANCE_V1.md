# Numerical runtime provenance v1

Task optimization is deterministic only relative to a numerical runtime. A code commit and RNG seed are not sufficient provenance if Python or numerical-library versions can change underneath the same artifact.

This protocol therefore separates two concepts:

1. **Numerical compatibility identity**, which is promotion-critical.
2. **Platform diagnostics**, which are recorded for reproducibility but do not gate promotion.

## Promotion-critical numerical identity

The official task-optimization CLI records the exact versions of:

- Python implementation and patch version;
- NumPy;
- SciPy;
- pandas;
- PyArrow;
- NetworkX;
- Matplotlib.

The canonical numerical identity is SHA-256 hashed. Trained E002 recomputes the current identity and refuses promotion if it differs from the identity that produced the training artifact.

This is deliberately stricter than the package lower bounds in `pyproject.toml`. The lower bounds remain useful for ordinary installation, while the artifact itself tells us exactly which numerical stack produced a scientific result.

## Diagnostic-only platform fields

The receipt also records:

- operating-system family;
- operating-system release;
- machine architecture;
- Python implementation reported by the executable.

These fields are **not** part of the numerical compatibility hash. Changing from one machine or OS to another is not itself scientific evidence that a result is invalid, provided the promotion-critical numerical identity matches and all artifact checks pass.

## Official training boundary

The installed command:

```text
fly-sniff-train-dynamics
```

routes through `fly_sniff.reproducible_training:main`.

The wrapper:

1. captures the numerical runtime before optimization;
2. executes the unchanged task optimizer;
3. captures the runtime again afterward;
4. refuses the artifact if numerical identity changed during the run;
5. seals the same runtime receipt into the intact report, every rewire report, and the lesion report;
6. re-hashes each training audit receipt so runtime identity is cryptographically bound to graph/config/seeds/parameters/budget;
7. persists the sealed artifact.

`optimize_dynamics()` itself remains available as a lower-level implementation function. Its raw return value is not promotion-eligible until it is runtime-sealed. This preserves clean unit testing and experimentation without confusing an exploratory in-memory result with a scientific artifact.

## E002 boundary

`fly-sniff-qualify-trained` requires:

- an intact runtime receipt;
- a matching receipt hash;
- a valid numerical compatibility hash;
- exact compatibility with the current promotion runtime;
- an audit receipt that binds both runtime hashes.

A report with correct graph, parameter, seed, development-gate, and compute-budget hashes still fails if its runtime receipt is absent, forged, or incompatible.

## Trained-final boundary

The matched trained-final path reconstructs every topology and calls the same training-report verifier. Therefore intact, every rewire, and the lesion must all be promotable under the current numerical runtime before final seeds can be evaluated.

The matched artifact itself also contains one shared runtime receipt, making it visible that all topology optimizations were sealed under a single numerical identity.

## What this does not prove

Runtime sealing does not prove that:

- an external process never performed additional experiments;
- the operating system or hardware cannot cause any floating-point difference;
- every BLAS/kernel implementation is bitwise identical across machines;
- the biological model is physiologically correct;
- the final navigation claim is true.

It closes the narrower provenance hole that previously allowed a report to claim deterministic seeded optimization without saying which numerical software stack generated it.

## Next reproducibility layer

With runtime identity sealed, an exact deterministic CEM replay audit can be added under a future red-team tranche. That audit can regenerate seed batches and candidate populations from the frozen optimizer seed, verify candidate parameter hashes, replay elite selection and distribution updates, and compare the final parameter hash.

That stronger replay should be treated as a provenance check, not as a way to tune the model or alter scientific thresholds.
