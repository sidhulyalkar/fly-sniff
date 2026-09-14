# Final entropy protocol v1

The experiment kernel distinguishes **not using final seeds** from making final seeds **unknowable before model freeze**.

Readable, deterministic final seed lists are useful for reproducibility after a run, but they do not prove that a developer could not inspect those exact environments while making modeling choices. The v1 final-entropy protocol therefore uses a commit-reveal sequence.

## Security / scientific goal

Before the confirmatory model and experiment are frozen, an independent party or protected environment holds at least 32 bytes of random secret entropy.

Only a domain-separated SHA-256 commitment is published:

```text
commitment = SHA256("fly-sniff-final-entropy-commitment-v1" || secret)
```

The raw secret is not emitted by `fly-sniff-final-entropy commit`.

The exact serialized commitment file is then included in the `ExperimentSpec` as an `ArtifactRef` with:

```text
kind = final_entropy_commitment
sha256 = <SHA-256 of exact commitment JSON bytes>
```

That spec is incorporated into `ExperimentLock`. After code, runtime, circuit, dynamics, environment, metrics, and other claim-bearing assumptions are frozen, the secret may be revealed.

Final seeds are then deterministically derived with HMAC-SHA256 from:

```text
secret + exact ExperimentLock hash + namespace + counter
```

The derivation uses rejection sampling into the requested integer range and removes duplicate seeds. Different experiment locks or namespaces therefore produce different deterministic streams from the same revealed secret.

## Why the commitment file itself is bound

Binding only the commitment digest in prose is insufficient. A developer could otherwise create a new commitment after seeing a frozen model.

The registered CLI therefore refuses derivation unless:

1. the commitment's experiment ID matches the ExperimentLock;
2. the ExperimentSpec contains exactly one `final_entropy_commitment` artifact;
3. that ArtifactRef SHA-256 matches the **exact bytes** of the supplied commitment JSON;
4. the revealed secret matches the precommitted digest.

The post-reveal `FinalSeedReceipt` additionally binds the exact ExperimentLock hash, seed namespace, seed bounds, complete ordered seed list, and seed-list hash.

## Recommended workflow

Generate the random secret outside the repository. For example, on a trusted machine:

```bash
openssl rand 32 > /secure/location/fly-sniff-final-secret.bin
```

Do not commit that file, copy it into experiment artifacts, or expose it to an optimization process.

Create the public commitment:

```bash
fly-sniff-final-entropy commit \
  --secret-file /secure/location/fly-sniff-final-secret.bin \
  --experiment-id odor-zero-shot-v1 \
  --output artifacts/final-entropy-commitment.json
```

The command prints the exact file SHA-256 but never prints the secret. Add that exact file as a `final_entropy_commitment` ArtifactRef before creating the confirmatory `ExperimentLock`.

After the frozen lock is independently archived/reviewed, reveal the secret and derive a namespace:

```bash
fly-sniff-final-entropy derive \
  --secret-file /secure/location/fly-sniff-final-secret.bin \
  --commitment artifacts/final-entropy-commitment.json \
  --lock artifacts/experiment-lock.json \
  --namespace heldout \
  --count 1000 \
  --output artifacts/final-heldout-seeds.json
```

Derive OOD seeds with a distinct namespace such as `ood`.

Anyone with the revealed secret can later verify the exact commitment and derivation:

```bash
fly-sniff-final-entropy verify \
  --secret-file revealed-secret.bin \
  --commitment artifacts/final-entropy-commitment.json \
  --lock artifacts/experiment-lock.json \
  --seed-receipt artifacts/final-heldout-seeds.json
```

## What this proves and does not prove

The protocol makes it externally checkable that a particular final seed stream was determined by entropy committed before the experiment lock and by the exact lock itself.

It does **not** prove that no human ran an unrelated benchmark, copied the repository, changed hardware, or otherwise violated the protocol outside the recorded workflow. Strong claims should therefore combine this mechanism with external archival timestamps, a one-way final-run receipt, retained negative results, and ideally independent review or execution.

This protocol also does not make final seeds scientifically representative by itself. Environment distributions, OOD definitions, sample sizes, topology-null ensembles, and statistical estimands still require independent preregistration.
