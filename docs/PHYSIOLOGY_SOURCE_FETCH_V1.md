# Physiology source evidence fetch v1

This utility closes source-ingress blockers without running navigation, fitting model parameters, or downloading full physiology releases by default.

## Setup

From a clean checkout:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## PFN: retrieve only the Dryad README

```bash
fly-sniff-source-fetch pfn-readme \
  --out results/source-ingress/pfn-readme-v1
```

The command is capped at 64 KiB. It expects the public Dryad MD5
`91e5213503788fcde11c0f5aa3e91f43` for `Currier2020README.rtf` and refuses to publish a receipt if the bytes differ.

Expected artifacts:

- `Currier2020README.rtf`
- `pfn-readme-fetch-receipt.json`

The receipt contains the observed SHA-256. That SHA can resolve only the PFN README-byte blocker. It does **not** by itself resolve the archive-member map or the exact Figure 4 recording set.

After retrieval, inspect the README for the multipart archive layout and identify the minimum members required for the 12 Figure 4 P-F2N3 recordings. Do not default to downloading the 117.49 GB release.

## DNa02: retrieve Harvard Dataverse metadata only

```bash
fly-sniff-source-fetch dna02-metadata \
  --contract authority/program-a-dna02-source-contract-v1.json \
  --out results/source-ingress/dna02-dataverse-v1
```

The command calls only the Dataverse persistent-ID metadata endpoint and is capped at 16 MiB. It never requests the contents of any deposited physiology file.

Expected artifacts:

- `dna02-dataverse-raw.json`
- `dna02-dataverse-fetch-receipt.json`
- `dna02-dataverse-review.json`

The review normalizes file IDs, filenames, directory labels, file sizes, content types, restriction flags and repository checksums. It also searches filenames/directories for the candidate aliases and raw session strings already present in the frozen DNa02 source contract.

A filename/session match is **discovery evidence only**. The review deliberately records:

```text
automatic_cohort_resolution = false
automatic_file_map_promotion = false
```

Do not resolve the Figure 3C cohort simply because four candidate aliases exist or because a filename resembles one of the known sessions. Panel-level provenance remains required.

## What these commands are allowed to change

Nothing in the scientific protocol automatically changes after a fetch.

A successful fetch may provide new evidence for a later reviewed authority update. Any source-contract update must occur in a separate commit/PR with the new byte hashes or metadata receipts cited explicitly.

These commands do not:

- inspect odor-navigation performance;
- reveal final evaluation seeds;
- choose or alter calibration statistics;
- modify PFN/DNa02 source contracts;
- download neural data files;
- promote a source contract to calibrated dynamics.
