# DNa02 extraction preflight v1

This lane begins only after the four Figure 3C source files are content-addressed and the DNa02 source contract is `READY_FOR_EXTRACTION`.

It deliberately stops **before** computing a physiology statistic.

## Goal

Authenticate the exact four frozen Harvard Dataverse files on a local machine and inventory their MATLAB schema without exporting neural values. The resulting inspection receipt is the evidence needed to freeze a deterministic portable field mapping before implementing the Figure 3C extractor.

## Scientific boundary

This preflight may establish only:

- the local files are byte-identical to the frozen source evidence;
- the MATLAB container format;
- variable/dataset names, shapes, and dtypes/classes;
- a hash-bound schema receipt.

It does **not** establish calibrated DNa02 dynamics, a universal 150 ms biological delay, transfer of absolute firing-rate gain to MaleCNS, or any odor-navigation result.

## Mac setup

From the repository root:

```bash
git switch feat/dna02-extraction-preflight-v2
git pull --ff-only

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,source]'
```

The `source` extra installs `h5py` only so MATLAB v7.3/HDF5 files can be inventoried without loading their arrays.

## 1. Download or verify the exact four files

The repository already freezes their Harvard Dataverse file IDs, sizes, MD5 values, and SHA-256 identities. The downloader writes each file only after its complete byte stream passes all three checks. Existing files are re-hashed and reused only if they match exactly.

```bash
mkdir -p data/raw/dna02
fly-sniff-dna02-download --out data/raw/dna02
```

Expected total source size is 929,441,392 bytes. A receipt is written to:

```text
data/raw/dna02/dna02-local-source-fetch-receipt.json
```

Do not rename the four `.mat` files before inspection because the frozen filename is part of source identity.

## 2. Inventory the MATLAB schema

```bash
mkdir -p data/cache/dna02-preflight
fly-sniff-dna02-inspect \
  data/raw/dna02/180410_gfp_3G_ss730_dual_08_data_for_SH_with_lat_vel.mat \
  data/raw/dna02/180430_gfp_3G_ss730_dual_12_data_for_SH_with_lat_vel.mat \
  data/raw/dna02/180501_gfp_3G_ss730_dual_13_data_for_SH_with_lat_vel.mat \
  data/raw/dna02/180517_gfp_3G_ss730_dual_14_data_for_SH_with_lat_vel.mat \
  --out data/cache/dna02-preflight/source-inspection-v1.json
```

The command hashes every file again before inspecting metadata. It does not export raw values and does not compute the Figure 3C relationship.

## 3. Send back the tiny receipt, not the neural files

The useful handoff is:

```text
data/cache/dna02-preflight/source-inspection-v1.json
```

That receipt is small enough to attach to the project or paste into a follow-up review. The next code tranche will use it to freeze the exact source-field mapping before any numeric statistic is extracted.

## What comes after the receipt

Only after schema review should the extractor be frozen. Its already-prespecified scientific target remains:

- relative-prominence spike detection as defined by the publication/method authority;
- 10 ms firing-rate bins;
- 30 ms exponential firing-rate smoothing;
- 150 ms neural-to-behavior alignment for publication reproduction only;
- 50 ms Figure 3 averaging;
- predictor: right DNa02 firing rate minus left DNa02 firing rate;
- outcome: rotational velocity;
- a normalized relational summary rather than blind transfer of absolute gain.

Any source-field or preprocessing ambiguity discovered during preflight must block extraction for review rather than be resolved by looking at navigation performance.
