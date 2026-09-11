# Data authority

## Canonical dataset

The connectome authority for this project is **Janelia MaleCNS v1.0**, exposed in neuPrint as `male-cns:v1.0` and by the Janelia MaleCNS download portal.

The repository does **not** vendor the multi-gigabyte raw connectome. Derived subgraphs must record:

- upstream dataset name and release (`male-cns:v1.0`);
- extraction timestamp;
- extraction code commit;
- exact query/patterns and minimum synapse threshold;
- node body IDs;
- edge source/target/weight;
- role annotations added by this project;
- SHA-256 digests of the derived files.

## Upstream access paths

- Project portal: `https://male-cns.janelia.org/`
- neuPrint dataset: `male-cns:v1.0`
- Bulk data root documented by Janelia: `gs://flyem-male-cns/v1.0/`
- Python access: `neuprint-python`
- R access: `natverse/malecns`

MaleCNS is structurally rich but is **not** a full physiological specification. In particular, fly-sniff must not silently treat synapse count as a measured conductance, infer receptor kinetics from connectivity, or label modeled rate activity as calcium/spike recordings.

## Circuit qualification workflow

1. **Discover** candidate annotated cell types with `fly-sniff-extract-malecns`.
2. **Inspect** every candidate family and resolve left/right identity.
3. **Trace** pathway coverage between odor/wind input populations, central-complex populations, and descending steering populations.
4. **Seal** selected body IDs into a reviewed `roles.json`.
5. **Extract** exact connectivity into `nodes.parquet` + `edges.parquet`.
6. **Hash** all three artifacts and record them in a circuit manifest.
7. Only a sealed graph may be called `MaleCNS` in benchmark figures or social media.

Until step 7, renderers must use labels such as `candidate`, `discovery`, or `proxy`.
