# DNa02 Figure 3C cohort adjudication v1

## Question

Which four bilateral DNa02 recordings constitute the `n=4` cohort shown in Figure 3C of Rayshubskiy et al. (eLife 2025, DOI `10.7554/eLife.102230.3`)?

This decision is made before any Program A navigation performance and does not use model behavior.

## Evidence used

The version-of-record states that Figure 3 is based on dual recordings from the right and left copies of DNa02 and that Figure 3C contains one line per fly, `n=4`.

The released Harvard Dataverse dataset (`10.7910/DVN/0NCLP1`, version 1.2) was queried through its persistent-ID metadata API. The exact response is frozen by SHA-256 in `authority/program-a-dna02-cohort-adjudication-v1.json`. Among 380 released files, there are exactly four raw folders matching the bilateral-DNa02 naming convention `ephys_data_a2_d_*`: `a2_d_08`, `a2_d_12`, `a2_d_13`, and `a2_d_14`.

The independently pinned secondary-analysis repository at commit `7e2895349266b5cc5fa1bf53ad56e8ecc6c842e8` declares those same four aliases as completed bilateral DNa02 inputs in both its import/preprocessing workspace and its kinematic-variable/firing-rate relationship workspace.

The conjunction of publication cardinality, exact released-data cardinality, and independent code set identity resolves the Figure 3C cohort to those four aliases.

## `a2_d_14` low-SNR note

`physiology_quant_analysis_figures.ipynb` marks `a2_d_14` low SNR/rejected for a separate physiology-quantification analysis. This is retained as conflicting/QC evidence. It is not treated as panel-level evidence that the recording was excluded from Figure 3C.

Cohort inclusion therefore must **not** be interpreted as a claim that `a2_d_14` passes every analysis-specific QC criterion in the repository.

## What remains blocked

Dataverse currently supplies MD5 checksums for the four raw `.mat` files. The Program A source contract requires exact SHA-256 identities of the consumed neural-data bytes.

Therefore the cohort is resolved, but the DNa02 source contract remains `BLOCKED_DATAVERSE_FILE_MAP_UNRESOLVED` and is not yet `READY_FOR_EXTRACTION`.

No physiology trace has been used for calibration in this adjudication, and no odor-navigation result has been viewed.
