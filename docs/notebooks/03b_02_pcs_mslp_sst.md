# 03B — MSLP/SST PCA (`02_PCs_MSLP_SST.ipynb`)

**Open in Jupyter:** `03_NC_Wave_Emulator/03B_Century_GCMs_Waves/02_PCs_MSLP_SST.ipynb`

## Role in the emulator

Builds **Principal Component Analysis** of large-scale **MSLP** and **SST** fields (20CR / ERA5 training period):

- EOF spatial patterns + daily PC time series
- Outputs under `outputs/PCA/` used by `03_Clusters_and_PCS.ipynb` and ALR notebooks

This implements the atmospheric side of the Camus et al. (2014) wave-emulator framework: synoptic patterns → PCs → wave clusters.

## Prerequisites

- Reanalysis MSLP/SST prepared (same domain as NC emulator)
- Wave clusters selected in `01_Testing_ClusteringAlgorithms_partitions_all.ipynb`

## Next step

→ [`03_Clusters_and_PCS.ipynb`](03b_03_clusters_and_pcs.html) — calibrate cluster occurrence vs PCs

---

**Note:** The live `.ipynb` currently has **truncated outputs** (invalid JSON). Clear outputs in Jupyter and re-save, or run fresh, before adding to automated builds.
