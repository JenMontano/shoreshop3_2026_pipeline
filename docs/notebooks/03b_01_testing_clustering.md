# 03B — Wave-cluster selection (`01_Testing_ClusteringAlgorithms_partitions_all.ipynb`)

**Open in Jupyter:** `03_NC_Wave_Emulator/03B_Century_GCMs_Waves/01_Testing_ClusteringAlgorithms_partitions_all.ipynb`

## Role in the emulator

After partition statistics are built at four **super points** (see [partitions at cluster points](03b_00_partitions_cluster_points.html)), this notebook:

1. Loads hindcast partition series from `outputs/partitions_SuperPoint/`
2. Tests KMA clustering algorithms and BMU counts
3. Selects the **wave-cluster catalogue** (80 BMUs) used downstream in ALR training

This is the **core 03B step** linking BinWaves+BMUS hindcast partitions to the stochastic wave emulator.

## Prerequisites

- **02** merged hindcast (`02/01A` or partition inputs used in 03B/00)
- `00_Partitions_cluster_points.ipynb` completed → `outputs/partitions_SuperPoint/`

## Next step

→ [`02_PCs_MSLP_SST.ipynb`](03b_02_pcs_mslp_sst.html) — atmospheric PCA for ALR forcing

---

**Note:** The live `.ipynb` on disk currently has **truncated cell outputs** (file ends mid-JSON). Open and run it in Jupyter, or clear all outputs and save, to restore a valid notebook file. A renderable companion for the partition setup step is included in this book as **03B — partitions at super points**.
