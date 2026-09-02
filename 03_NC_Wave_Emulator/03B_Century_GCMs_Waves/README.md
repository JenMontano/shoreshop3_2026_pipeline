# 03B — Century GCM wave emulator (KMA clusters + ALR)

Historical and future wave climate emulation for North Carolina using **80 wave behavioural map units (BMUs)**. Combines:

1. Partition time series at four offshore super points (from BinWaves spectra)
2. KMA clustering on wave + partition features
3. MSLP PCA covariates (ERA5 / 20CR historical; CMIP6 future from [`03A`](../03A_Stochastic_GCMs_NC/README.md))
4. ALR (autoregressive logistic regression) to simulate BMU sequences under GCM forcing
5. BMU bootstrap of bulk wave parameters from the [`02_Wind_Metamodel`](../../02_Wind_Metamodel/README.md) hindcast

---

## Upstream dependencies

### From `01_BinWaves` (required for step 0)

| Input | Path | Notebook |
|-------|------|----------|
| Corrected offshore spectra (grid1, grid4) | `01_BinWaves/gridN/inputs/*_spec_WHACS_buoy_correted_15D.nc` | `00_Partitions_cluster_points.ipynb` |
| SuperPoint spectra (grid2, grid3) | `01_BinWaves/gridN/inputs/gridN_superPoint_15D.nc` | Same |
| WHACS winds | `01_BinWaves/inputs/WHACS/north_carolina_*_{uwnd,vwnd}_WHACS.nc` | Same |
| GEBCO bathymetry | `01_BinWaves/inputs/gebco_bathymetry.nc` | Same |

### From `02_Wind_Metamodel` (required for step 5 / validation)

| Input | Path | Used for |
|-------|------|----------|
| **Merged BinWaves+BMUS bulk** | **`02_Wind_Metamodel/outputs/NorthCarolina/{var}_NorthCarolina.nc`** | Historical reference; BMU bootstrap site grid (`05_ALR_*`) |
| 500 m merge (optional) | `02_Wind_Metamodel/outputs/merged_500m_binwaves_bmus/{var}_500m.nc` | Alternative bootstrap product |

### From `03A` (required for GCM projections)

| Input | Path | Used for |
|-------|------|----------|
| **GCM MSLP PCs** | **`../03A_Stochastic_GCMs_NC/outputs/cmip6_models/{model}/{ssp}/mslp_{model}_{ssp}_pcs_95.csv`** | Future covariates in `05_ALR_Clusters_GCMs_*.ipynb` |

Produced by `03A/04_pcs_mslp.ipynb`; read directly (no copy step).

---

## Directory structure

```text
03B_Century_GCMs_Waves/
├── inputs/
│   ├── global_mslp_1day_1degree.nc    # ERA5 MSLP (1° daily)
│   ├── estela_sea.nc, uwnd.nc, vwnd.nc
│   └── 20CR/                          # 20th Century Reanalysis
├── Projections/PCA/                   # Legacy copies (optional; notebooks read 03A directly)
├── outputs/
│   ├── partitions_SuperPoint/         # 4-point partition NetCDFs
│   ├── KMA/                           # Wave-cluster centroids + BMU time series
│   ├── PCA/                           # Historical MSLP PCs + training CSV
│   ├── Emulator/ALR/                  # Trained ALR model (20CR)
│   ├── {gcm}_{ssp}/                   # Simulated bulk per scenario
│   └── Figures/
├── utils/
│   ├── bmu_bootstrap_timeseries.py    # BMU → bulk wave reconstruction
│   ├── alr_plotting.py
│   └── gcm_comparison.py
├── 00_Partitions_cluster_points.ipynb
├── 01_Testing_ClusteringAlgorithms_partitions_all.ipynb
├── 02_PCs_MSLP_SST.ipynb
├── 03_Clusters_and_PCS.ipynb
├── 04_ALR_Clusters_20CR.ipynb         # Train emulator on historical
└── 05_ALR_Clusters_GCMs_{model}_{245|585}.ipynb  # One per GCM/SSP
```

---

## Pipeline steps

### 0. Partitions at super points

**Notebook:** `00_Partitions_cluster_points.ipynb`

- One offshore point per grid (4 total); PTM4 partitioning on buoy-corrected / SuperPoint spectra
- **Inputs:** `01_BinWaves/gridN/inputs/` spectra, WHACS winds, GEBCO
- **Outputs:** `outputs/partitions_SuperPoint/{hs,tp,dp,phs,ptp,pdp,spr}_grid{N}.nc`, `{var}_NorthCarolina.nc`

### 1. Wave clustering (KMA)

**Notebook:** `01_Testing_ClusteringAlgorithms_partitions_all.ipynb`

- MDA → KMA on partition + bulk features (32 variables × 4 grids)
- **80 wave BMUs** (`kma_bmus` 0–79)
- **Outputs:** `outputs/KMA/centroids_kma_sorted.csv`, `nearest_centroids_idxs_kma_sorted.csv`, etc.

### 2. Historical MSLP PCs

**Notebook:** `02_PCs_MSLP_SST.ipynb`

- PCA on ERA5 MSLP (+ ESTELA gradient predictor)
- 20CR extension and bias correction
- **Outputs:** `outputs/PCA/pcs_mslp_df.csv`, `pca_mslp_model.pkl`, `20cr_era5_pcs.csv`, `eofs_mslp.nc`

### 3. Combine wave BMUs + MSLP PCs

**Notebook:** `03_Clusters_and_PCS.ipynb`

- Merge daily MSLP PCs with sorted wave BMU assignments
- **Output:** **`outputs/PCA/pcs_clusters_ordered_ERA5.csv`** (PC columns + `kma_bmus` target)

### 4. Train ALR emulator (historical)

**Notebook:** `04_ALR_Clusters_20CR.ipynb`

- Fit ALR model: wave BMU ~ lagged BMUs + MSLP PCs
- Calibrate on 20CR / ERA5 period (from 1980)
- **Outputs:** `outputs/Emulator/ALR/model.sav`, `terms.sav`, `xds_input.nc`, `xds_output.nc`

### 5. GCM projections + bulk bootstrap ★

**Notebooks:** `05_ALR_Clusters_GCMs_{model}_{245|585}.ipynb`

One notebook per GCM and SSP (e.g. `05_ALR_Clusters_GCMs_access_585.ipynb`).

| Step | Input | Output |
|------|-------|--------|
| Load training CSV | `outputs/PCA/pcs_clusters_ordered_ERA5.csv` | — |
| Load GCM forcing | `../03A_Stochastic_GCMs_NC/outputs/cmip6_models/{model}/{ssp}/mslp_{model}_{ssp}_pcs_95.csv` | Simulated BMU sequences |
| Run ALR | `outputs/Emulator/ALR/` | Per-member BMU time series |
| BMU bootstrap | **`../02_Wind_Metamodel/outputs/NorthCarolina/`** | **`outputs/{gcm}_{ssp}/OffshorePoints/{var}.nc`** |
| Optional 500 m crop | Same hindcast or `merged_500m_binwaves_bmus` | `outputs/{gcm}_{ssp}/{var}_500m.nc` |

The bootstrap step (`utils/bmu_bootstrap_timeseries.py`) maps each simulated BMU hour to the corresponding wave bulk from the historical hindcast at each site.

**Validation plots** compare simulated vs historical bulk from `02_Wind_Metamodel/outputs/NorthCarolina/`.

---

## Outputs (final products)

| Path | Description |
|------|-------------|
| `outputs/{gcm}_{ssp}/OffshorePoints/*.nc` | Emulated bulk/partition time series at hindcast sites |
| `outputs/{gcm}_{ssp}/*_500m.nc` | Cropped 500 m reference-point products |
| `outputs/Figures/ALR/{gcm}_{ssp}/` | BMU transition / perpetual-year diagnostics |
| `outputs/Figures/ALR_bulk_bootstrap_month/` | Historical vs simulated bulk validation |

Example scenario folders: `access_245`, `access_585`, `mpi_esm1_245`, `ipsl_585`, …

---

## GCM notebooks

| Notebook pattern | GCM | SSP |
|------------------|-----|-----|
| `05_ALR_Clusters_GCMs_access_{245\|585}.ipynb` | ACCESS-ESM1-5 | 2.6 / 8.5 |
| `05_ALR_Clusters_GCMs_earth3_veg_{245\|585}.ipynb` | EC-Earth3-Veg | 2.6 / 8.5 |
| `05_ALR_Clusters_GCMs_ipsl_cm6a_lr_{245\|585}.ipynb` | IPSL-CM6A-LR | 2.6 / 8.5 |
| `05_ALR_Clusters_GCMs_miro6_{245\|585}.ipynb` | MIROC6 | 2.6 / 8.5 |
| `05_ALR_Clusters_GCMs_mpi_esm1_{245\|585}.ipynb` | MPI-ESM1-2-HR | 2.6 / 8.5 |

Requires matching `03A/outputs/cmip6_models/{model}/{ssp}/mslp_*_pcs_95.csv`.

---

## Quick run order

```text
# Prerequisites
01_BinWaves  →  corrected spectra in gridN/inputs/
02_Wind_Metamodel  →  01A  →  outputs/NorthCarolina/

# 03B historical emulator
00_Partitions_cluster_points.ipynb
01_Testing_ClusteringAlgorithms_partitions_all.ipynb
02_PCs_MSLP_SST.ipynb
03_Clusters_and_PCS.ipynb
04_ALR_Clusters_20CR.ipynb

# 03A GCM PCs (if not done)
03A/04_pcs_mslp.ipynb   # produces pcs_95 CSVs read directly by 03B

# 03B per GCM/SSP
05_ALR_Clusters_GCMs_{model}_{245|585}.ipynb
```

---

## Troubleshooting

| Issue | Check |
|-------|-------|
| Missing GCM MSLP PCs | Run 03A `04_pcs_mslp.ipynb` for the model/SSP |
| BMU bootstrap fails | Verify `02_Wind_Metamodel/outputs/NorthCarolina/hs_NorthCarolina.nc` exists |
