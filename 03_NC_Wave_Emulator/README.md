# 03_NC_Wave_Emulator — North Carolina wave climate emulator

Future wave climate emulation for the ShoreShop North Carolina domain. Two complementary pipelines:

| Subfolder | Purpose |
|-----------|---------|
| [`03A_Stochastic_GCMs_NC`](03A_Stochastic_GCMs_NC/README.md) | CMIP6 preprocessing, MSLP/SST PCA, **tropical-cyclone track-type** MLR emulation |
| [`03B_Century_GCMs_Waves`](03B_Century_GCMs_Waves/README.md) | **Wave-cluster** KMA + ALR emulator driven by large-scale MSLP PCs and historical hindcast |

Upstream dependencies: [`01_BinWaves`](../01_BinWaves/README.md) (offshore spectra) and [`02_Wind_Metamodel`](../02_Wind_Metamodel/README.md) (BinWaves + KMA bulk hindcast).

---

## End-to-end flow

```text
01_BinWaves                          02_Wind_Metamodel
(corrected offshore spectra)         (BinWaves + BMUS bulk)
        │                                    │
        └──────────────┬─────────────────────┘
                       ▼
              03B — wave partitions @ 4 super points
              03B — KMA wave clusters (80 BMUs)
              03B — ERA5/20CR MSLP PCA + ALR training
                       │
         03A — CMIP6 MSLP/SST PCA (bias-corrected)
         03A — TC track-type MLR  ──►  (parallel TC branch)
                       │
                       ▼
              03A: mslp_*_pcs_95.csv  ──►  03B/05_ALR_* (read directly from 03A/outputs)
                       │
                       ▼
              03B — ALR GCM notebooks (05_ALR_Clusters_GCMs_*)
              simulate wave BMUs + bootstrap bulk from 02 hindcast
                       │
                       ▼
              outputs/{gcm}_{ssp}/OffshorePoints/*.nc
              outputs/{gcm}_{ssp}/*_500m.nc
```

---

## Cross-pipeline connections

### `02_Wind_Metamodel` → `03B`

| Source (02) | Used in 03B | Purpose |
|-------------|-------------|---------|
| **`outputs/NorthCarolina/{var}_NorthCarolina.nc`** | `05_ALR_Clusters_GCMs_*.ipynb` | Historical reference for validation; site coordinates for BMU bootstrap |
| `outputs/merged_500m_binwaves_bmus/{var}_500m.nc` | Optional in BMU bootstrap (`product="500m"`) | Lighter 500 m reference product instead of full domain |
| `gridN/outputs/BinWaves_BMUS/` | Indirect (via NorthCarolina merge) | Source of merged hindcast bulk |

Minimum before running 03B ALR validation: complete `02` notebook **`01A_postProcessing_binwaves_bmus_all_grids.ipynb`**.

### `01_BinWaves` → `03B`

| Source (01) | Used in 03B | Purpose |
|-------------|-------------|---------|
| **`gridN/inputs/*_spec_WHACS_buoy_correted_15D.nc`** | `00_Partitions_cluster_points.ipynb` | Offshore corrected spectra at one point per grid |
| `gridN/inputs/gridN_superPoint_15D.nc` | Same (grid2, grid3) | SuperPoint spectra |
| `inputs/WHACS/north_carolina_*_uwnd/vwnd_WHACS.nc` | Partitioning (wind-sea split) | WHACS winds for PTM4 |
| `inputs/gebco_bathymetry.nc` | Partitioning | Depth at super points |

### `03A` → `03B`

| Source (03A) | Destination (03B) | Purpose |
|--------------|-------------------|---------|
| **`03A_Stochastic_GCMs_NC/outputs/cmip6_models/{model}/{ssp}/mslp_{model}_{ssp}_pcs_95.csv`** | **`05_ALR_Clusters_GCMs_*.ipynb`** | Future MSLP PCs fed as covariates to the wave-cluster ALR emulator |

No copy step needed — 03B notebooks read directly from 03A outputs after running `04_pcs_mslp.ipynb`.

**Note:** 03A tropical-cyclone outputs (`outputs/tc_logit/`) feed the cyclone metamodel in [`04A`](../../04_NC_Cyclones/04A_Cyclones_Metamodel/README.md) (notebook `13_reconstrucion_trazas.ipynb`). They are not required for the core 03B wave-cluster ALR workflow.

---

## CMIP6 models (both pipelines)

| Model key | CMIP6 name | SSP scenarios |
|-----------|------------|---------------|
| `access_esm1_5` | ACCESS-ESM1-5 | ssp245, ssp585 |
| `ec_earth3_veg_lr` | EC-Earth3-Veg | ssp245, ssp585 |
| `ipsl_cm6a_lr` | IPSL-CM6A-LR | ssp245, ssp585 |
| `miroc6` | MIROC6 | ssp245, ssp585 |
| `mpi_esm1_2_hr` | MPI-ESM1-2-HR | ssp245, ssp585 |

---

## Quick run order

```text
# Prerequisites
01_BinWaves  →  complete buoy-corrected spectra per grid
02_Wind_Metamodel  →  01A merge  →  outputs/NorthCarolina/

# 03B — historical emulator core
03B/00_Partitions_cluster_points.ipynb
03B/01_Testing_ClusteringAlgorithms_partitions_all.ipynb
03B/02_PCs_MSLP_SST.ipynb
03B/03_Clusters_and_PCS.ipynb
03B/04_ALR_Clusters_20CR.ipynb

# 03A — GCM MSLP PCs (for future forcing)
03A/01_preprocess_mslp*.ipynb  (per model)
03A/04_pcs_mslp.ipynb   # produces pcs_95 CSVs consumed directly by 03B

# 03B — future scenarios
03B/05_ALR_Clusters_GCMs_{model}_{245|585}.ipynb  (one per GCM/SSP)
```

---

## Sub-documentation

- [`03A_Stochastic_GCMs_NC/README.md`](03A_Stochastic_GCMs_NC/README.md) — CMIP6 preprocessing + TC stochastic emulation
- [`03B_Century_GCMs_Waves/README.md`](03B_Century_GCMs_Waves/README.md) — wave-cluster KMA/ALR emulator + GCM projections
