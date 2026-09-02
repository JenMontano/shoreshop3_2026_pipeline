# 03A — Stochastic GCMs (MSLP / SST / tropical cyclones)

Preprocesses CMIP6 MSLP and SST for the North Atlantic, computes bias-corrected principal components, and emulates **tropical-cyclone (TC) track-type** occurrence using multivariate logistic regression (MLR/ALR). Outputs the **future MSLP PCs** consumed by the wave emulator in [`03B`](../03B_Century_GCMs_Waves/README.md).

---

## Role in the ShoreShop chain

```text
CMIP6 MSLP/SST (external)
        │
        ▼
03A: preprocess → bias correct → PCA
        │
        ├──► outputs/cmip6_models/.../mslp_*_pcs_95.csv  ──► 03B/05_ALR_* (direct read)
        │
        └──► outputs/tc_logit/...  (TC track-type branch; → 04A/13 → 04B cyclone splice)
```

This folder does **not** read directly from `01_BinWaves` or `02_Wind_Metamodel`. Its main handoff to the rest of ShoreShop is the GCM MSLP PC time series for 03B.

---

## Directory structure

```text
03A_Stochastic_GCMs_NC/
├── inputs/
│   ├── ibtracs_na_tracks.nc          # IBTrACS NA hurricane tracks
│   ├── track_types_hist.csv          # Historical TC track-type labels
│   ├── temperature_increase/         # Global warming level metadata
│   └── change_in_intensity.csv
├── outputs/
│   ├── era5/                         # ERA5 reference PCs
│   ├── cmip6_models/{model}/{ssp}/   # Bias-corrected GCM fields + PCs
│   ├── tc_logit/{model}/{ssp}/       # Simulated TC track types + emu tracks
│   └── figures/
├── utils/
│   ├── preprocess_gcms.py
│   ├── gcms_tcs.py
│   └── var_utils.py
├── 01_preprocess_mslp*.ipynb         # One per GCM (mpi, ipsl, default)
├── 02_regrid_gcms_sst.ipynb
├── 03_preprocess_sst.ipynb
├── 04_pcs_mslp.ipynb                 # ★ produces pcs_95 CSVs for 03B
├── 05_pcs_sst.ipynb
├── 06_mlr_tcs.ipynb                  # TC track-type MLR
├── 07_assing_tcs.ipynb               # Assign synthetic TC tracks
├── 08_projected_changes.ipynb
└── A0–A2_figures_*.ipynb             # Validation / figure notebooks
```

---

## External inputs

| Input | Source | Used in |
|-------|--------|---------|
| CMIP6 MSLP | `/lustre/geocean/DATA/PROJECTIONS/MSLP/CMIP6` | `01_preprocess_mslp*.ipynb` |
| CMIP6 SST | Regridded in `02_regrid_gcms_sst.ipynb` | `03_preprocess_sst.ipynb` |
| ERA5 MSLP (1° daily) | Shared with 03B: `global_mslp_1day_1degree.nc` | `04_pcs_mslp.ipynb` (reference PCA) |
| IBTrACS NA tracks | `inputs/ibtracs_na_tracks.nc` | TC MLR (`06_mlr_tcs.ipynb`) |
| Synthetic TC pool | External VAR/track-gen datasets | `07_assing_tcs.ipynb` |

---

## Pipeline steps

### 1. Preprocess MSLP (per GCM)

**Notebooks:** `01_preprocess_mslp.ipynb`, `01_preprocess_mslp_mpi.ipynb`, `01_preprocess_mslp_ipsl.ipynb`

- Mask, regrid to 1°, bias-correct against ERA5
- **Outputs:** `outputs/cmip6_models/{model}/{ssp}/mslp_{model}_{ssp}_1deg_bias_corrected.nc`

Configure `model_import` / `model_export` and `ssp_import` / `ssp_export` in each notebook.

### 2. Preprocess SST

**Notebooks:** `02_regrid_gcms_sst.ipynb` → `03_preprocess_sst.ipynb`

- Regrid CMIP6 SST, bias-correct
- **Outputs:** `outputs/cmip6_models/{model}/{ssp}/sst_{model}_{ssp}_2deg_masked_bias_corrected.nc`

### 3. Principal components — MSLP ★ (handoff to 03B)

**Notebook:** `04_pcs_mslp.ipynb`

1. Fit PCA on ERA5 MSLP (+ gradient, ESTELA predictor)
2. Project bias-corrected GCM MSLP onto ERA5 EOFs
3. Export 95% variance-retaining PCs

**Outputs:**

| File | Description |
|------|-------------|
| `outputs/era5/era5_mslp_pcs_95.csv` | ERA5 reference PCs |
| **`outputs/cmip6_models/{model}/{ssp}/mslp_{model}_{ssp}_pcs_95.csv`** | **Read directly by `03B/05_ALR_Clusters_GCMs_*.ipynb`** |

### 4. Principal components — SST

**Notebook:** `05_pcs_sst.ipynb`

- **Outputs:** `outputs/cmip6_models/{model}/{ssp}/sst_{model}_{ssp}_pcs.csv`, `outputs/era5/era5_sst_pcs.csv`

### 5. TC track-type MLR

**Notebook:** `06_mlr_tcs.ipynb`

- Multivariate logistic regression: TC track-type occurrence ~ MSLP PCs (+ SST PCs)
- Trained on IBTrACS + ERA5 PCs
- **Outputs:** `outputs/tc_logit/{model}/{ssp}/simulated_tracktypes_{model}_{ssp}.nc`

### 6. Assign synthetic TC tracks

**Notebook:** `07_assing_tcs.ipynb`

- Match emulated track types to synthetic track pool
- **Outputs:**
  - `outputs/tc_logit/{model}/{ssp}/emu_tracks_{model}_{ssp}.nc`
  - `outputs/tc_logit/{model}/{ssp}/simulated_tracktypes_with_tc_id_{model}_{ssp}.nc`

### 7. Projected changes & figures

**Notebooks:** `08_projected_changes.ipynb`, `A0`–`A2`

- TC frequency/intensity change analysis and validation plots

---

## Outputs → other folders

| Output | Destination | Required for |
|--------|-------------|--------------|
| **`mslp_{model}_{ssp}_pcs_95.csv`** | `03B/05_ALR_Clusters_GCMs_*.ipynb` (via `../03A_Stochastic_GCMs_NC/outputs/...`) | Future MSLP covariates |
| `tc_logit/{model}/{ssp}/emu_tracks_*.nc` | [`04A/13`](../../04_NC_Cyclones/04A_Cyclones_Metamodel/README.md) → [`04B/inputs`](../../04_NC_Cyclones/04B_Cyclones_emulator/README.md) | TC cyclone metamodel & splice |
| `sst_{model}_{ssp}_pcs.csv` | — | Optional; TC MLR uses SST PCs internally |

---

## CMIP6 models

Run preprocessing + `04_pcs_mslp` for each:

| Export name | Import name |
|-------------|-------------|
| `access_esm1_5` | ACCESS-ESM1-5 |
| `ec_earth3_veg_lr` | EC-Earth3-Veg |
| `ipsl_cm6a_lr` | IPSL-CM6A-LR |
| `miroc6` | MIROC6 |
| `mpi_esm1_2_hr` | MPI-ESM1-2-HR |

Scenarios: `historical`, `ssp245`, `ssp585`.

---

## Quick run order

```text
# Per GCM (example: MPI-ESM1-2-HR)
01_preprocess_mslp_mpi.ipynb          # historical, ssp245, ssp585
02_regrid_gcms_sst.ipynb
03_preprocess_sst.ipynb
04_pcs_mslp.ipynb                     # ★ copy pcs_95 CSVs to 03B
05_pcs_sst.ipynb

# TC branch (optional)
06_mlr_tcs.ipynb
07_assing_tcs.ipynb
08_projected_changes.ipynb
```

Then proceed to [`03B/05_ALR_Clusters_GCMs_*.ipynb`](../03B_Century_GCMs_Waves/README.md).
