# 04A — Cyclones metamodel (SWAN → PCA/RBF)

Trains a **fast surrogate** for SWAN parametric cyclone runs on the Carolinas grid. Given TC track parameters (from historical, synthetic, or **03A-emulated** tracks), predicts wave height, period, direction, and wind fields at the **furthest influence radius (7r)**.

Downstream consumer: [`04B`](../04B_Cyclones_emulator/README.md) reads `outputs/predicted_emu_tracks_*`.

---

## Role in the ShoreShop chain

```text
External TC track catalogues (IBTrACS, synthetic, blind-period)
        │
        ▼
04A/01–06  track parametrisation + MDA case selection
04A/05     SWAN parametric runs (cases/metamodel/)
04A/07–09  preprocess → PCA + RBF train/validate
        │
        ├──► outputs/predicted_syntetic_all_OK*.nc   (synthetic validation)
        │
03A/outputs/tc_logit/.../emu_tracks_*_corrected.nc
        │
        ▼
04A/13  extract params → reconstruct via metamodel
        │
        ▼
outputs/predicted_emu_tracks_{model}_{ssp}_corrected[_mf].nc
        │
        ▼  copy to 04B/inputs/
04B cyclone emulator
```

---

## Directory structure

```text
04A_Cyclones_Metamodel/
├── inputs/
│   ├── carolinas_Gebco.tif
│   └── buoy_data/                  # NDBC bulk parameters for validation
├── cases/
│   ├── historical_param/           # SWAN runs — historical TC params
│   ├── metamodel/                  # MDA-selected representative cases
│   └── metamodel_added/
├── outputs/                        # ★ handoff to 04B
│   ├── hist_tracks_7r*.nc/csv      # Historical track params at 7r
│   ├── syntetic_tracks_7r*.nc/csv
│   ├── pca_model*.pkl, rbf_model*_OK.pkl
│   ├── Vars_postprocessed*.nc
│   ├── predicted_syntetic_all_OK*.nc
│   ├── emu_tracks_*_param.csv      # Params extracted from 03A tracks
│   └── predicted_emu_tracks_*_corrected[_mf].nc
├── figures/
├── utils/                          # SWAN wrappers, preprocess, TC helpers
└── 00_Grid_generation.ipynb … 13_reconstrucion_trazas.ipynb
```

---

## Pipeline steps

### 0. Grid generation

**Notebook:** `00_Grid_generation.ipynb`

- Builds Carolinas SWAN computational grid from GEBCO (`inputs/carolinas_Gebco.tif`)

### 1–3. Track parametrisation

| Notebook | Input | Output |
|----------|-------|--------|
| `01_TC_track_params_hist.ipynb` | External `hist_tracks_7r_filter.nc` (or `outputs/hist_tracks_7r.nc` after first run) | `outputs/hist_tracks_7r_params.csv` |
| `03_TC_track_params_syntetics.ipynb` | Synthetic NA track catalogue | `outputs/syntetic_tracks_7r_params.csv` |
| `02_Validation_Parametrisation_Hist.ipynb` | Params + buoys | Validation figures |

> Notebooks `01`/`03` still reference legacy lustre paths for original track NetCDFs. After the first successful run, use files under `outputs/`.

### 4–6. MDA + SWAN cases

| Notebook | Purpose | Key outputs |
|----------|---------|-------------|
| `04_LHS_MDA.ipynb` | Latin Hypercube + MDA case selection | `outputs/mda_model.pkl`, `selected_cases_MDA.csv` |
| `05_Wrapper_MDA.ipynb` | Launch SWAN for MDA cases | `cases/metamodel/` |
| `06_plot_fer.ipynb` | Furthest-point metric plots | `figures/` |

### 7–9. Metamodel training

| Notebook / script | Purpose | Key outputs |
|-------------------|---------|-------------|
| `07_Preprocess.ipynb` + `07_postprocess.py` | SWAN `.mat` → xarray | `outputs/Vars_postprocessed*.nc` |
| `08_Metamodel.ipynb` | PCA + RBF fitting | `pca_model*.pkl`, `rbf_model*_OK.pkl` |
| `09_reconstruction_dataset.ipynb` | Inverse PCA on test cases | `predicted_syntetic_all_OK*.nc`, `ds_predicted_test_all.nc` |
| `09_k_fold.py`, `09_Pca_RBF_final.py` | Cross-validation / final model | `OK_k_fold_metamodel_*.nc` |

### 10–12. Validation & dynamics

| Notebook | Purpose |
|----------|---------|
| `10_plot_fer.ipynb` | Metamodel FER validation |
| `11_data_preprocess_k_fold.ipynb` | K-fold dataset prep |
| `12_dynamicos.ipynb` | Dynamic TC tracks (blind-period IBTrACS) |

### 13. GCM emulated tracks (03A → 04A bridge)

**Notebook:** `13_reconstrucion_trazas.ipynb`

1. Read emulated tracks from **03A**:

   ```text
   ../../03_NC_Wave_Emulator/03A_Stochastic_GCMs_NC/outputs/tc_logit/ec_earth3_veg_lr/{historical,ssp245,ssp585}/emu_tracks_*_corrected.nc
   ```

2. Extract furthest-point parameters → `outputs/emu_tracks_*_param.csv`
3. Apply metamodel → `outputs/predicted_emu_tracks_*_corrected.nc`
4. Merge track metadata → `outputs/predicted_emu_tracks_*_corrected_mf.nc`

**Copy (or symlink) the `predicted_emu_tracks_*` files into `04B/inputs/` before running 04B.**

---

## Inputs from other ShoreShop stages

| Source | Path | Required for |
|--------|------|--------------|
| **03A TC emulator** | `03A/outputs/tc_logit/ec_earth3_veg_lr/{ssp}/emu_tracks_*_corrected.nc` | Step 13 (GCM scenarios) |
| Historical TC tracks | External lustre catalogues (see notebook 01) | Steps 1–2 (one-time) |
| Buoy data | `inputs/buoy_data/` | Validation notebooks |

---

## Key outputs → 04B

| 04A file | 04B usage |
|----------|-----------|
| `predicted_emu_tracks_*_corrected.nc` | Wind/wave forcing for spectrum reconstruction & PTM4 |
| `predicted_emu_tracks_*_corrected_mf.nc` | Cyclone catalog for splicing (`utils/splice_cyclones_*.py`) |
| `ds_predicted_test_all.nc` | Legacy partition workflows |

---

## Quick run order

```text
04A/00_Grid_generation.ipynb
04A/01_TC_track_params_hist.ipynb
04A/03_TC_track_params_syntetics.ipynb
04A/04_LHS_MDA.ipynb
04A/05_Wrapper_MDA.ipynb          # HPC SWAN
04A/07_Preprocess.ipynb
04A/08_Metamodel.ipynb
04A/09_reconstruction_dataset.ipynb

# After 03A TC branch:
04A/13_reconstrucion_trazas.ipynb
# → copy predicted_emu_tracks_* to 04B/inputs/
```

---

## Troubleshooting

| Issue | Check |
|-------|-------|
| Step 13 cannot find emu tracks | Run 03A `06_mlr_tcs` + `07_assing_tcs`; verify `03A/outputs/tc_logit/ec_earth3_veg_lr/` |
| SWAN cases missing | `05_Wrapper_MDA.ipynb` + `cases/metamodel/` |
| Metamodel pickle not found | Complete `08_Metamodel.ipynb` |
| 04B inputs stale | Re-copy `outputs/predicted_emu_tracks_*` from 04A |
