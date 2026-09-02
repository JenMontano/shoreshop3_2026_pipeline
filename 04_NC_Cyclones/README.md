# 04_NC_Cyclones — Tropical-cyclone wave metamodel & downscaling

High-resolution **tropical cyclone (TC) wave fields** for the North Carolina domain. Two stages:

| Subfolder | Purpose |
|-----------|---------|
| [`04A_Cyclones_Metamodel`](04A_Cyclones_Metamodel/README.md) | SWAN parametric cases → PCA/RBF **metamodel** → predicted cyclone wave fields from emulated tracks |
| [`04B_Cyclones_emulator`](04B_Cyclones_emulator/README.md) | Spectrum reconstruction → PTM4 partitions → **splice cyclone peaks** into century-scale baselines |

Upstream: [`03A`](../03_NC_Wave_Emulator/03A_Stochastic_GCMs_NC/README.md) (emulated TC tracks), [`01_BinWaves`](../01_BinWaves/README.md) (kp / offshore spectra), [`02_Wind_Metamodel`](../02_Wind_Metamodel/README.md) and [`03B`](../03_NC_Wave_Emulator/03B_Century_GCMs_Waves/README.md) (500 m climate baselines).

---

## End-to-end flow

```text
03A — TC track-type MLR
  outputs/tc_logit/{model}/{ssp}/emu_tracks_*_corrected.nc
        │
        ▼
04A — extract track params → apply PCA/RBF metamodel (trained on SWAN cases)
  outputs/predicted_emu_tracks_{model}_{ssp}_corrected[_mf].nc
        │
        ▼  (04B reads directly via paths.py — no copy step)
04B/00 — JONSWAP spectra from metamodel fields × BinWaves kp
04B/01 — PTM4 partitions per grid × cyclone → merge per cyclone
04B splice — replace baseline peaks on cyclone days
        │
        ▼
  outputs/earth3_ssp{245,585}_with_cyclones/
  outputs/hs_500m_with_cyclones*.nc
```

**03B** provides the **non-cyclone** century-scale 500 m fields (`earth3_veg_ssp245/585`) that 04B splices into. **02** provides the **historical** 500 m reference used for validation and optional baseline merges.

---

## Cross-pipeline connections

### `03A` → `04A`

| Source (03A) | Used in 04A | Purpose |
|--------------|-------------|---------|
| **`outputs/tc_logit/ec_earth3_veg_lr/{historical,ssp245,ssp585}/emu_tracks_*_corrected.nc`** | `13_reconstrucion_trazas.ipynb` | Emulated TC tracks from stochastic GCM branch |
| Same (raw tracks) | `04B/inputs/emu_tracks_*_corrected.nc` | Cyclone catalog metadata (dates, positions) |

Run 03A notebooks **`06_mlr_tcs.ipynb`** → **`07_assing_tcs.ipynb`** before 04A step 13.

### `04A` → `04B`

| Source (04A) | Read by 04B | Purpose |
|--------------|--------------|---------|
| **`outputs/predicted_emu_tracks_*_corrected.nc`** | `paths.predicted_emu_tracks()` | Metamodel wave/wind fields per emulated cyclone |
| **`outputs/predicted_emu_tracks_*_corrected_mf.nc`** | `paths.predicted_emu_tracks(..., mf=True)` | Same + track-parameter columns (splice scripts) |
| `outputs/ds_predicted_test_all.nc` | `04B/inputs/` | Synthetic test dataset (optional / legacy workflows) |
| `outputs/emu_tracks_*_param.csv` | embedded in `*_mf.nc` via 04A/13 | Track parameters at furthest point |

After updating 04A outputs, refresh `04B/inputs/` (copy or symlink the `predicted_emu_tracks_*` files).

### `01_BinWaves` → `04B`

| Source (01) | Used in 04B | Purpose |
|-------------|-------------|---------|
| **`gridN/outputs/kp_coeffs_filtered_gridN.nc`** | `00_Cylcones_NC.ipynb`, `utils/reconstruct_spectra_cyclones_checkpointed.py` | Offshore-to-nearshore transfer for cyclone spectra |
| `gridN/inputs/*_spec*15D.nc` | spectrum reference geometry | Template for reconstructed cyclone spectra |
| `inputs/gebco_bathymetry.nc` | partitioning, depth | Bathymetry (also copied under `04B/inputs/`) |

### `02_Wind_Metamodel` → `04B`

| Source (02) | Used in 04B | Purpose |
|-------------|-------------|---------|
| **`outputs/merged_500m_binwaves_bmus/{var}_500m.nc`** | `postProcessing_*`, `compare_*`, splice validation | Historical 500 m BinWaves+BMUS baseline |
| `outputs/cropped_500m_binwaves_bmus/` | `postProcessing_all_grids_500m_only.ipynb` | Per-grid cropped inputs for partition merges |
| `inputs/water_level_statistics.geojson` | 500 m point list | Reference site coordinates |

### `03B` → `04B`

| Source (03B) | Used in 04B | Purpose |
|--------------|-------------|---------|
| **`outputs/earth3_veg_ssp245/*_500m.nc`** | `paths.earth3_baseline_dir("ssp245")` | EC-Earth3 SSP2-4.5 **without** cyclones (splice target) |
| **`outputs/earth3_veg_ssp585/*_500m.nc`** | `paths.earth3_baseline_dir("ssp585")` | EC-Earth3 SSP5-8.5 baseline |
| `outputs/partitions_SuperPoint/` | `utils/compare_superpoint_vs_merged500m.py` | Optional SuperPoint validation |

---

## Quick run order

```text
# Prerequisites
03A/06_mlr_tcs.ipynb + 07_assing_tcs.ipynb   →  tc_logit/emu_tracks
01_BinWaves (kp per grid)                     →  gridN/outputs/kp_coeffs_*
02/01B                                        →  merged_500m_binwaves_bmus
03B/05_ALR_Clusters_GCMs_earth3_veg_*       →  earth3_veg_ssp245/585 baselines

# 04A — metamodel (one-time / when SWAN cases change)
04A/00 … 09  (grid → SWAN cases → PCA/RBF train)
04A/13_reconstrucion_trazas.ipynb            →  predicted_emu_tracks_*

# Refresh 04B inputs — not needed; 04B reads 04A + 03B outputs via paths.py

# 04B — cyclone downscaling
04B/00_Cylcones_NC.ipynb                     →  gridN/inputs/spectra_point_*
04B/utils/reconstruct_spectra_cyclones_checkpointed.py  →  gridN/outputs/reconstructed_spectra_*
04B/01_Partitions_cyclones_compact_PTM4.ipynb
04B/utils/splice_cyclones_all_sites.py       →  earth3_ssp*_with_cyclones/
```

---

## Sub-documentation

- [`04A_Cyclones_Metamodel/README.md`](04A_Cyclones_Metamodel/README.md) — SWAN metamodel training & GCM track reconstruction
- [`04B_Cyclones_emulator/README.md`](04B_Cyclones_emulator/README.md) — spectra, partitions, splicing into climate series
