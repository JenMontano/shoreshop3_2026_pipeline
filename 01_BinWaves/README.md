# 01_BinWaves — ShoreShop wave hindcast pipeline

BinWaves spectral propagation and reconstruction for the North Carolina ShoreShop domain, split into four overlapping SWAN grids (`grid1`–`grid4`). This folder produces the **BinWaves-only** bulk hindcast and partition fields that feed the wind metamodel in [`02_Wind_Metamodel`](../02_Wind_Metamodel/README.md).

---

## Overview

```text
External WHACS + bathymetry + buoys
        │
        ▼
  Per-grid BinWaves (grid1–4)
  Propagation → Buoy correction → Spectra reconstruction → Partitions
        │
        ▼
  All-grids post-processing (04 / utils)
  concat → mask → crop → merge to 500 m points
        │
        ▼
  outputs/cropped_variables/  ──►  02_Wind_Metamodel (primary handoff/ per grid)
  outputs/merged_grids/       ──►  02_Wind_Metamodel (reference / partitions)
```

| Grid | Domain (approx.) | Reference buoy | Notes |
|------|------------------|----------------|-------|
| grid1 | SC / GA coast | 41013 | Standard propagation |
| grid2 | Mid-Atlantic | 41007 | Uses `00_SuperPoint.ipynb` (irregular domain shape) |
| grid3 | VA / MD | 41125 | Uses `00_SuperPoint.ipynb` |
| grid4 | North Carolina | 44014 / 44088 | Multiple buoy-correction notebooks |

---

## Directory structure

```text
01_BinWaves/
├── inputs/                 # Shared static inputs (WHACS, bathymetry, GeoJSON, buoys)
├── grid1/ … grid4/         # Per-grid SWAN/BinWaves workflow
│   ├── 01_Propagation.ipynb
│   ├── WHACS_Buoy_Correction_*.ipynb
│   ├── CASES/              # SWAN case definitions + HPC sbatch templates
│   ├── inputs/             # Grid-specific corrected offshore spectra
│   └── outputs/            # kp, reconstructed spectra, per-variable time series
├── utils/                  # Python helpers (reconstruction, post-processing, plotting)
├── outputs/                # Merged / cropped products across all grids
├── 00_NDBC_buoy_data_download.ipynb
├── 03_Partitions_grid{N}_all.ipynb
├── 04_PostProcessing_all_grids (Original Slow).ipynb
└── WHACS_vs_BUOY.ipynb     # Optional WHACS check
```

---

## Prerequisites

- Python environment with `bluemath-tk[waves]`, `xarray`, `wavespectra`, `geopandas`, `dask`
- Access to WHACS hindcast data (stored under `inputs/WHACS/`)
- HPC access for SWAN propagation cases (`gridN/CASES/sbatch_example*.sh`)

Large reconstructed spectra may be stored on shared Lustre (e.g. `/lustre/geocean/DATA/GEOOCEAN/BinWavesDuke/gridN`) rather than inside this repo — see the `spectra_input_dir` path in each `03_Partitions_gridN_all.ipynb`.

---

## Inputs (required)

### Shared (`inputs/`)

| File / folder | Purpose |
|---------------|---------|
| `gebco_bathymetry.nc` | Bathymetry for propagation and downstream sediment checks |
| `inputs/WHACS/north_carolina_*_{hs,fp,dp,uwnd,vwnd,...}_WHACS.nc` | Offshore WHACS wave and wind forcing |
| `CoastSat_shoreline_NC_merged.geojson` | Land mask for spatial cropping |
| `isobath_10m_points_500m.geojson` | Nearshore reference points |
| `water_level_statistics.geojson` | 500 m reference point list (also used in `02_Wind_Metamodel`) |
| `swan_inputs_points.gpkg` | SWAN input / output site locations |
| `WHACS_NorthCarolina_points.json`, `whacs_cropped.json` | WHACS seapoint metadata |
| `buoy_data/` | NDBC buoy time series (from `00_NDBC_buoy_data_download.ipynb`) |
| `satellite_dataset_carolinas.nc` | Optional satellite validation |

### Per grid (`gridN/`)

| Step | Input |
|------|-------|
| Propagation | GEBCO bathymetry, SWAN templates in `templates/` |
| Buoy correction | WHACS spectra + NDBC buoy data for the grid reference buoy |
| Spectra reconstruction | `outputs/offshore_spectra_case_gridN.nc` (or corrected buoy spectrum), `outputs/kp_coefficients_averaged.nc` or `kp_coeffs_filtered_gridN.nc`, `CASES/swan_cases_averaged.csv` |
| Partitions | Reconstructed onshore spectra (yearly `.nc` files), WHACS u/v winds, GEBCO |

---

## Pipeline — per grid

Run steps **in order** for each `gridN`. Grids are independent except for the final merge.

### 0. Optional — SuperPoint (grid2, grid3 only)

**Notebook:** `gridN/00_SuperPoint.ipynb`

Handles irregular domain geometry by defining a super-point offshore source before standard propagation.

### 1. Propagation

**Notebook:** `gridN/01_Propagation.ipynb`

- Runs SWAN case matrix and computes **Kp** (transfer) coefficients.
- **Outputs:** `gridN/outputs/kp_coefficients_averaged.nc`, `gridN/CASES/swan_cases_averaged.csv`, `gridN/outputs/offshore_spectra_case_gridN.nc`

### 2. WHACS buoy correction

**Notebook:** `gridN/WHACS_Buoy_Correction_{buoy_id}.ipynb`

- Corrects offshore WHACS spectra using the grid reference NDBC buoy.
- **Outputs:** corrected spectrum NetCDF in `gridN/inputs/` (e.g. `grid1_41013_spec_WHACS_buoy_correted_15D.nc`)

**Jan 1997 patch:** separate `*_Jan97.ipynb` notebooks exist because WHACS download for that month was incomplete and required a targeted re-run.

### 3. Spectra reconstruction

**Script:** `utils/reconstruct_spectra.py` (fast BLAS version; replaces older Dask workflow)

```bash
cd 01_BinWaves
python utils/reconstruct_spectra.py 1980-2023 --grid 1
python utils/reconstruct_spectra.py 1997-01 --grid 4 \
  --offshore grid4/inputs/44014_spec_WHACS_buoy_correted_15D_Jan97.nc \
  --tag WHACS_44014
```

- **Inputs:** offshore case-binned or raw buoy-corrected spectrum, Kp file, SWAN cases CSV.
- **Outputs:** `gridN/outputs/reconstructed_spectra/reconstructed_onshore_spectra_{period}.nc`

> **Note:** An older workflow averaged Kp coefficients (`03_Validation_All_Buoys_averaged_kp.ipynb`) to reduce reconstruction cost. With the current fast `reconstruct_spectra.py`, that averaging step is **no longer required**.

### 4. Partitions and bulk parameters

**Notebook:** `03_Partitions_gridN_all.ipynb`

Computes from reconstructed spectra (1980–2023 by default):

- Bulk: `hs`, `tp`, `tm02`, `dp`, `dm`
- PTM1 partitions: `phs0`–`phs3`, `ptp0`–`ptp3`, `dp0`–`dp3`
- Spreading: `spr0`–`spr3`

**Outputs:** yearly NetCDF per variable under `gridN/outputs/output_variables/{var}/{var}_YYYY.nc`

Existing files are skipped on re-run.

### 5. Optional validation

| Notebook | Purpose |
|----------|---------|
| `gridN/03_Validation_All_Buoys_averaged_kp.ipynb` | Legacy Kp validation (optional) |
| `WHACS_vs_BUOY.ipynb` | WHACS vs NDBC comparison |

---

## Pipeline — all grids (post-processing)

**Notebook:** `04_PostProcessing_all_grids (Original Slow).ipynb`  
**Module:** `utils/postprocessing_all_grids.py` (preferred for large runs; uses Dask where applicable)

| Step | Function / section | Output |
|------|-------------------|--------|
| 1. QA | `run_hs_postprocessing()` | NaN/zero/flat-period report |
| 2. Float32 | `convert_output_variables_to_float32()` | `gridN/outputs/output_variables/float_32/` |
| 3. Concatenate | `concatenate_float32_grids_by_variable()` | `outputs/concatenated_grids_float32/{var}_gridN.nc` |
| 4. Land mask | `crop_concatenated_files_by_spatial_mask()` | `outputs/cropped_variables/{var}_gridN_masked.nc` |
| 5. Merge grids | `merge_variable_across_grids()` | `outputs/merged_grids/{var}_merged_all.nc` |
| 6. 500 m crop | `crop_concatenated_files_by_points_geojson()` | `outputs/cropped_500m/{var}_gridN_points500m.nc` |

Additional products: `outputs/geojson_statistics_merged/`, `outputs/Figures/`, buoy time-series extracts in `outputs/time_series/`.

---

## Outputs → `02_Wind_Metamodel`

These files are the **primary handoff** to the wind metamodel pipeline:

| Output path | Variables | Used for |
|-------------|-----------|----------|
| **`outputs/cropped_variables/{var}_gridN_masked.nc`** | `hs`, `tp`, `dp`, `dm`, `tm02`, `phs0`–`phs3`, `ptp0`–`ptp3`, `dp0`–`dp3`, … | BinWaves bulk input to KMA+SWAN merge (`build_kma_merged_grids.sh`) |
| `outputs/merged_grids/{var}_merged_all.nc` | Same variables, all grids blended | Reference hindcast, partition overlays, buoy validation |
| `outputs/cropped_500m/{var}_gridN_points500m.nc` | Subset at 500 m points | Lightweight products / cross-check with `02` 500 m workflow |
| `inputs/gebco_bathymetry.nc` | Bathymetry | Sediment-transport and Qs checks in `02` |
| `inputs/water_level_statistics.geojson` | 500 m site coordinates | Cropping in both pipelines (copied/linked in `02/inputs/`) |

**Minimum requirement before starting `02_Wind_Metamodel`:** complete steps 1–4 for all grids and produce `outputs/cropped_variables/` with at least `hs`, `phs0`, `ptp0`, `dp0`, `tp`, `dp`, `dm`, `tm02` per grid.

---

## Quick reference — run order

```text
# Once (shared)
00_NDBC_buoy_data_download.ipynb

# Per grid N = 1..4
gridN/00_SuperPoint.ipynb          # grid2, grid3 only
gridN/01_Propagation.ipynb
gridN/WHACS_Buoy_Correction_*.ipynb
python utils/reconstruct_spectra.py 1980-2023 --grid N
03_Partitions_gridN_all.ipynb

# All grids
04_PostProcessing_all_grids (Original Slow).ipynb   # or call utils/postprocessing_all_grids.py directly
```

Then proceed to [`02_Wind_Metamodel/README.md`](../02_Wind_Metamodel/README.md).
