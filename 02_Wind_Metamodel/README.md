# 02_Wind_Metamodel — KMA wind clusters + BinWaves merge

K-means archetype (KMA) wind classification and SWAN re-runs on **250 behavioural map units (BMUs)**, merged with the BinWaves swell hindcast from [`01_BinWaves`](../01_BinWaves/README.md). Produces the final **BinWaves + BMUS** bulk wave fields used for webpage statistics, buoy validation, and sediment-transport analysis.

---

## Relationship to `01_BinWaves`

```text
01_BinWaves                          02_Wind_Metamodel
─────────────────                    ─────────────────
outputs/cropped_variables/    ──►    build_kma_merged_grids (per grid)
  hs_gridN_masked.nc                   + KMA cluster SWAN (CASES_ONLY_WIND)
  phs0/ptp0/dp0_gridN_masked.nc              │
  tp/dp/dm/tm02_gridN_masked.nc              ▼
                                  gridN/outputs/BinWaves_BMUS/
                                           │
outputs/merged_grids/         ──►    partition overlays & reference lines
  (original BinWaves only)               (validation notebooks)
                                           │
                                           ▼
                                  01A / 01B post-processing
                                  → merged NorthCarolina / 500 m products
```

| From `01_BinWaves` | Required? | Role in this pipeline |
|--------------------|-----------|------------------------|
| `outputs/cropped_variables/{var}_gridN_masked.nc` | **Yes** | Template geometry + BinWaves swell/partitions for KMA merge |
| `outputs/merged_grids/{var}_merged_all.nc` | Optional | Original BinWaves reference in validation plots |
| `inputs/gebco_bathymetry.nc` | Optional | Bathymetry for sediment-transport (`Sediment_transport_check_all_NC.ipynb`) |
| `inputs/water_level_statistics.geojson` | **Yes** (500 m workflow) | Reference point list for `01B` cropping |

---

## Directory structure

```text
02_Wind_Metamodel/
├── inputs/                         # WHACS winds, SWAN INPGRID defs, seapoint GPKG
├── grid1/ … grid4/
│   ├── 00_kma_wind_pipeline.ipynb  # PCA → MDA → KMA → CASES_ONLY_WIND
│   ├── inputs/CASES_ONLY_WIND/     # 250 SWAN runs (000–249), each with output.mat
│   ├── outputs/KMA/                # Cluster centroids, BMU time series
│   ├── outputs/BinWaves_BMUS/      # Per-grid merged bulk NetCDFs
│   └── utils/                      # whacs_wind, kma_cluster_swan, build_kma_merged_grids.*
├── utils/
│   └── postprocessing_binwaves_bmus.py   # Crop, merge, webpage GeoJSON
├── outputs/
│   ├── NorthCarolina/              # All-grids smooth merge (full domain)
│   ├── merged_500m_binwaves_bmus/  # 500 m point merge
│   └── cropped_500m_binwaves_bmus/ # Per-grid 500 m crops
├── 01A_postProcessing_binwaves_bmus_all_grids.ipynb
├── 01B_postProcessing_all_grids_500m_only.ipynb
├── Sediment_transport_check_all_NC.ipynb
└── webpage_binwaves_bmus/          # GeoJSON statistics for web map
```

---

## Prerequisites

- Completed [`01_BinWaves`](../01_BinWaves/README.md) through `outputs/cropped_variables/`
- Python: `xarray`, `pandas`, `scikit-learn`, `netCDF4`, `geopandas`, `scipy`
- SWAN cluster outputs (`output.mat`) from the KMA wind pipeline (generated in this folder)

---

## Inputs (local)

### Wind / KMA (`inputs/`)

| File | Purpose |
|------|---------|
| `north_carolina_winds_2_uwnd_WHACS.nc`, `north_carolina_winds_2_vwnd_WHACS.nc` | Full-domain WHACS 10 m winds |
| `uwnd_gridN.nc`, `vwnd_gridN.nc` | Winds cropped to SWAN CGRID (grids 1 & 3; run `crop_whacs_winds_by_grid.py`) |
| `INPUT_g1` … `INPUT_g4` | SWAN CGRID definitions per grid |
| `csiro_whacs_points.gpkg` → `csiro_whacs_points_cropped_corrected.gpkg` | WHACS seapoint locations for PCA/KMA |
| `water_level_statistics.geojson` | 500 m reference points (from `01_BinWaves/inputs/`) |

Generate cropped winds:

```bash
cd 02_Wind_Metamodel/inputs
python crop_whacs_winds_by_grid.py
```

### From `01_BinWaves` (per grid)

Expected under `../01_BinWaves/outputs/cropped_variables/`:

```text
hs_gridN_masked.nc
tp_gridN_masked.nc
dp_gridN_masked.nc
dm_gridN_masked.nc
tm02_gridN_masked.nc
phs0_gridN_masked.nc
ptp0_gridN_masked.nc
dp0_gridN_masked.nc
```

---

## Pipeline — per grid (KMA + BinWaves merge)

### 1. KMA wind pipeline

**Notebook:** `gridN/00_kma_wind_pipeline.ipynb`

| Step | Description | Output |
|------|-------------|--------|
| Load WHACS seapoints | Grid-specific u/v wind time series (1980+) | — |
| PCA → MDA → KMA | Reduce dimensionality; **250 clusters (BMUs)** | `outputs/KMA/` |
| Export cluster winds | Wind fields per BMU | `inputs/CASES_ONLY_WIND/{000..249}/` |
| SWAN CASES_ONLY_WIND | Static SWAN run per cluster on INPGRID | `inputs/CASES_ONLY_WIND/{bmu}/output.mat` |
| Hourly BMU assignment | Nearest centroid index per hour | `outputs/KMA/nearest_centroids_idxs_kma_pcs.csv` |

### 2. Build BinWaves + BMUS bulk fields

**Script:** `gridN/utils/build_kma_merged_grids.sh`

```bash
cd 02_Wind_Metamodel/grid1
bash utils/build_kma_merged_grids.sh
# Optional: pass --no-dm or --no-tm02 to skip variables
```

**Logic (per hour, per site):**

1. Read active BMU from `nearest_centroids_idxs_kma_pcs.csv`
2. Take wind-sea partition (`phs0`, `ptp0`, `dp0`) from that BMU's SWAN `output.mat`
3. Combine with BinWaves swell partitions from `01_BinWaves/outputs/cropped_variables/`
4. Recompute total bulk: `hs`, `tp`, `dp`, `dm`, `tm02`

**Outputs:** `gridN/outputs/BinWaves_BMUS/{var}_gridN_BinWaves_BMUS.nc`

| Variable | Description |
|----------|-------------|
| `phs0`, `ptp0`, `dp0` | Wind-sea partition (from KMA SWAN) |
| `hs`, `tp`, `dp`, `dm`, `tm02` | Total bulk (wind-sea + swell combined) |

Repeat for `grid2`, `grid3`, `grid4`.

---

## Pipeline — all grids (post-processing)

### 01A — Full domain merge

**Notebook:** `01A_postProcessing_binwaves_bmus_all_grids.ipynb`  
**Module:** `utils/postprocessing_binwaves_bmus.py`

| Step | Output |
|------|--------|
| Audit per-grid `BinWaves_BMUS` files | Summary table |
| Smooth buffer merge across grids | `outputs/NorthCarolina/{var}_NorthCarolina.nc` |
| Webpage statistics GeoJSON | `webpage_binwaves_bmus/wave_statistics_{all,hs,dp}.geojson` |
| Buoy validation | Scatter / time-series vs NDBC |

Merge parameters (defaults): `SMOOTH_STEEPNESS=2.0`, `BLEND_BUFFER_KM=30.0`.

### 01B — 500 m reference points only

**Notebook:** `01B_postProcessing_all_grids_500m_only.ipynb`

| Step | Output |
|------|--------|
| Crop to `inputs/water_level_statistics.geojson` | `outputs/cropped_500m_binwaves_bmus/{var}_gridN_points500m.nc` |
| Smooth merge (8 variables) | `outputs/merged_500m_binwaves_bmus/{var}_500m.nc` |
| Webpage GeoJSON | `webpage_binwaves_bmus_500m/wave_statistics_*.geojson` |

Variables: `hs`, `tp`, `dm`, `dp`, `tm02`, `phs0`, `ptp0`, `dp0`.

### Sediment transport validation

**Notebook:** `Sediment_transport_check_all_NC.ipynb` (and `grid4/Sediment_transport_check_all_NC.ipynb`)

- Hindcast bulk from **BinWaves + KMA** merged products
- Partition overlays still read original BinWaves `01_BinWaves/outputs/merged_grids/`
- Requires `gebco_bathymetry.nc` from `01_BinWaves/inputs/`

---

## Outputs (final products)

| Path | Description |
|------|-------------|
| `gridN/outputs/BinWaves_BMUS/*_gridN_BinWaves_BMUS.nc` | Per-grid BinWaves + KMA bulk hindcast |
| `outputs/NorthCarolina/{var}_NorthCarolina.nc` | All-grids smooth merge (full seapoint set) |
| `outputs/merged_500m_binwaves_bmus/{var}_500m.nc` | All-grids merge at 500 m reference points |
| `webpage_binwaves_bmus/` | Full-domain wave statistics GeoJSON |
| `webpage_binwaves_bmus_500m/` | 500 m wave statistics GeoJSON |
| `outputs/Figures/` | Validation and QA plots |

---

## Quick reference — run order

```text
# Prerequisites: 01_BinWaves outputs/cropped_variables/ complete

# Once (winds)
inputs/crop_whacs_winds_by_grid.py

# Per grid N = 1..4
gridN/00_kma_wind_pipeline.ipynb
bash gridN/utils/build_kma_merged_grids.sh

# All grids
01A_postProcessing_binwaves_bmus_all_grids.ipynb    # full domain
01B_postProcessing_all_grids_500m_only.ipynb        # 500 m points

# Optional validation
Sediment_transport_check_all_NC.ipynb
gridN/Sediment_transport_check_all_NC.ipynb
```

---

## Grids summary

| Grid | Domain | ~Sites | KMA notebook | Build script |
|------|--------|--------|--------------|--------------|
| grid1 | SC / GA coast | 845 | `grid1/00_kma_wind_pipeline.ipynb` | `grid1/utils/build_kma_merged_grids.sh` |
| grid2 | Mid-Atlantic | 1105 | `grid2/00_kma_wind_pipeline.ipynb` | `grid2/utils/build_kma_merged_grids.sh` |
| grid3 | VA / MD | 1386 | `grid3/00_kma_wind_pipeline.ipynb` | `grid3/utils/build_kma_merged_grids.sh` |
| grid4 | North Carolina | 848 | `grid4/00_kma_wind_pipeline.ipynb` | `grid4/utils/build_kma_merged_grids.sh` |

---

## Troubleshooting

| Issue | Check |
|-------|-------|
| `build_kma_merged_grids` fails on missing input | Verify `01_BinWaves/outputs/cropped_variables/hs_gridN_masked.nc` exists |
| Missing BMU SWAN output | Re-run `00_kma_wind_pipeline.ipynb` CASES_ONLY_WIND section for that grid |
| Merge skips a grid | Run audit cell in `01A`; only grids with existing `BinWaves_BMUS` files are merged |
| Jan 1997 gaps in BinWaves | Re-run targeted reconstruction in `01_BinWaves` (`*_Jan97.ipynb` + `reconstruct_spectra.py`) before rebuilding BMUS |
