# 04B — Cyclones emulator (spectra → partitions → splice)

Downscales **04A metamodel cyclone fields** to full 2D wave spectra, partitions them (PTM4), and **splices** cyclone peaks into the century-scale **03B** climate baselines at 500 m reference sites.

---

## Role in the ShoreShop chain

```text
04A/outputs/predicted_emu_tracks_*     01_BinWaves/gridN/outputs/kp_*
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
              04B/00 — JONSWAP spectra per grid point
              04B/utils/reconstruct_spectra_cyclones_checkpointed.py — offshore → nearshore
              04B/01 — PTM4 partition per grid × cyclone
                       │
03B/outputs/earth3_veg_ssp*  (baseline, no TCs)
02/outputs/merged_500m_*   (historical reference)
                       │
                       ▼
              splice_cyclones_all_sites.py
                       │
                       ▼
              outputs/earth3_ssp*_with_cyclones/
              outputs/hs_500m_with_cyclones*.nc
```

---

## Directory structure

```text
04B_Cyclones_emulator/
├── inputs/                         # Local geometry only (isobath points, optional geojson)
│   └── isobath_10m_points_500m.geojson
├── paths.py                        # ★ upstream reads: 04A tracks, 03B baselines, 01 GEBCO
├── grid1/ … grid4/
│   ├── inputs/                     # spectra_point_*.nc (from 00)
│   └── outputs/
│       ├── kp_coeffs_filtered_gridN.nc   ← 01_BinWaves (or computed here)
│       └── reconstructed_spectra_earth3_{245,585}/
├── outputs/
│   ├── partitions_cyclones_earth3_{245,585}/   # per grid × cyclone
│   ├── merged_cyclones_earth3_{245,585}/       # one file per cyclone
│   ├── earth3_ssp245_with_cyclones/            # ★ final spliced 500 m series
│   ├── earth3_ssp585_with_cyclones/
│   └── hs/tp/dp_500m_with_cyclones*.nc
├── utils/
│   ├── reconstruct_spectra_cyclones_checkpointed.py   # ★ cyclone spectra reconstruction
│   ├── partitions_cyclones_ptm4_compact.py
│   ├── splice_cyclones_all_sites.py            # ★ main splice script
│   ├── splice_cyclones_into_hs.py
│   └── compare_earth3_cyclones.py
├── 00_Cylcones_NC.ipynb
├── 01_Partitions_cyclones_compact_PTM4.ipynb
└── postProcessing_all_grids_500m_only.ipynb
```

---

## Upstream inputs (read directly — no copies)

| Data | Source | How 04B finds it |
|------|--------|------------------|
| `predicted_emu_tracks_*_corrected[_mf].nc` | **04A** `outputs/` | `paths.predicted_emu_tracks(ssp, mf=…)` |
| `earth3_veg_ssp245\|585/*.nc` | **03B** `outputs/` | `paths.earth3_baseline_dir(ssp)` |
| `gebco_bathymetry.nc` | **01_BinWaves** `inputs/` | `paths.GEBCO_FILE` |
| `isobath_10m_points_500m.geojson` | **04B** `inputs/` | `paths.ISOBATH_GEOJSON` |

Legacy copies under `04B/inputs/` or the 04B root are **not required** if scripts/notebooks use `paths.py`.

---

## Pipeline steps

### 0. Cyclone spectra reconstruction

**Notebook:** `00_Cylcones_NC.ipynb`

- Reads `inputs/predicted_emu_tracks_*_corrected_mf.nc` (04A)
- Loads `gridN/outputs/kp_coeffs_filtered_gridN.nc` (01_BinWaves)
- Builds JONSWAP spectra → `gridN/inputs/spectra_point_{N}.nc`

**Script:** `utils/reconstruct_spectra_cyclones_checkpointed.py`

- Batch reconstruction per cyclone ID → `gridN/outputs/reconstructed_spectra_grid{N}_cyclone_*.nc`

### 1. PTM4 partitioning (compact)

**Notebook:** `01_Partitions_cyclones_compact_PTM4.ipynb`

- **Inputs:** reconstructed spectra, WHACS wind from `predicted_emu_tracks_*`, GEBCO, isobath points
- **Outputs:** `outputs/partitions_cyclones_earth3_{245,585}/`, `outputs/merged_cyclones_earth3_{245,585}/`
- Config uses `PROJECT_ROOT = Path('.').resolve()` — run with cwd = this folder

**Module:** `utils/partitions_cyclones_ptm4_compact.py`

### 2. Post-processing & baseline merge (optional)

**Notebook:** `postProcessing_all_grids_500m_only.ipynb`

- Merges BinWaves+BMUS at 500 m using **`../../02_Wind_Metamodel`** paths
- Can merge partition vars from **`../../01_BinWaves/outputs/cropped_500m`**

### 3. Splice cyclones into climate series

**Scripts:** `utils/splice_cyclones_all_sites.py`, `utils/splice_cyclones_into_hs.py`

- **Baseline (no TCs):** `inputs/earth3_veg_ssp245/` or `ssp585/` (from **03B**)
- **Cyclone catalog:** `inputs/predicted_emu_tracks_*_corrected_mf.nc` (from **04A**)
- **Cyclone partitions:** `outputs/merged_cyclones_earth3_{245,585}/`
- **Output:** `outputs/earth3_ssp245_with_cyclones/`, `outputs/hs_500m_with_cyclones.nc`

On each emulated cyclone day, replace baseline values with cyclone partition peaks (max Hs rule; companion vars at peak hour).

### 4. Validation

| Script | Compares |
|--------|----------|
| `utils/compare_earth3_cyclones.py` | earth3 ± cyclones vs **02** historical 500 m |
| `utils/compare_superpoint_vs_merged500m.py` | **03B** SuperPoint vs **02** merged 500 m |

---

## Path conventions (updated for ShoreShop3_2026)

| Old path | Current |
|----------|---------|
| `/lustre/.../Cyclones_NC` | `.` (this folder) or `Path(__file__).resolve().parents[1]` in utils |
| `/lustre/.../Wind_Metamodel` | `../../02_Wind_Metamodel` |
| `/lustre/.../ShoreShop2026` | `../../01_BinWaves` or `../../02_Wind_Metamodel` |
| `/nfs/.../New_data_Shoreshop` | `../../02_Wind_Metamodel/outputs/merged_500m_binwaves_bmus` |

---

## Quick run order

```text
# Prerequisites — upstream outputs must exist (no copy step)
04A/13  →  predicted_emu_tracks_* in 04A/outputs/
03B/05_ALR earth3_veg  →  earth3_veg_ssp245|585 in 03B/outputs/
01_BinWaves  →  kp_coeffs in gridN/outputs/

04B/00_Cylcones_NC.ipynb
python utils/reconstruct_spectra_cyclones_checkpointed.py --grid N   # per grid
04B/01_Partitions_cyclones_compact_PTM4.ipynb
python utils/splice_cyclones_all_sites.py --scenario ssp245
python utils/splice_cyclones_all_sites.py --scenario ssp585
```

---

## Troubleshooting

| Issue | Check |
|-------|-------|
| Missing `predicted_emu_tracks_*` | Run 04A/13; 04B reads from `04A/outputs/` via `paths.py` |
| Missing baseline `hs_500m.nc` | Run 03B ALR earth3_veg; 04B reads from `03B/outputs/earth3_veg_ssp*/` |
| Missing kp coefficients | `01_BinWaves/gridN/outputs/kp_coeffs_filtered_gridN.nc` |
| Splice finds no cyclones | Verify `*_mf.nc` catalog matches partition file naming |
| Historical comparison fails | Run `02/01B` → `outputs/merged_500m_binwaves_bmus/` |
