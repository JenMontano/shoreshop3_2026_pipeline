<!-- Book intro — links target Jupyter Book chapters. GitHub copy: ../README.md -->

# ShoreShop3_2026 — Hydrodynamic forcing for North Carolina

Quantitative predictions of coastal erosion and accretion rely critically on the quality of atmospheric and oceanic forcing. This repository implements the **wave-climate component** of the [ShoreShop3 benchmark](https://shoreshop3.netlify.app/) for the North Carolina coast: computationally efficient, high-resolution hydrodynamic forcing for shoreline evolution models at scales from **hourly events** to **multi-decadal variability**, and from the **nearshore** (up to **500 m** along reference transects) to the **regional coastline** .

---

## Scientific scope

### Present climate (1980–2023)

**BinWaves** — a hybrid additive downscaling model (Cagigal et al., 2023) — propagates WHACS offshore spectra through four overlapping SWAN grids and reconstructs nearshore directional wave spectra. A **wind-wave metamodel** (PCA => MDA => KMA (250) wind clusters + SWAN re-runs on 250 behavioural map units) is merged with BinWaves swell to represent **locally generated wind seas**, producing an hourly **BinWaves + BMUS** hindcast.

### Past century & future climate (1870–2100)

A **climate-based wave emulator** jointly simulates **regular sea states** and **tropical cyclones (TCs)**:

- **Regular climate:** MSLP (and SST) PCA identifies dominant atmospheric modes (EOFs/PCs; Camus et al., 2014). Daily PCs are linked to **wave clusters** (KMA) via an **Autoregressive Logistic Regression (ALR)** model, with seasonal structure and large-scale modes (NAO, MJO). Synthetic atmospheric sequences drive offshore → nearshore translation through BinWaves + the wind metamodel.
- **Tropical cyclones:** Historical and emulated TC tracks expand plausible storm scenarios. A **TC track-type MLR** (03A) feeds a **SWAN parametric metamodel** (04A) that predicts cyclone wave fields; **04B** downscales spectra, and **merge** TC  into the non-TC GCM (earth3) baselines from 03B.

Together, the framework delivers:

| Product | Period | Resolution | Folder |
|---------|--------|------------|--------|
| Hourly wave hindcast | 1980–2023 | Full domain + 500 m sites | `02_Wind_Metamodel/outputs/NorthCarolina & merged_500m_binwaves_bmus` |
| Century regular + TC climate | 1870–2015 (emulated) | 500 m reference sites | `03B`, `04B` |
| Future wave projections | 2015–2100 | 500 m; multiple GCMs, SSP2-4.5 / SSP5-8.5 | `03B`, `04B` |

---

## Computational pipeline (this repo)

Four linked stages. Each has a detailed README — use the sidebar or the documentation map (sidebar) below.

```text
  PRESENT CLIMATE                    EMULATOR & PROJECTIONS
  ─────────────────                  ─────────────────────────

  01 BinWaves          WHACS → SWAN propagation → spectral
       │               reconstruction → partitions (4 grids)
       ▼
  02 Wind Metamodel    KMA wind clusters → merge wind-sea
       │               with BinWaves → NorthCarolina + 500 m
       │
       ├──────────────────────────────┐
       ▼                              ▼
  03B Century waves        03A TC track MLR (CMIP6 MSLP/SST)
  Wave-cluster KMA         emu_tracks → 04A SWAN metamodel
  ALR => Century & GCM 
  projections             → 04B spectra reconstruction TCs + GCM Non-TC emulator
       │                              │
       └──────────┬───────────────────┘
                  ▼
         earth3_ssp245|585_with_cyclones/
         (500 m forcing with TC peaks embedded)
```

### Stage summary

| Stage | Scientific role | Key method | Main outputs |
|-------|-----------------|------------|--------------|
| [**01 — BinWaves**](chapters/01_binwaves) | Nearshore swell hindcast | BinWaves / SWAN, kp reconstruction | `outputs/cropped_variables/`, `outputs/merged_grids/`, `gridN/outputs/kp_coeffs_*` |
| [**02 — Wind Metamodel**](chapters/02_wind_metamodel) | Wind-sea correction | KMA (250 BMUs) + smooth grid merge | `outputs/NorthCarolina/`, `outputs/merged_500m_binwaves_bmus/` |
| [**03 — Wave Emulator**](chapters/03_wave_emulator) | Past & future wave climate | PCA + wave-cluster ALR; TC MLR (03A) | `03B/outputs/GCMs_ssp*/`, `03A/outputs/tc_logit/` |
| [**04 — Cyclones**](chapters/04_cyclones) | TC wave fields & splice | PCA/RBF metamodel (04A); BinWaves downscale + Partitions (04B) | `04A/outputs/predicted_emu_tracks_*`, `04B/outputs/earth3_ssp*_with_cyclones/` |

**03** and **04** sub-branches:

| Subfolder | Branch | Feeds |
|-----------|--------|-------|
| [03A — Stochastic GCMs](chapters/03a_stochastic_gcm) | CMIP6 MSLP/SST PCA + TC track-type MLR | → **04A** |
| [03B — Century GCM waves](chapters/03b_century_gcm_waves) | Wave-cluster KMA + ALR + GCM runs | → **04B** baselines (no TCs) |
| [04A — Cyclones metamodel](chapters/04a_cyclones_metamodel) | SWAN parametric cases → metamodel | → **04B** catalog |
| [04B — Cyclones emulator](chapters/04b_cyclones_emulator) | Spectra → partitions → splice into 03B | Final 500 m series **with** TCs |

---

## Key process notebooks

Each stage README covers merge/post-processing across all four grids. The **core physics / ML steps** for one grid (or the emulator chain) are illustrated by these notebooks (included in the Jupyter Book sidebar):

| Stage | Notebook | What it shows |
|-------|----------|---------------|
| **01** | `grid1/01_Propagation.ipynb` | BinWaves SWAN propagation → kp coefficients |
| **02** | `grid1/00_kma_wind_pipeline.ipynb` | KMA wind clusters → SWAN wind-sea cases |
| **03B** | `01_Testing_ClusteringAlgorithms_partitions_all.ipynb` | Wave-cluster (BMU) selection |
| **03B** | `02_PCs_MSLP_SST.ipynb` | MSLP/SST PCA for training |
| **03B** | `03_Clusters_and_PCS.ipynb` | Calibrate clusters ↔ atmospheric PCs |
| **03B** | `04_ALR_Clusters_20CR.ipynb` | ALR model training (20CR) |
| **03A** | `06_mlr_tcs.ipynb`, `07_assing_tcs.ipynb` | TC track-type MLR → GCM catalogue |
| **04A** | `13_reconstrucion_trazas.ipynb` | Metamodel on emulated tracks |
| **04B** | `00_Cylcones_NC.ipynb`, `01_Partitions_cyclones_compact_PTM4.ipynb` | Cyclone spectra → PTM4 → merge |

Grid merge / 500 m post-processing: `01/04_PostProcessing…`, `02/01A`, `02/01B`.

---

| Producer | Output | Consumer | Purpose |
|----------|--------|----------|---------|
| **01** | `outputs/cropped_variables/` | **02** | Geometry template for KMA merge |
| **01** | `gridN/outputs/kp_coeffs_*` | **04B** | Offshore → nearshore TC spectra |
| **02** | `outputs/NorthCarolina/{var}_*.nc` | **03B** | ALR validation & BMU bootstrap |
| **02** | `outputs/merged_500m_binwaves_bmus/` | **03B**, **04B** | 500 m hindcast reference |
| **03A** | `outputs/tc_logit/.../emu_tracks_*` | **04A** | Emulated TC catalogue |
| **03A** | `outputs/cmip6_models/mslp_*_pcs_95.csv` | **03B** | GCM MSLP PCs for ALR |
| **03B** | `outputs/earth3_veg_ssp245\|585/` | **04B** | Baseline climate **without** TCs |
| **04A** | `outputs/predicted_emu_tracks_*` | **04B** | Metamodel cyclone Hs, Tp, wind |
| **04B** | `outputs/earth3_ssp*_with_cyclones/` | shoreline models | Final forcing **with** TC peaks |

Paths are **repo-relative**; **04B** reads upstream products via `04B/paths.py`
---

## Recommended run order

```text
1. 01_BinWaves          propagation → reconstruction → partitions (per grid) → cropped each grid → merged (smooth borders) all grids
2. 02_Wind_Metamodel    01A full domain + 01B 500 m products
3. 03B                  partitions → PCA → MDA →  KMA → ALR GCM notebooks
4. 03A                  CMIP6 preprocess + TC MLR (parallel with 3B after step 2)
5. 04A                  SWAN metamodel train → 13_reconstrucion_trazas
6. 04B                  00 spectra → reconstruct → 01 partitions → splice
```

**Prerequisites:** 03B ALR needs **02/01A**; 04A step 13 needs **03A/06 + 07**; 04B splice needs **04A/outputs**, **03B/earth3_veg_ssp***, **01/kp_coeffs**.

---

## Shared conventions

- **Grids:** `grid1`–`grid4` — four overlapping SWAN domains (see [01 — BinWaves](chapters/01_binwaves)). Single-grid setups skip crop/merge; use kp coefficients directly after propagation.
- **Spatial merge:** smooth buffer blend across grid overlaps ([02](chapters/02_wind_metamodel); same math in [04B](chapters/04b_cyclones_emulator) cyclone merge).
- **500 m sites:** `02/inputs/water_level_statistics.geojson`; cyclone partitioning uses `04B/inputs/isobath_10m_points_500m.geojson`.

---

(documentation-map)=
## Documentation map

| Chapter | Contents |
|---------|----------|
| [01 — BinWaves](chapters/01_binwaves) | WHACS propagation, buoy correction, reconstruction, partitions |
| [02 — Wind Metamodel](chapters/02_wind_metamodel) | KMA pipeline, BinWaves+BMUS merge,full domain and 500 m products |
| [03 — Wave Emulator](chapters/03_wave_emulator) | 03A ↔ 03B overview |
| [03A — Stochastic GCMs](chapters/03a_stochastic_gcm) | CMIP6 preprocessing, TC track MLR |
| [03B — Century GCM waves](chapters/03b_century_gcm_waves) | Wave clusters, ALR, GCM projections |
| [04 — Cyclones](chapters/04_cyclones) | 04A ↔ 04B overview |
| [04A — Metamodel](chapters/04a_cyclones_metamodel) | SWAN cases, PCA/RBF training |
| [04B — Emulator & splice](chapters/04b_cyclones_emulator) | Spectra, PTM4 partitions, climate splice |

> **Browse as a website:** run `bash docs/build.sh`, then open `intro.html` in a browser (see [Build instructions](README)).

---

## References

- Montaño et al. (2020) — ShoreShop benchmark framework
- Mao et al. (2025) — ShoreShop3
- Cagigal et al. (2023) — BinWaves hybrid downscaling
- Camus et al. (2014) — Wave climate emulation (PCA + classification)

---

## Repository notes

- **Git:** track code and notebooks only; `outputs/`, `*.nc`, and large inputs are in `.gitignore`.
- **Legacy paths:** `old/` notebooks and stale `.ipynb` outputs may reference retired paths; active workflows use in-repo relative paths and `04B/paths.py`.
