# ShoreShop3_2026 — Hydrodynamic forcing for North Carolina

Quantitative predictions of coastal erosion and accretion rely critically on the quality of atmospheric and oceanic forcing. This repository implements the **wave-climate component** of the [ShoreShop3 benchmark](https://shoreshop3.netlify.app/) for the North Carolina coast: computationally efficient, high-resolution hydrodynamic forcing for shoreline evolution models at scales from **hourly events** to **multi-decadal variability**, and from the **nearshore** (up to **500 m** along reference transects) to the **regional coastline**.

> **Jupyter Book:** the same overview is rendered at `docs/intro.html` — run `bash docs/build.sh` and `bash docs/serve.sh` ([build instructions](docs/README.md)).

---

## Scientific scope

### Present climate (1980–2023)

**BinWaves** — a hybrid additive downscaling model (Cagigal et al., 2023) — propagates WHACS offshore spectra through four overlapping SWAN grids and reconstructs nearshore directional wave spectra. A **wind-wave metamodel** (PCA => MDA => KMA (250) wind clusters + SWAN re-runs on 250 behavioural map units) is merged with BinWaves swell to represent **locally generated wind seas**, producing an hourly **BinWaves + BMUS** hindcast.

### Past century & future climate (1870–2100)

A **climate-based wave emulator** jointly simulates **regular sea states** and **tropical cyclones (TCs)**:

- **Regular climate:** MSLP (and SST) PCA identifies dominant atmospheric modes (EOFs/PCs; Camus et al., 2014). Daily PCs are linked to **wave clusters** (KMA) via an **Autoregressive Logistic Regression (ALR)** model, with seasonal structure and large-scale modes (NAO, MJO). Synthetic atmospheric sequences drive offshore → nearshore translation through BinWaves + the wind metamodel.
- **Tropical cyclones:** Historical and emulated TC tracks expand plausible storm scenarios. A **TC track-type MLR** (03A) feeds a **SWAN parametric metamodel** (04A) that predicts cyclone wave fields; **04B** downscales spectra and **splices** TC peaks into the non-TC GCM (`earth3`) baselines from 03B.

Together, the framework delivers:

| Product | Period | Resolution | Folder |
|---------|--------|------------|--------|
| Hourly wave hindcast | 1980–2023 | Full domain + 500 m sites | `02_Wind_Metamodel/outputs/NorthCarolina/`, `.../merged_500m_binwaves_bmus/` |
| Century regular + TC climate | 1870–2015 (emulated) | 500 m reference sites | `03B`, `04B` |
| Future wave projections | 2015–2100 | 500 m; multiple GCMs, SSP2-4.5 / SSP5-8.5 | `03B`, `04B` |

---

## Computational pipeline (this repo)

Four linked stages. Each has a detailed README — see the [documentation map](#documentation-map) below.

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
| [**01 — BinWaves**](01_BinWaves/README.md) | Nearshore swell hindcast | BinWaves / SWAN, kp reconstruction | `01_BinWaves/outputs/cropped_variables/`, `.../merged_grids/`, `gridN/outputs/kp_coeffs_*` |
| [**02 — Wind Metamodel**](02_Wind_Metamodel/README.md) | Wind-sea correction | KMA (250 BMUs) + smooth grid merge | `02_Wind_Metamodel/outputs/NorthCarolina/`, `.../merged_500m_binwaves_bmus/` |
| [**03 — Wave Emulator**](03_NC_Wave_Emulator/README.md) | Past & future wave climate | PCA + wave-cluster ALR; TC MLR (03A) | `03B/outputs/earth3_veg_ssp245\|585/`, `03A/outputs/tc_logit/` |
| [**04 — Cyclones**](04_NC_Cyclones/README.md) | TC wave fields & splice | PCA/RBF metamodel (04A); BinWaves downscale + partitions (04B) | `04A/outputs/predicted_emu_tracks_*`, `04B/outputs/earth3_ssp*_with_cyclones/` |

**03** and **04** sub-branches:

| Subfolder | Branch | Feeds |
|-----------|--------|-------|
| [03A — Stochastic GCMs](03_NC_Wave_Emulator/03A_Stochastic_GCMs_NC/README.md) | CMIP6 MSLP/SST PCA + TC track-type MLR | → **04A** |
| [03B — Century GCM waves](03_NC_Wave_Emulator/03B_Century_GCMs_Waves/README.md) | Wave-cluster KMA + ALR + GCM runs | → **04B** baselines (no TCs) |
| [04A — Cyclones metamodel](04_NC_Cyclones/04A_Cyclones_Metamodel/README.md) | SWAN parametric cases → metamodel | → **04B** catalog |
| [04B — Cyclones emulator](04_NC_Cyclones/04B_Cyclones_emulator/README.md) | Spectra → partitions → splice into 03B | Final 500 m series **with** TCs |

---

## Key process notebooks

Each stage README covers merge/post-processing across all four grids. The **core physics / ML steps** for one grid (or the emulator chain) are illustrated by these notebooks (also in the [Jupyter Book](docs/README.md) sidebar):

| Stage | Notebook | What it shows |
|-------|----------|---------------|
| **01** | `01_BinWaves/grid1/01_Propagation.ipynb` | BinWaves SWAN propagation → kp coefficients |
| **02** | `02_Wind_Metamodel/grid1/00_kma_wind_pipeline.ipynb` | KMA wind clusters → SWAN wind-sea cases |
| **03B** | `03B/00_Partitions_cluster_points.ipynb` | PTM4 partitions at four super points |
| **03B** | `03B/01_Testing_ClusteringAlgorithms_partitions_all.ipynb` | Wave-cluster (BMU) selection |
| **03B** | `03B/02_PCs_MSLP_SST.ipynb` | MSLP/SST PCA for training |
| **03B** | `03B/03_Clusters_and_PCS.ipynb` | Calibrate clusters ↔ atmospheric PCs |
| **03B** | `03B/04_ALR_Clusters_20CR.ipynb` | ALR model training (20CR) |
| **03A** | `03A/06_mlr_tcs.ipynb`, `03A/07_assing_tcs.ipynb` | TC track-type MLR → GCM catalogue |
| **04A** | `04A/13_reconstrucion_trazas.ipynb` | Metamodel on emulated tracks |
| **04B** | `04B/00_Cylcones_NC.ipynb`, `04B/01_Partitions_cyclones_compact_PTM4.ipynb` | Cyclone spectra → PTM4 → splice |

Grid merge / 500 m post-processing: `01_BinWaves/04_PostProcessing…`, `02_Wind_Metamodel/01A`, `02_Wind_Metamodel/01B`.

---

## Data handoffs

| Producer | Output | Consumer | Purpose |
|----------|--------|----------|---------|
| **01** | `01_BinWaves/outputs/cropped_variables/` | **02** | Geometry template for KMA merge |
| **01** | `gridN/outputs/kp_coeffs_*` | **04B** | Offshore → nearshore TC spectra |
| **02** | `02_Wind_Metamodel/outputs/NorthCarolina/{var}_*.nc` | **03B** | ALR validation & BMU bootstrap |
| **02** | `02_Wind_Metamodel/outputs/merged_500m_binwaves_bmus/` | **03B**, **04B** | 500 m hindcast reference |
| **03A** | `03A/outputs/tc_logit/.../emu_tracks_*` | **04A** | Emulated TC catalogue |
| **03A** | `03A/outputs/cmip6_models/mslp_*_pcs_95.csv` | **03B** | GCM MSLP PCs for ALR |
| **03B** | `03B/outputs/earth3_veg_ssp245\|585/` | **04B** | Baseline climate **without** TCs |
| **04A** | `04A/outputs/predicted_emu_tracks_*` | **04B** | Metamodel cyclone Hs, Tp, wind |
| **04B** | `04B/outputs/earth3_ssp*_with_cyclones/` | shoreline models | Final forcing **with** TC peaks |

Paths are **repo-relative**; **04B** reads upstream products via [`04B/paths.py`](04_NC_Cyclones/04B_Cyclones_emulator/paths.py) (no copies in `04B/inputs/` for tracks or baselines).

---

## Recommended run order

```text
1. 01_BinWaves          propagation → reconstruction → partitions (per grid) → crop → merge all grids
2. 02_Wind_Metamodel    01A full domain + 01B 500 m products
3. 03B                  partitions → PCA → MDA → KMA → ALR GCM notebooks
4. 03A                  CMIP6 preprocess + TC MLR (parallel with 3B after step 2)
5. 04A                  SWAN metamodel train → 13_reconstrucion_trazas
6. 04B                  00 spectra → reconstruct → 01 partitions → splice
```

**Prerequisites:** 03B ALR needs **02/01A**; 04A step 13 needs **03A/06 + 07**; 04B splice needs **04A/outputs**, **03B/earth3_veg_ssp***, **01/kp_coeffs**.

---

## Shared conventions

- **Grids:** `grid1`–`grid4` — four overlapping SWAN domains (see [01 — BinWaves](01_BinWaves/README.md)). Single-grid setups skip crop/merge; use kp coefficients directly after propagation.
- **Spatial merge:** smooth buffer blend across grid overlaps ([02](02_Wind_Metamodel/README.md); same math in [04B](04_NC_Cyclones/04B_Cyclones_emulator/README.md) cyclone merge).
- **500 m sites:** `02_Wind_Metamodel/inputs/water_level_statistics.geojson`; cyclone partitioning uses `04B/inputs/isobath_10m_points_500m.geojson`.

---

## Documentation map

| Folder | Contents |
|--------|----------|
| [01 — BinWaves](01_BinWaves/README.md) | WHACS propagation, buoy correction, reconstruction, partitions |
| [02 — Wind Metamodel](02_Wind_Metamodel/README.md) | KMA pipeline, BinWaves+BMUS merge, full domain and 500 m products |
| [03 — Wave Emulator](03_NC_Wave_Emulator/README.md) | 03A ↔ 03B overview |
| [03A — Stochastic GCMs](03_NC_Wave_Emulator/03A_Stochastic_GCMs_NC/README.md) | CMIP6 preprocessing, TC track MLR |
| [03B — Century GCM waves](03_NC_Wave_Emulator/03B_Century_GCMs_Waves/README.md) | Wave clusters, ALR, GCM projections |
| [04 — Cyclones](04_NC_Cyclones/README.md) | 04A ↔ 04B overview |
| [04A — Metamodel](04_NC_Cyclones/04A_Cyclones_Metamodel/README.md) | SWAN cases, PCA/RBF training |
| [04B — Emulator & splice](04_NC_Cyclones/04B_Cyclones_emulator/README.md) | Spectra, PTM4 partitions, climate splice |

---

## References

- Montaño et al. (2020) — ShoreShop benchmark framework
- Mao et al. (2025) — ShoreShop3
- Cagigal et al. (2023) — BinWaves hybrid downscaling
- Camus et al. (2014) — Wave climate emulation (PCA + classification)

---

## Repository notes

- **Git:** track code and notebooks only; `outputs/`, `*.nc`, and large inputs are in `.gitignore`.
- **Legacy paths:** outputs may still show old absolute paths; active workflows use in-repo relative paths and `04B/paths.py`.
