"""Compare SuperPoint partitions vs nearest merged_500m BinWaves sites.

For each TARGET_POINTS coordinate (grid1..grid4):
  - SuperPoint: partitions_SuperPoint/{hs,tp,dp}_gridN.nc  (single-point series)
  - Merged:     merged_500m_binwaves_bmus/{hs,tp,dp}_500m.nc at nearest site

Produces:
  - distance table (deg + km)
  - one multi-panel figure per grid (time series, seasonality, density, Q-Q)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[3]
SUPERPOINT_DIR = REPO_ROOT / "03_NC_Wave_Emulator/03B_Century_GCMs_Waves/outputs/partitions_SuperPoint"
MERGED_DIR = REPO_ROOT / "02_Wind_Metamodel/outputs/merged_500m_binwaves_bmus"
OUTPUT_DIR = Path("./outputs")

TARGET_POINTS = {
    "grid1": {"label": "grid1", "lat": 33.441, "lon": -77.766},
    "grid2": {"label": "grid2", "lat": 33.8928, "lon": -76.9845},
    "grid3": {"label": "grid3", "lat": 35.10, "lon": -75.36},
    "grid4": {"label": "grid4", "lat": 36.603, "lon": -74.837},
}

VARIABLES = ("hs", "tp", "dp")
LABEL_SP = "SuperPoint"
LABEL_MG = "merged_500m BinWaves"


def haversine_km(lon1, lat1, lon2, lat2) -> float:
    r = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def nearest_merged_site(target_lon: float, target_lat: float) -> dict:
    with xr.open_dataset(MERGED_DIR / "hs_500m.nc") as ds:
        lon = np.asarray(ds.lon.values, dtype=np.float64)
        lat = np.asarray(ds.lat.values, dtype=np.float64)
    d2 = (lon - target_lon) ** 2 + (lat - target_lat) ** 2
    i = int(np.argmin(d2))
    return {
        "site_index": i,
        "lon": float(lon[i]),
        "lat": float(lat[i]),
        "dist_deg": float(np.sqrt(d2[i])),
        "dist_km": haversine_km(target_lon, target_lat, lon[i], lat[i]),
    }


def build_distance_table(targets: dict = TARGET_POINTS) -> pd.DataFrame:
    rows = []
    for grid, info in targets.items():
        n = nearest_merged_site(info["lon"], info["lat"])
        rows.append(
            {
                "grid": grid,
                "target_lon": info["lon"],
                "target_lat": info["lat"],
                "merged_site": n["site_index"],
                "merged_lon": n["lon"],
                "merged_lat": n["lat"],
                "dist_deg": n["dist_deg"],
                "dist_km": n["dist_km"],
            }
        )
    return pd.DataFrame(rows)


def _load_superpoint(grid: str, var: str) -> pd.Series:
    path = SUPERPOINT_DIR / f"{var}_{grid}.nc"
    with xr.open_dataset(path) as ds:
        name = var if var in ds else list(ds.data_vars)[0]
        return pd.Series(np.asarray(ds[name].values, dtype=np.float64), index=pd.DatetimeIndex(ds.time.values), name=var)


def _load_merged_site(var: str, site_index: int) -> pd.Series:
    path = MERGED_DIR / f"{var}_500m.nc"
    with xr.open_dataset(path) as ds:
        name = var if var in ds else list(ds.data_vars)[0]
        vals = np.asarray(ds[name].isel(site=site_index).values, dtype=np.float64)
        return pd.Series(vals, index=pd.DatetimeIndex(ds.time.values), name=var)


def load_aligned_pair(grid: str, site_index: int) -> dict[str, pd.DataFrame]:
    """Return per-variable DataFrame with columns SuperPoint / merged on common hours."""
    out = {}
    for var in VARIABLES:
        sp = _load_superpoint(grid, var)
        mg = _load_merged_site(var, site_index)
        df = pd.concat([sp.rename(LABEL_SP), mg.rename(LABEL_MG)], axis=1, join="inner")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        out[var] = df
    return out


def _finite_pair(a: np.ndarray, b: np.ndarray):
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def plot_grid_comparison(
    grid: str,
    data: dict[str, pd.DataFrame],
    meta: dict,
    *,
    output_png: Path,
    ts_years: tuple[int, int] = (2015, 2018),
) -> Path:
    """One figure: time series + seasonality + density + Q-Q for hs/tp/dp."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.1, 1.0, 1.0, 1.0], hspace=0.35, wspace=0.28)

    # --- Row 0: time series (daily mean in a window for readability) ---
    for j, var in enumerate(VARIABLES):
        ax = fig.add_subplot(gs[0, j])
        df = data[var]
        daily = df.resample("1D").mean()
        m = (daily.index.year >= ts_years[0]) & (daily.index.year <= ts_years[1])
        sub = daily.loc[m]
        ax.plot(sub.index, sub[LABEL_MG], color="k", lw=0.8, label=LABEL_MG)
        ax.plot(sub.index, sub[LABEL_SP], color="tab:orange", lw=0.8, alpha=0.85, label=LABEL_SP)
        ax.set_title(f"{var.upper()} daily mean ({ts_years[0]}–{ts_years[1]})")
        ax.set_ylabel(var)
        ax.grid(True, alpha=0.3)
        if j == 2:
            ax.legend(fontsize=7, loc="upper right")

    # --- Row 1: seasonality (monthly means) ---
    for j, var in enumerate(VARIABLES):
        ax = fig.add_subplot(gs[1, j])
        df = data[var]
        months = df.index.month
        x = np.arange(1, 13)
        sp_m = np.array([np.nanmean(df[LABEL_SP].values[months == m]) for m in x])
        mg_m = np.array([np.nanmean(df[LABEL_MG].values[months == m]) for m in x])
        ax.plot(x, mg_m, "-o", color="k", lw=1.4, markersize=5, label=LABEL_MG)
        ax.plot(x, sp_m, "-o", color="m", lw=1.4, markersize=5, label=LABEL_SP)
        ax.set_xticks(x)
        ax.set_xlabel("Month")
        ax.set_ylabel(var)
        ax.set_title(f"{var.upper()} seasonal cycle")
        ax.grid(True, alpha=0.3)
        if j == 2:
            ax.legend(fontsize=7, loc="best")

    # --- Row 2: density histograms ---
    for j, var in enumerate(VARIABLES):
        ax = fig.add_subplot(gs[2, j])
        a, b = _finite_pair(data[var][LABEL_MG].values, data[var][LABEL_SP].values)
        # subsample for speed if huge
        if len(a) > 200_000:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(a), size=200_000, replace=False)
            a, b = a[idx], b[idx]
        lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
        edges = np.linspace(lo, hi, 45)
        ax.hist(b, bins=edges, density=True, alpha=0.55, color="c", label=LABEL_SP)
        ax.hist(a, bins=edges, density=True, alpha=0.55, color="m", label=LABEL_MG)
        ax.set_title(f"{var.upper()} density")
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)
        if j == 2:
            ax.legend(fontsize=7, loc="upper right")

    # --- Row 3: Q-Q ---
    qs = np.linspace(0.01, 0.99, 200)
    for j, var in enumerate(VARIABLES):
        ax = fig.add_subplot(gs[3, j])
        a, b = _finite_pair(data[var][LABEL_MG].values, data[var][LABEL_SP].values)
        qa, qb = np.quantile(a, qs), np.quantile(b, qs)
        lims = (float(min(qa.min(), qb.min())), float(max(qa.max(), qb.max())))
        ax.plot(lims, lims, "k--", lw=1.0)
        ax.scatter(qa, qb, s=8, c="m", alpha=0.8)
        ax.set_title(f"{var.upper()} Q-Q")
        ax.set_xlabel(LABEL_MG)
        ax.set_ylabel(LABEL_SP)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{grid}: SuperPoint ({meta['target_lon']:.3f}, {meta['target_lat']:.3f}) vs "
        f"merged site {meta['merged_site']} ({meta['merged_lon']:.4f}, {meta['merged_lat']:.4f})  "
        f"| dist = {meta['dist_km']:.2f} km",
        fontsize=12,
        y=0.995,
    )
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_png}")
    return output_png


def run_all(
    *,
    output_dir: Path = OUTPUT_DIR,
    ts_years: tuple[int, int] = (2015, 2018),
) -> tuple[pd.DataFrame, dict[str, Path]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = build_distance_table()
    table_path = output_dir / "superpoint_vs_merged500m_distances.csv"
    table.to_csv(table_path, index=False, float_format="%.6f")
    print("\n=== Nearest merged_500m site for each SuperPoint target ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote {table_path}\n")

    paths: dict[str, Path] = {}
    for _, row in table.iterrows():
        grid = row["grid"]
        print(f"--- Loading {grid} (merged site {int(row['merged_site'])}) ---")
        data = load_aligned_pair(grid, int(row["merged_site"]))
        for var, df in data.items():
            print(f"  {var}: n={len(df)}  {df.index.min()} -> {df.index.max()}")
        meta = row.to_dict()
        paths[grid] = plot_grid_comparison(
            grid,
            data,
            meta,
            output_png=output_dir / f"compare_superpoint_vs_merged500m_{grid}.png",
            ts_years=ts_years,
        )
    return table, paths


if __name__ == "__main__":
    run_all()
