"""Compare historical vs earth3_veg_ssp245 at one site; save test_*.png in project root."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from utils.alr_plotting import (
    DEFAULT_VAR_YLABEL,
    _daily_agg,
    _ensure_datetime64_time,
    haversine_km,
)

ROOT = "/nfs/home/geocean/montanoj/New_data_Shoreshop"
HIST_FOLDER = os.path.join(ROOT, "outputs/merged_500m_binwaves_bmus")
SIM_FOLDER = os.path.join(ROOT, "outputs/earth3_veg_ssp245")
POINT_LAT, POINT_LON = 36.1942, -75.7316
VARS = ("hs", "tp", "dp")
SAVE_DIR = ROOT


def load_point_series(folder: str, var: str, lat: float, lon: float) -> xr.DataArray:
    path = os.path.join(folder, f"{var}_500m.nc")
    ds = xr.open_dataset(path)
    lats = np.asarray(ds["lat"].values)
    lons = np.asarray(ds["lon"].values)
    dists = haversine_km(lat, lon, lats, lons)
    site_idx = int(np.argmin(dists))
    print(
        f"{os.path.basename(folder)} | {var.upper()}: nearest site {site_idx} "
        f"({float(lats[site_idx]):.4f}, {float(lons[site_idx]):.4f}), "
        f"dist={float(dists[site_idx]):.3f} km"
    )
    da = ds[var].isel(site=site_idx)
    da = _ensure_datetime64_time(da)
    t = pd.to_datetime(da["time"].values)
    if len(t) > 1:
        dt_h = (t[1] - t[0]).total_seconds() / 3600.0
        if dt_h < 20:
            da = _daily_agg(da, name_hint=var)
    ds.close()
    return da.rename(var)


def monthly_mean(da: xr.DataArray, var: str) -> xr.DataArray:
    if var == "dp":

        def circ_mean(x, axis=None):
            rad = np.deg2rad(x)
            s = np.nanmean(np.sin(rad), axis=axis)
            c = np.nanmean(np.cos(rad), axis=axis)
            return (np.rad2deg(np.arctan2(s, c)) + 360.0) % 360.0

        return da.groupby("time.month").reduce(circ_mean, dim="time")
    return da.groupby("time.month").mean()


def main() -> None:
    daily: dict[str, tuple[xr.DataArray, xr.DataArray]] = {}
    for var in VARS:
        hist = load_point_series(HIST_FOLDER, var, POINT_LAT, POINT_LON)
        sim = load_point_series(SIM_FOLDER, var, POINT_LAT, POINT_LON)
        daily[var] = (sim, hist)

    fig_h, axes_h = plt.subplots(len(VARS), 1, figsize=(10, 3.0 * len(VARS)), sharex=False)
    axes_h = np.atleast_1d(axes_h)
    for row, var in enumerate(VARS):
        sim, hist = daily[var]
        vals = np.concatenate([np.asarray(sim.values).ravel(), np.asarray(hist.values).ravel()])
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            bins = np.linspace(vals.min(), vals.max(), 41)
            axes_h[row].hist(
                sim.values, bins=bins, density=True, color="turquoise", alpha=0.6, label=f"Simulated {var}"
            )
            axes_h[row].hist(
                hist.values, bins=bins, density=True, color="fuchsia", alpha=0.6, label=f"Historical {var}"
            )
            axes_h[row].legend(fontsize=7)
        axes_h[row].set_xlabel("Value")
        axes_h[row].set_ylabel("density")
        axes_h[row].set_title(var.upper())
        axes_h[row].grid(True, alpha=0.3)
    plt.suptitle("Histograms: Simulated vs Historical", y=1.02)
    plt.tight_layout()
    out_h = os.path.join(SAVE_DIR, "test_Histograms_hs_tp_dp.png")
    fig_h.savefig(out_h, dpi=150, bbox_inches="tight")
    print("Saved:", out_h)
    plt.close(fig_h)

    for var in VARS:
        sim, hist = daily[var]
        sim_mon = monthly_mean(sim, var)
        hist_mon = monthly_mean(hist, var)
        fig_s, ax = plt.subplots(1, 1, figsize=(10, 3.2))
        ax.plot(sim_mon["month"].values, sim_mon.values, "-o", color="fuchsia", label=f"Simulated {var}")
        ax.plot(hist_mon["month"].values, hist_mon.values, "-o", color="k", label=f"Historical {var}")
        ax.set_ylabel(DEFAULT_VAR_YLABEL.get(var, var.upper()))
        ax.set_title(f"{var.upper()} seasonal cycle")
        ax.set_xlabel("Month")
        ax.set_xticks(range(1, 13))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        plt.suptitle(f"Seasonality group: {var.upper()}", y=1.02)
        plt.tight_layout()
        out_s = os.path.join(SAVE_DIR, f"test_Seasonal_{var}.png")
        fig_s.savefig(out_s, dpi=150, bbox_inches="tight")
        print("Saved:", out_s)
        plt.close(fig_s)

    print("Done.")


if __name__ == "__main__":
    main()
