"""Grid WHACS seapoint wind helpers (load, PCA prep, maps). No cropping — uses pre-cut files."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

NC_DYNAMIC_UTILS = Path("/lustre/geocean/WORK/users/montanoj/personal/NC_Dynamic /utils")
sys.path.insert(0, str(NC_DYNAMIC_UTILS))

from wind_to_bathy_grid import SWAN_INPGRID  # noqa: E402

GRID_DIR = Path(__file__).resolve().parent
INPUTS_DIR = GRID_DIR / "inputs"
WIND_NC = {
    "uwnd": INPUTS_DIR / "uwnd.nc",
    "vwnd": INPUTS_DIR / "vwnd.nc",
}
WIND_VALID_START = np.datetime64("1980-01-01T00:00:00")


def restrict_valid_time(ds: xr.Dataset) -> xr.Dataset:
    """WHACS u/v are only valid from 1980 onwards."""
    return ds.sel(time=slice(WIND_VALID_START, None))


def build_inpgrid_lonlat() -> tuple[np.ndarray, np.ndarray]:
    """Geographic lon/lat at each SWAN INPGRID node (same layout as depth.dat)."""
    g = SWAN_INPGRID
    alpc = np.deg2rad(g["alpc"])
    m = np.arange(g["nx"] + 1, dtype=np.float64)
    n = np.arange(g["ny"] + 1, dtype=np.float64)
    mm, nn = np.meshgrid(m, n)
    lon2d = g["x0"] + mm * g["dx"] * np.cos(alpc) - nn * g["dy"] * np.sin(alpc)
    lat2d = g["y0"] + mm * g["dx"] * np.sin(alpc) + nn * g["dy"] * np.cos(alpc)
    return lon2d, lat2d


def open_wind_dataset(
    uwnd_path: Path | None = None,
    vwnd_path: Path | None = None,
) -> xr.Dataset:
    """Merge grid uwnd/vwnd WHACS seapoint files into one dataset for PCA."""
    uwnd_path = Path(uwnd_path or WIND_NC["uwnd"])
    vwnd_path = Path(vwnd_path or WIND_NC["vwnd"])
    if not uwnd_path.is_file() or not vwnd_path.is_file():
        raise FileNotFoundError(f"Missing wind input(s): {uwnd_path}, {vwnd_path}")
    with xr.open_dataset(uwnd_path) as u_ds, xr.open_dataset(vwnd_path) as v_ds:
        if not u_ds.time.identical(v_ds.time):
            raise ValueError("uwnd and vwnd time coordinates must match")
        if u_ds.sizes["seapoint"] != v_ds.sizes["seapoint"]:
            raise ValueError("uwnd and vwnd seapoint counts must match")
        return xr.merge([u_ds, v_ds], compat="override")


def pca_ready_wind(ds: xr.Dataset) -> xr.Dataset:
    """Keep seapoints with finite u/v over the full valid period (1980+)."""
    ds = restrict_valid_time(ds)
    valid = np.isfinite(ds["uwnd"]).all("time") & np.isfinite(ds["vwnd"]).all("time")
    return ds.isel(seapoint=valid.values)


def median_seapoint_lonlat(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Median lon/lat per seapoint (for EOF / cluster maps)."""
    lon = ds["longitude"].median("time").values
    lat = ds["latitude"].median("time").values
    return lon, lat


def plot_eof_maps(
    pca_winds,
    wind_ds: xr.Dataset,
    *,
    vars_to_plot: tuple[str, ...] | list[str] = ("uwnd", "vwnd"),
    num_eofs: int = 5,
    cmap: str = "RdBu_r",
    point_size: float = 18.0,
    map_pad: float = 0.15,
) -> None:
    """
    Plot PCA EOFs as geographic scatter maps on WHACS seapoints.

    bluemath ``plot_eofs()`` expects a 2-D lat/lon grid; seapoint PCA needs this instead.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    eofs = pca_winds.eofs.isel(n_component=slice(0, num_eofs))
    lon, lat = median_seapoint_lonlat(wind_ds)

    lon_inp, lat_inp = build_inpgrid_lonlat()
    extent = [
        float(lon_inp.min()) - map_pad,
        float(lon_inp.max()) + map_pad,
        float(lat_inp.min()) - map_pad,
        float(lat_inp.max()) + map_pad,
    ]

    for var in vars_to_plot:
        if var not in eofs:
            raise KeyError(f"EOF variable '{var}' not in {list(eofs.data_vars)}")

        ncols = min(3, num_eofs)
        nrows = int(np.ceil(num_eofs / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.2 * ncols, 3.6 * nrows),
            subplot_kw={"projection": ccrs.PlateCarree()},
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).ravel()

        fields = eofs[var].values
        vmax = np.nanpercentile(np.abs(fields), 98)
        vmax = vmax if vmax > 0 else 1.0
        norm = plt.Normalize(vmin=-vmax, vmax=vmax)

        last_sc = None
        for i in range(num_eofs):
            ax = axes[i]
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="0.92", edgecolor="none", zorder=0)
            ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.5, zorder=2)
            last_sc = ax.scatter(
                lon,
                lat,
                c=fields[i],
                s=point_size,
                cmap=cmap,
                norm=norm,
                transform=ccrs.PlateCarree(),
                zorder=1,
            )
            ax.set_title(f"EOF {i + 1} — {var}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        for j in range(num_eofs, len(axes)):
            axes[j].set_visible(False)

        if last_sc is not None:
            fig.colorbar(last_sc, ax=axes[:num_eofs].tolist(), fraction=0.03, pad=0.02)
        fig.suptitle(f"PCA EOF maps on WHACS seapoints ({var})", y=1.02)
        plt.show()


def load_wind_file_dat(wind_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read SWAN wind_file.dat (u block then v block)."""
    lines = Path(wind_path).read_text().strip().splitlines()
    half = len(lines) // 2
    u = np.array([[float(x) for x in line.split()] for line in lines[:half]], dtype=float)
    v = np.array([[float(x) for x in line.split()] for line in lines[half:]], dtype=float)
    return u, v, np.hypot(u, v)


def plot_wind_file_input(
    wind_path: Path,
    depth_path: Path | None = None,
    *,
    title: str | None = None,
    out_fig: Path | None = None,
    quiver_step: int = 12,
    map_pad: float = 0.15,
) -> dict[str, float]:
    """
    Plot cluster ``wind_file.dat`` on the SWAN INPGRID for QC before running SWAN.

    Uses depth > 0 mask only (same as dynamic cases). Fields are flipped vertically
    for map display (SWAN row order vs Cartopy).
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    wind_path = Path(wind_path)
    depth_path = Path(depth_path or INPUTS_DIR / "depth.dat")

    u, v, spd = load_wind_file_dat(wind_path)
    lon_inp, lat_inp = build_inpgrid_lonlat()

    u = np.flipud(u)
    v = np.flipud(v)
    spd = np.flipud(spd)

    depth = np.loadtxt(depth_path)
    wet = np.isfinite(depth) & (depth > 0)
    domain = np.flipud(wet)

    ocean = domain & np.isfinite(spd)
    zero_ocean = ocean & (spd <= 1e-6)
    stats = {
        "wet_cells": int(domain.sum()),
        "ocean_cells": int(ocean.sum()),
        "ocean_zero_wind": int(zero_ocean.sum()),
        "mean_u": float(np.nanmean(u[ocean])) if ocean.any() else float("nan"),
        "mean_v": float(np.nanmean(v[ocean])) if ocean.any() else float("nan"),
        "mean_spd": float(np.nanmean(spd[ocean])) if ocean.any() else float("nan"),
    }
    print(
        f"{wind_path.parent.name}: shape {u.shape}, wet={stats['wet_cells']}, "
        f"ocean zero-wind={stats['ocean_zero_wind']}/{stats['ocean_cells']}"
    )
    print(
        f"  mean u={stats['mean_u']:.2f}, v={stats['mean_v']:.2f}, "
        f"speed={stats['mean_spd']:.2f} m/s"
    )

    extent = [
        float(lon_inp.min()) - map_pad,
        float(lon_inp.max()) + map_pad,
        float(lat_inp.min()) - map_pad,
        float(lat_inp.max()) + map_pad,
    ]

    vmax = float(np.nanpercentile(spd[domain], 98)) if domain.any() else 10.0
    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="0.92", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6, zorder=2)

    u_plot = np.where(domain, u, np.nan)
    v_plot = np.where(domain, v, np.nan)
    spd_plot = np.where(domain, spd, np.nan)

    norm = plt.Normalize(vmin=0, vmax=vmax)
    qm = ax.pcolormesh(
        lon_inp,
        lat_inp,
        spd_plot,
        cmap="viridis",
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="auto",
        alpha=0.85,
        zorder=1,
    )

    lon_q = lon_inp[::quiver_step, ::quiver_step]
    lat_q = lat_inp[::quiver_step, ::quiver_step]
    u_q = u_plot[::quiver_step, ::quiver_step]
    v_q = v_plot[::quiver_step, ::quiver_step]
    spd_q = spd_plot[::quiver_step, ::quiver_step]
    quiver_ok = np.isfinite(u_q) & np.isfinite(v_q) & np.isfinite(spd_q)

    ax.quiver(
        lon_q[quiver_ok],
        lat_q[quiver_ok],
        u_q[quiver_ok],
        v_q[quiver_ok],
        spd_q[quiver_ok],
        cmap="viridis",
        norm=norm,
        transform=ccrs.PlateCarree(),
        scale=250,
        width=0.003,
        zorder=2,
    )
    plt.colorbar(qm, ax=ax, shrink=0.75, label="wind speed (m/s)")

    if title is None:
        title = f"BMU {wind_path.parent.name} — wind_file.dat"
    ax.set_title(
        f"{title}\n"
        f"mean u={stats['mean_u']:.1f}, v={stats['mean_v']:.1f}, "
        f"speed={stats['mean_spd']:.1f} m/s",
        fontsize=11,
    )

    if out_fig is not None:
        out_fig = Path(out_fig)
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_fig, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_fig}")

    plt.show()
    return stats
