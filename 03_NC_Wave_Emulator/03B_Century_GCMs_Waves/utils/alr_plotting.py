import glob
import os
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_PARTITION_PREFIX = {"hs": "phs", "tp": "ptp", "dp": "pdp"}
DEFAULT_VAR_YLABEL = {"hs": "Hs (m)", "tp": "Tp", "dp": "Dp"}


def _circular_mean_deg_np(a: np.ndarray, axis: int) -> np.ndarray:
    rad = np.deg2rad(a)
    sinm = np.nanmean(np.sin(rad), axis=axis)
    cosm = np.nanmean(np.cos(rad), axis=axis)
    ang = np.rad2deg(np.arctan2(sinm, cosm))
    return (ang + 360.0) % 360.0


def _is_directional_name(name: str) -> bool:
    n = str(name).lower()
    return n.startswith(("dp", "dm", "spr", "pdp", "pdm", "pspr"))


def _daily_agg(da: xr.DataArray, *, name_hint: str) -> xr.DataArray:
    if _is_directional_name(name_hint):
        out = da.resample(time="1D").reduce(_circular_mean_deg_np, dim="time")
    else:
        out = da.resample(time="1D").mean()
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; supports numpy broadcasting."""
    r = 6371.0
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))



def _ensure_folder(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _ensure_datetime64_time(da: xr.DataArray) -> xr.DataArray:
    """
    Ensure the ``time`` coordinate uses ``np.datetime64`` dtype.

    Some simulators emit time coordinates as Python strings, ``cftime`` objects,
    or generic ``object`` arrays. NetCDF historical files normally store time as
    ``np.datetime64``, so element-wise comparisons used by ``xr.align`` fail.
    This helper normalizes the coordinate so alignment can succeed.
    """
    if "time" not in da.coords:
        return da
    times = da["time"].values
    if np.issubdtype(times.dtype, np.datetime64):
        return da
    try:
        new_times = pd.to_datetime(times)
    except Exception:
        try:
            new_times = xr.CFTimeIndex(times).to_datetimeindex()
        except Exception:
            return da
    return da.assign_coords(time=np.asarray(new_times, dtype="datetime64[ns]"))


def _point_density_from_hist2d(x: np.ndarray, y: np.ndarray, bins: int = 60) -> np.ndarray:
    """
    Approximate per-point density using a 2D histogram lookup.
    """
    if x.size == 0:
        return np.array([])
    h, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    x_idx = np.clip(np.digitize(x, x_edges) - 1, 0, h.shape[0] - 1)
    y_idx = np.clip(np.digitize(y, y_edges) - 1, 0, h.shape[1] - 1)
    return h[x_idx, y_idx]


def _density_weighted_linear_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float] | None:
    """
    Fit y = m*x + b using weighted least squares.

    Returns (m, b) when a stable fit is possible, else None.
    """
    if x.size < 2 or y.size < 2 or w.size < 2:
        return None
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x = x[valid]
    y = y[valid]
    w = w[valid]
    if x.size < 2:
        return None
    x_mean = np.average(x, weights=w)
    y_mean = np.average(y, weights=w)
    var_x = np.average((x - x_mean) ** 2, weights=w)
    if var_x <= 0:
        return None
    cov_xy = np.average((x - x_mean) * (y - y_mean), weights=w)
    slope = cov_xy / var_x
    intercept = y_mean - slope * x_mean
    return float(slope), float(intercept)


def _normalize_project_root(project_root: str | None) -> str:
    """
    Normalize a project root folder.

    If project_root is None/empty, defaults to the current working directory.
    """
    if project_root is None or str(project_root).strip() == "":
        return os.getcwd()
    return os.path.abspath(os.path.expanduser(str(project_root)))


def _resolve_existing_folder(project_root: str | None, relative_folder: str) -> tuple[str, str]:
    """
    Resolve an existing folder from project_root with a local fallback.

    ``relative_folder`` may be absolute, or a path relative to project_root / cwd
    (including ``..`` components).

    Returns:
        tuple(resolved_project_root, resolved_folder)
    """
    requested_root = _normalize_project_root(project_root)
    candidates: list[tuple[str, str]] = []

    if os.path.isabs(relative_folder):
        candidates.append((os.path.dirname(relative_folder.rstrip(os.sep)), relative_folder))
    else:
        candidates.append(
            (requested_root, os.path.abspath(os.path.join(requested_root, relative_folder)))
        )
        cwd_root = os.getcwd()
        candidates.append(
            (cwd_root, os.path.abspath(os.path.join(cwd_root, relative_folder)))
        )

    for root, folder in candidates:
        if os.path.isdir(folder):
            if root != requested_root and not os.path.isabs(relative_folder):
                print(
                    "Warning: requested project_root does not contain "
                    f"'{relative_folder}'. Falling back to: {folder}"
                )
            return root, folder

    tried = "', '".join(folder for _, folder in candidates)
    raise FileNotFoundError(
        f"Could not find folder '{relative_folder}'. Tried: '{tried}'."
    )


def _plot_hist_qq_seasonal(
    daily_pairs: dict,
    variables: Sequence[str],
    systems: Sequence[int],
    grid: int,
    save_folder: str,
    tag: str,
    left_label: str,
    right_label: str,
    partition_prefix: dict,
    original_band: str | None = "std",
    band_color: str = "0.7",
    band_alpha: float = 0.25,
):
    fig_h, axes_h = plt.subplots(len(variables), len(systems), figsize=(10, 9), sharex=False)
    for row, var in enumerate(variables):
        for col, system in enumerate(systems):
            axh = axes_h[row, col]
            left, right = daily_pairs[(var, system)]
            vals = np.concatenate([left.values.ravel(), right.values.ravel()])
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            bins = np.linspace(vals.min(), vals.max(), 41)
            axh.hist(left.values, bins=bins, density=True, color="turquoise", alpha=0.6, label=left_label)
            axh.hist(right.values, bins=bins, density=True, color="fuchsia", alpha=0.6, label=right_label)
            axh.set_xlabel("Value")
            axh.set_ylabel("density")
            axh.set_title(f"{var.upper()} partition {system}")
            axh.grid(True, alpha=0.3)
            axh.legend(fontsize=7)
    plt.suptitle(f"Histograms ({tag}, grid {grid})", y=1.02)
    plt.tight_layout()
    out = os.path.join(save_folder, f"Histograms_allvars_grid{grid}_{tag}.png")
    fig_h.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved:", out)
    plt.show()

    fig_q, axes_q = plt.subplots(len(variables), len(systems), figsize=(10, 9), sharex=False, sharey=False)
    for row, var in enumerate(variables):
        for col, system in enumerate(systems):
            axq = axes_q[row, col]
            left, right = daily_pairs[(var, system)]
            y = np.asarray(left.values).ravel()
            x = np.asarray(right.values).ravel()
            x = x[np.isfinite(x)]
            y = y[np.isfinite(y)]
            if x.size == 0 or y.size == 0:
                continue
            q = np.linspace(0.01, 0.99, 100)
            x_q = np.quantile(x, q)
            y_q = np.quantile(y, q)
            axq.scatter(x_q, y_q, s=5, color="purple")
            lo = min(x_q.min(), y_q.min())
            hi = max(x_q.max(), y_q.max())
            axq.plot([lo, hi], [lo, hi], "k--", lw=1)
            axq.set_xlabel(right_label)
            axq.set_ylabel(left_label)
            axq.set_title(f"{var.upper()} partition {system}")
            axq.grid(True, alpha=0.3)
    plt.suptitle(f"QQ plots ({tag}, grid {grid})", y=1.02)
    plt.tight_layout()
    out = os.path.join(save_folder, f"QQ_allvars_grid{grid}_{tag}.png")
    fig_q.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved:", out)
    plt.show()

    for var in variables:
        left0, right0 = daily_pairs[(var, systems[0])]
        left1, right1 = daily_pairs[(var, systems[1])]
        left0_mon = left0.groupby("time.month").mean()
        right0_mon = right0.groupby("time.month").mean()
        left1_mon = left1.groupby("time.month").mean()
        right1_mon = right1.groupby("time.month").mean()
        months = left0_mon["month"].values
        fig_s, axes_s = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        if original_band == "std":
            r0_std = right0.groupby("time.month").std()
            r1_std = right1.groupby("time.month").std()
        elif original_band == "minmax":
            r0_min = right0.groupby("time.month").min()
            r0_max = right0.groupby("time.month").max()
            r1_min = right1.groupby("time.month").min()
            r1_max = right1.groupby("time.month").max()

        axes_s[0].plot(months, left0_mon.values, "-o", color="fuchsia", label=f"{left_label} {var}0")
        axes_s[0].plot(months, right0_mon.values, "-o", color="k", label=f"{right_label} {partition_prefix[var]}0")
        if original_band == "std":
            axes_s[0].fill_between(
                months,
                (right0_mon - r0_std).values,
                (right0_mon + r0_std).values,
                color=band_color,
                alpha=band_alpha,
                linewidth=0,
                label=f"{right_label} ±1σ",
            )
        elif original_band == "minmax":
            axes_s[0].fill_between(
                months,
                r0_min.values,
                r0_max.values,
                color=band_color,
                alpha=band_alpha,
                linewidth=0,
                label=f"{right_label} min–max",
            )
        axes_s[0].set_ylabel(DEFAULT_VAR_YLABEL.get(var, var.upper()))
        axes_s[0].set_title(f"{var.upper()} seasonal cycle (sea, grid {grid})")
        axes_s[0].grid(True, alpha=0.3)
        axes_s[0].legend()
        axes_s[1].plot(months, left1_mon.values, "-o", color="fuchsia", label=f"{left_label} {var}1")
        axes_s[1].plot(months, right1_mon.values, "-o", color="k", label=f"{right_label} {partition_prefix[var]}1")
        if original_band == "std":
            axes_s[1].fill_between(
                months,
                (right1_mon - r1_std).values,
                (right1_mon + r1_std).values,
                color=band_color,
                alpha=band_alpha,
                linewidth=0,
                label=f"{right_label} ±1σ",
            )
        elif original_band == "minmax":
            axes_s[1].fill_between(
                months,
                r1_min.values,
                r1_max.values,
                color=band_color,
                alpha=band_alpha,
                linewidth=0,
                label=f"{right_label} min–max",
            )
        axes_s[1].set_ylabel(DEFAULT_VAR_YLABEL.get(var, var.upper()))
        axes_s[1].set_title(f"{var.upper()} seasonal cycle (swell, grid {grid})")
        axes_s[1].set_xlabel("Month")
        axes_s[1].set_xticks(range(1, 13))
        axes_s[1].grid(True, alpha=0.3)
        axes_s[1].legend()
        plt.tight_layout()
        out = os.path.join(save_folder, f"Seasonal_{var}_grid{grid}_{tag}.png")
        fig_s.savefig(out, dpi=150, bbox_inches="tight")
        print("Saved:", out)
        plt.show()



def _with_suffix(base: str, suffix: str | None) -> str:
    if not suffix:
        return base
    return f"{base}_{suffix}"


def _split_var_family(name: str) -> tuple[str, str]:
    """
    Split variable name into (family, level).

    Examples:
        hs -> ("hs", "")
        phs0 -> ("hs", "0")
        ptp2 -> ("tp", "2")
        dp3 -> ("dp", "3")
    """
    var = str(name).lower()
    base = var[1:] if var.startswith("p") else var
    idx = len(base)
    while idx > 0 and base[idx - 1].isdigit():
        idx -= 1
    family = base[:idx]
    level = base[idx:]
    return family, level


def _resolve_bulk_nc_var_name(var: str, ds: xr.Dataset) -> str | None:
    """
    Map logical bulk names (e.g. phs0, ptp1) to the variable name stored in the NetCDF
    (often hs, tp, dp for WAVEWATCH-style exports).
    """
    if var in ds.data_vars:
        return str(var)
    family, _ = _split_var_family(var)
    if family in ds.data_vars:
        return family
    fam_lower = family.lower()
    for name in ds.data_vars:
        if str(name).lower() == fam_lower:
            return str(name)
    return None


def _group_vars_by_family(variables: Sequence[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for var in variables:
        family, _ = _split_var_family(var)
        grouped.setdefault(family, []).append(var)
    return grouped


def _group_vars_by_level_and_family(variables: Sequence[str], families: Sequence[str] = ("hs", "tp", "dp")) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for level in sorted({_split_var_family(v)[1] for v in variables}, key=lambda x: (x != "", x)):
        row_vars: list[str] = []
        for fam in families:
            candidate = fam if level == "" else f"p{fam}{level}"
            if candidate in variables:
                row_vars.append(candidate)
        if row_vars:
            grouped[level] = row_vars
    return grouped


def _resolve_var_nc_path(
    folder: str,
    var: str,
    *,
    filename_template: str | None = None,
    bulk_files: Mapping[str, str] | None = None,
) -> str | None:
    """
    Resolve a NetCDF path for ``var`` inside ``folder``.

    Order: explicit ``bulk_files`` → ``filename_template`` → common name patterns →
    first ``{var}_*.nc`` match.
    """
    if bulk_files and var in bulk_files:
        path = bulk_files[var]
        if not os.path.isabs(path):
            path = os.path.join(folder, path)
        return path if os.path.isfile(path) else None

    candidates: list[str] = []
    if filename_template:
        candidates.append(filename_template.format(var=var))
    candidates.extend(
        [
            f"{var}_500m.nc",
            f"{var}_CenturyHindcast_500m.nc",
            f"{var}_NorthCarolina.nc",
            f"{var}_merged_all.nc",
            f"{var}.nc",
        ]
    )
    for name in candidates:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path

    matches = sorted(glob.glob(os.path.join(folder, f"{var}_*.nc")))
    return matches[0] if matches else None


def _nearest_site_index(lats, lons, point_lat: float, point_lon: float) -> tuple[int, float, float, float]:
    dists_km = haversine_km(point_lat, point_lon, np.asarray(lats), np.asarray(lons))
    site_idx = int(np.argmin(dists_km))
    return (
        site_idx,
        float(dists_km[site_idx]),
        float(np.asarray(lats)[site_idx]),
        float(np.asarray(lons)[site_idx]),
    )


def _render_bulk_daily_pair_figures(
    daily_pairs: dict,
    loaded_variables: Sequence[str],
    save_folder: str,
    *,
    figure_suffix: str | None = None,
    original_band: str | None = "std",
    band_color: str = "0.7",
    band_alpha: float = 0.25,
):
    """Shared histograms / QQ / seasonal / scatter figures for (sim, hist) daily pairs."""
    fig_h, axes_h = plt.subplots(len(loaded_variables), 1, figsize=(10, 3.0 * len(loaded_variables)), sharex=False)
    axes_h = np.atleast_1d(axes_h)
    fig_q, axes_q = plt.subplots(
        1,
        len(loaded_variables),
        figsize=(5.2 * len(loaded_variables), 4.8),
        sharex=False,
        sharey=False,
    )
    axes_q = np.atleast_1d(axes_q)
    for row, var in enumerate(loaded_variables):
        emu, bulk_daily = daily_pairs[var]
        vals = np.concatenate([emu.values.ravel(), bulk_daily.values.ravel()])
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            bins = np.linspace(vals.min(), vals.max(), 41)
            axes_h[row].hist(emu.values, bins=bins, density=True, color="turquoise", alpha=0.6, label=f"Simulated {var}")
            axes_h[row].hist(
                bulk_daily.values, bins=bins, density=True, color="fuchsia", alpha=0.6, label=f"Historical {var}"
            )
            axes_h[row].legend(fontsize=7)
        axes_h[row].set_xlabel("Value")
        axes_h[row].set_ylabel("density")
        axes_h[row].set_title(var.upper())
        axes_h[row].grid(True, alpha=0.3)

        y = np.asarray(emu.values).ravel()
        x = np.asarray(bulk_daily.values).ravel()
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if x.size > 0 and y.size > 0:
            q = np.linspace(0.01, 0.99, 100)
            x_q = np.quantile(x, q)
            y_q = np.quantile(y, q)
            axes_q[row].scatter(x_q, y_q, s=5, color="fuchsia")
            lo = min(x_q.min(), y_q.min())
            hi = max(x_q.max(), y_q.max())
            axes_q[row].plot([lo, hi], [lo, hi], "k--", lw=1)
        axes_q[row].set_xlabel("Historical")
        axes_q[row].set_ylabel("Simulated")
        axes_q[row].set_title(var.upper())
        axes_q[row].grid(True, alpha=0.3)

    plt.suptitle("Histograms: Simulated vs Historical", y=1.02)
    plt.tight_layout()
    out = os.path.join(
        save_folder,
        f"Histograms_allvars_bulk_random_member{'' if not figure_suffix else '_' + figure_suffix}.png",
    )
    fig_h.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved:", out)
    plt.show()

    plt.suptitle("QQ plots: Historical vs Simulated", y=1.02)
    plt.tight_layout()
    out = os.path.join(
        save_folder, f"QQ_allvars_Historical_vs_Simulated{'' if not figure_suffix else '_' + figure_suffix}.png"
    )
    fig_q.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved:", out)
    plt.show()

    family_groups = _group_vars_by_family(loaded_variables)
    for family, group_vars in family_groups.items():
        fig_s, axes_s = plt.subplots(len(group_vars), 1, figsize=(10, 3.2 * len(group_vars)), sharex=True)
        axes_s = np.atleast_1d(axes_s)
        for row, var in enumerate(group_vars):
            emu, bulk_daily = daily_pairs[var]
            emu_mon = emu.groupby("time.month").mean()
            bulk_mon = bulk_daily.groupby("time.month").mean()
            if original_band == "std":
                bulk_std = bulk_daily.groupby("time.month").std()
            elif original_band == "minmax":
                bulk_min = bulk_daily.groupby("time.month").min()
                bulk_max = bulk_daily.groupby("time.month").max()
            months = emu_mon["month"].values
            ax = axes_s[row]
            ax.plot(months, emu_mon.values, "-o", color="fuchsia", label=f"Simulated {var}")
            ax.plot(months, bulk_mon.values, "-o", color="k", label=f"Historical {var}")
            if original_band == "std":
                ax.fill_between(
                    months,
                    (bulk_mon - bulk_std).values,
                    (bulk_mon + bulk_std).values,
                    color=band_color,
                    alpha=band_alpha,
                    linewidth=0,
                    label="Historical +/-1sigma",
                )
            elif original_band == "minmax":
                ax.fill_between(
                    months,
                    bulk_min.values,
                    bulk_max.values,
                    color=band_color,
                    alpha=band_alpha,
                    linewidth=0,
                    label="Historical min-max",
                )
            ax.set_ylabel(DEFAULT_VAR_YLABEL.get(family, family.upper()))
            ax.set_title(f"{var.upper()} seasonal cycle")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        axes_s[-1].set_xlabel("Month")
        axes_s[-1].set_xticks(range(1, 13))
        plt.suptitle(f"Seasonality group: {family.upper()}", y=1.02)
        plt.tight_layout()
        out = os.path.join(
            save_folder,
            f"Seasonal_group_{family}_bulk{'' if not figure_suffix else '_' + figure_suffix}.png",
        )
        fig_s.savefig(out, dpi=150, bbox_inches="tight")
        print("Saved:", out)
        plt.show()

    scatter_groups = _group_vars_by_level_and_family(loaded_variables, families=("hs", "tp", "dp"))
    for level, group_vars in scatter_groups.items():
        fig_sc, axes_sc = plt.subplots(1, len(group_vars), figsize=(5.2 * len(group_vars), 4.8), sharex=False, sharey=False)
        axes_sc = np.atleast_1d(axes_sc)
        for col, var in enumerate(group_vars):
            emu, bulk_daily = daily_pairs[var]
            emu_ov, bulk_ov = xr.align(emu, bulk_daily, join="inner")
            sx = np.asarray(bulk_ov.values).ravel()
            sy = np.asarray(emu_ov.values).ravel()
            valid = np.isfinite(sx) & np.isfinite(sy)
            sx = sx[valid]
            sy = sy[valid]
            ax = axes_sc[col]
            if sx.size > 0:
                z = _point_density_from_hist2d(sx, sy, bins=65)
                order = np.argsort(z)
                sx_plot = sx[order]
                sy_plot = sy[order]
                z_plot = z[order]
                ax.scatter(
                    sx_plot,
                    sy_plot,
                    c=z_plot,
                    s=8,
                    cmap="YlGnBu_r",
                    alpha=0.75,
                    edgecolors="none",
                )
                lo = min(sx.min(), sy.min())
                hi = max(sx.max(), sy.max())
                ax.plot([lo, hi], [lo, hi], "k--", lw=1)
                rms = np.sqrt(np.mean((sy - sx) ** 2))
                bias = np.mean(sy - sx)
                if sx.size > 1:
                    fit = _density_weighted_linear_fit(sx, sy, z)
                    xfit = np.linspace(lo, hi, 100)
                    if fit is not None:
                        ax.plot(xfit, fit[0] * xfit + fit[1], color="darkorange", lw=1.2)
                    r = np.corrcoef(sx, sy)[0, 1]
                    ax.text(
                        0.96,
                        0.09,
                        f"rho={r:.2f}\nRMS={rms:.2f}\nBIAS={bias:.2f}",
                        transform=ax.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                        color="black",
                    )
            ax.set_title(f"{var.upper()} overlap scatter")
            ax.set_xlabel("Historical (overlap)")
            ax.set_ylabel("Simulated (overlap)")
            ax.grid(True, alpha=0.3)
        level_tag = "base" if level == "" else level
        plt.suptitle(f"Scatter group level {level_tag}: Historical vs Simulated", y=1.02)
        plt.tight_layout()
        out = os.path.join(
            save_folder,
            f"Scatter_group_level_{level_tag}_Historical_vs_Simulated_overlap{'' if not figure_suffix else '_' + figure_suffix}.png",
        )
        fig_sc.savefig(out, dpi=150, bbox_inches="tight")
        print("Saved:", out)
        plt.show()


def _plot_site_location_map(
    lats,
    lons,
    point_lat: float,
    point_lon: float,
    site_lat: float,
    site_lon: float,
    save_folder: str,
    *,
    site_label: str | int | None = None,
    distance_km: float | None = None,
    figure_suffix: str | None = None,
    pad_deg: float = 1.5,
) -> str:
    """Map of all sites with target point and selected nearest site (first figure)."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as exc:
        raise ImportError(
            "plot_from_folder location map requires cartopy. "
            "Install cartopy or set plot_location_map=False."
        ) from exc

    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)

    fig = plt.figure(figsize=(9, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(
        [float(np.nanmin(lons)) - pad_deg, float(np.nanmax(lons)) + pad_deg,
         float(np.nanmin(lats)) - pad_deg, float(np.nanmax(lats)) + pad_deg],
        crs=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="aliceblue", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.6, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    ax.scatter(
        lons, lats, s=8, c="tab:blue", alpha=0.35,
        transform=ccrs.PlateCarree(), label="Dataset sites", zorder=3,
    )
    ax.scatter(
        point_lon, point_lat, s=140, c="red", marker="*", edgecolor="k", linewidth=0.7,
        transform=ccrs.PlateCarree(), label="Target point", zorder=5,
    )
    ax.scatter(
        site_lon, site_lat, s=100, c="yellow", marker="o", edgecolor="k", linewidth=0.7,
        transform=ccrs.PlateCarree(), label="Nearest site used", zorder=4,
    )

    title = f"Selected site ({site_lat:.4f}, {site_lon:.4f})"
    if site_label is not None:
        title = f"Selected site {site_label} ({site_lat:.4f}, {site_lon:.4f})"
    if distance_km is not None:
        title += f" — {distance_km:.2f} km from target"
    ax.set_title(title)
    ax.legend(loc="lower left")

    suffix = f"_{figure_suffix}" if figure_suffix else ""
    out = os.path.join(save_folder, f"Location_map_selected_site{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved:", out)
    plt.show()
    return out


def plot_from_folder(
    data_folder: str,
    point_lat: float,
    point_lon: float,
    project_root: str | None = None,
    variables: Sequence[str] = ("hs", "tp", "dp"),
    filename_template: str | None = None,
    bulk_files: dict | None = None,
    historical_folder: str | None = None,
    historical_filename_template: str | None = None,
    save_subfolder: str = "outputs/Figures/ALR_bulk_bootstrap_month",
    figure_suffix: str | None = None,
    plot_non_overlap_timeseries: bool = True,
    original_band: str | None = "std",
    band_color: str = "0.7",
    band_alpha: float = 0.25,
    shoreline_orientation_deg: float | None = None,
    plot_location_map: bool = True,
):
    """
    Plot historical vs simulated series from any folder of already-produced NetCDFs.

    Does **not** re-sample BMUs. Opens files under ``data_folder`` (any absolute or
    relative path), selects the nearest site to (``point_lat``, ``point_lon``), and
    uses that series as Simulated.

    By default the first figure is a location map of the target point and nearest site
    (``plot_location_map=True``).

    Historical daily means come from ``historical_folder`` if given, otherwise from
    the NetCDF ``source_dir`` attribute when present, matched by ``src_index`` or
    nearest lat/lon so figures match the saved files.

    If ``shoreline_orientation_deg`` is set and both ``hs`` and ``dp`` are loaded,
    also plots the CERC-like longshore transport index (timeseries + cumulative).

    Examples::

        plot_from_folder("outputs/century/OffshorePoints", 36.19, -75.73, ...)
        plot_from_folder("/abs/path/to/any/bootstrap_dir", 33.9, -78.2, ...)
        plot_from_folder("../02_Wind_Metamodel/outputs/NorthCarolina", ...)
    """
    project_root, data_dir = _resolve_existing_folder(project_root, data_folder)
    save_folder = _ensure_folder(os.path.join(project_root, _with_suffix(save_subfolder, figure_suffix)))

    daily_pairs: dict = {}
    loaded_variables: list[str] = []
    resolved_hist_folder: str | None = None
    location_map_done = False

    for var in variables:
        path = _resolve_var_nc_path(
            data_dir, var, filename_template=filename_template, bulk_files=bulk_files
        )
        if path is None:
            print(f"Warning: no NetCDF for '{var}' in '{data_dir}'. Skipping.")
            continue

        ds = xr.open_dataset(path)
        nc_var = _resolve_bulk_nc_var_name(var, ds)
        if nc_var is None:
            print(
                f"Warning: no variable for '{var}' in '{path}'. "
                f"Available vars={list(ds.data_vars)}. Skipping."
            )
            ds.close()
            continue

        site_idx, nearest_dist_km, nearest_lat, nearest_lon = _nearest_site_index(
            ds["lat"].values, ds["lon"].values, point_lat, point_lon
        )
        site_label = ds["site"].values[site_idx] if "site" in ds.coords or "site" in ds.dims else site_idx
        print(
            f"{var.upper()}: {os.path.basename(path)} | target ({point_lat:.4f}, {point_lon:.4f}) -> "
            f"nearest site {site_label} ({nearest_lat:.4f}, {nearest_lon:.4f}), "
            f"distance={nearest_dist_km:.3f} km"
        )

        if plot_location_map and not location_map_done:
            _plot_site_location_map(
                ds["lat"].values,
                ds["lon"].values,
                point_lat,
                point_lon,
                nearest_lat,
                nearest_lon,
                save_folder,
                site_label=site_label,
                distance_km=nearest_dist_km,
                figure_suffix=figure_suffix,
            )
            location_map_done = True

        emu = _ensure_datetime64_time(ds[nc_var].isel(site=site_idx).load())
        emu = emu.rename(f"{var}_boot")
        src_index = None
        if "src_index" in ds:
            src_index = int(ds["src_index"].isel(site=site_idx).values)

        hist_folder_guess = historical_folder
        if hist_folder_guess is None:
            hist_folder_guess = ds.attrs.get("source_dir")
        source_file = ds.attrs.get("source_file")
        ds.close()

        if hist_folder_guess is None:
            print(f"Warning: no historical_folder / source_dir for '{var}'. Skipping.")
            continue

        hist_candidates: list[str] = []
        if os.path.isabs(str(hist_folder_guess)):
            hist_candidates.append(str(hist_folder_guess))
        else:
            hist_candidates.append(os.path.abspath(os.path.join(project_root, str(hist_folder_guess))))
            hist_candidates.append(os.path.abspath(os.path.join(os.getcwd(), str(hist_folder_guess))))

        hist_dir = next((p for p in hist_candidates if os.path.isdir(p)), None)
        if hist_dir is None:
            print(
                f"Warning: historical folder not found for '{var}' "
                f"(tried {hist_candidates}). Skipping."
            )
            continue
        if resolved_hist_folder is None:
            resolved_hist_folder = hist_dir
            print(f"Historical folder: {hist_dir}")

        hist_template = historical_filename_template
        hist_bulk = None
        if source_file and hist_template is None:
            # Prefer the exact source basename only when it exists in hist_dir
            # (attrs can point at a different product naming scheme than historical_folder).
            src_path = os.path.join(hist_dir, str(source_file))
            if os.path.isfile(src_path):
                hist_bulk = {var: str(source_file)}
        hist_path = _resolve_var_nc_path(
            hist_dir,
            var,
            filename_template=hist_template,
            bulk_files=hist_bulk,
        )
        if hist_path is None:
            print(
                f"Warning: historical file for '{var}' not found in '{hist_dir}' "
                f"(source_file={source_file!r}). Skipping."
            )
            continue

        ds_h = xr.open_dataset(hist_path)
        hist_nc_var = _resolve_bulk_nc_var_name(var, ds_h)
        if hist_nc_var is None:
            print(f"Warning: no historical variable for '{var}' in '{hist_path}'. Skipping.")
            ds_h.close()
            continue

        hist_da = None
        if src_index is not None and "site" in ds_h.dims and 0 <= src_index < ds_h.sizes["site"]:
            # Trust src_index only when that site's coordinates match the simulated site
            # (indices are not portable across different product grids).
            cand_lat = float(ds_h["lat"].isel(site=src_index).values)
            cand_lon = float(ds_h["lon"].isel(site=src_index).values)
            if abs(cand_lat - nearest_lat) < 1e-4 and abs(cand_lon - nearest_lon) < 1e-4:
                hist_da = ds_h[hist_nc_var].isel(site=src_index)
                print(f"  historical via src_index={src_index} ({os.path.basename(hist_path)})")
        if hist_da is None:
            h_idx, h_dist, h_lat, h_lon = _nearest_site_index(
                ds_h["lat"].values, ds_h["lon"].values, nearest_lat, nearest_lon
            )
            hist_da = ds_h[hist_nc_var].isel(site=h_idx)
            print(
                f"  historical nearest site {h_idx} ({h_lat:.4f}, {h_lon:.4f}), "
                f"distance={h_dist:.3f} km ({os.path.basename(hist_path)})"
            )

        bulk_daily_full = _ensure_datetime64_time(_daily_agg(hist_da.load(), name_hint=var))
        ds_h.close()

        emu_ov, hist_ov = xr.align(emu, bulk_daily_full, join="inner")
        if emu_ov.sizes.get("time", 0) == 0:
            print(f"Warning: no time overlap for '{var}'. Skipping.")
            continue

        daily_pairs[var] = (emu_ov, hist_ov)
        loaded_variables.append(var)

        emu_plot, hist_plot = emu_ov, hist_ov
        if plot_non_overlap_timeseries:
            emu_plot, hist_plot = xr.align(emu, bulk_daily_full, join="outer")

        fig_ts, ax = plt.subplots(1, 1, figsize=(14, 4), sharex=True)
        ax.plot(hist_plot["time"].values, hist_plot.values, lw=1.5, color="k", label=f"Historical {var}")
        ax.plot(emu_plot["time"].values, emu_plot.values, lw=1.2, color="fuchsia", alpha=0.8, label=f"Simulated {var}")
        ax.set_ylabel(var.upper())
        ax.set_xlabel("Time")
        ax.set_title(f"{var.upper()} - Historical vs Simulated")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        if plot_non_overlap_timeseries:
            tvals = pd.to_datetime(hist_plot["time"].values)
            if tvals.size > 0:
                ax.set_xlim(tvals.min(), tvals.max())
        else:
            ax.set_xlim(pd.to_datetime("1980-01-01"), pd.to_datetime("2024-12-31"))
        out = os.path.join(
            save_folder,
            f"Daily_mean_{var}_Historical_vs_Simulated{'' if not figure_suffix else '_' + figure_suffix}.png",
        )
        fig_ts.savefig(out, dpi=150, bbox_inches="tight")
        print("Saved:", out)
        plt.show()

    if not loaded_variables:
        raise ValueError(
            f"No variables loaded from '{data_dir}'. Check folder contents and 'variables'."
        )

    _render_bulk_daily_pair_figures(
        daily_pairs,
        loaded_variables,
        save_folder,
        figure_suffix=figure_suffix,
        original_band=original_band,
        band_color=band_color,
        band_alpha=band_alpha,
    )

    if shoreline_orientation_deg is not None and "hs" in daily_pairs and "dp" in daily_pairs:
        plot_longshore_transport(
            daily_pairs,
            save_folder,
            shoreline_orientation_deg=shoreline_orientation_deg,
            figure_suffix=figure_suffix,
            plot_non_overlap_timeseries=plot_non_overlap_timeseries,
        )

    return daily_pairs, save_folder


def longshore_transport_index(
    H: xr.DataArray,
    Dp_deg: xr.DataArray,
    shoreline_orientation_deg: float,
    *,
    K: float = 0.23,
) -> xr.DataArray:
    """
    Relative CERC-like longshore transport index: Q ~ K * Hs**(5/2) * sin(2*alpha).

    ``alpha`` is wave direction minus shoreline orientation (degrees).
    No breaking-wave transformation is applied.
    """
    alpha_rad = np.deg2rad(Dp_deg - shoreline_orientation_deg)
    return (H ** (5.0 / 2.0)) * np.sin(2.0 * alpha_rad) * K


def plot_longshore_transport(
    daily_pairs: Mapping[str, tuple[xr.DataArray, xr.DataArray]],
    save_folder: str,
    shoreline_orientation_deg: float = 72.0,
    *,
    figure_suffix: str | None = None,
    K: float = 0.23,
    xlim: tuple[str, str] | None = ("1980-01-01", "2024-12-31"),
    plot_non_overlap_timeseries: bool = False,
    hist_label: str = "Historical Qs",
    sim_label: str = "Simulated Qs",
):
    """
    Plot longshore transport index and cumulative series from ``daily_pairs``
    produced by ``plot_from_folder`` (requires ``hs`` and ``dp``).
    """
    if "hs" not in daily_pairs or "dp" not in daily_pairs:
        raise ValueError("plot_longshore_transport requires daily_pairs entries for 'hs' and 'dp'.")

    emu_hs, bulk_hs = daily_pairs["hs"]
    emu_dp, bulk_dp = daily_pairs["dp"]

    Qs_bulk = longshore_transport_index(bulk_hs, bulk_dp, shoreline_orientation_deg, K=K)
    Qs_emu = longshore_transport_index(emu_hs, emu_dp, shoreline_orientation_deg, K=K)
    Qs_bulk = Qs_bulk.rename("Qs_Historical")
    Qs_emu = Qs_emu.rename("Qs_Simulated")

    if plot_non_overlap_timeseries:
        Qs_emu_plot, Qs_bulk_plot = xr.align(Qs_emu, Qs_bulk, join="outer")
    else:
        Qs_emu_plot, Qs_bulk_plot = xr.align(Qs_emu, Qs_bulk, join="inner")

    Qs_bulk_cum = Qs_bulk_plot.cumsum("time")
    Qs_emu_cum = Qs_emu_plot.cumsum("time")

    suffix = f"_{figure_suffix}" if figure_suffix else ""

    fig_qs, ax_qs = plt.subplots(1, 1, figsize=(14, 4), sharex=True)
    ax_qs.plot(Qs_bulk_plot["time"].values, Qs_bulk_plot.values, color="k", lw=1.0, label=hist_label)
    ax_qs.plot(
        Qs_emu_plot["time"].values,
        Qs_emu_plot.values,
        color="fuchsia",
        lw=0.8,
        alpha=0.8,
        label=sim_label,
    )
    ax_qs.set_ylabel("Qs")
    ax_qs.set_xlabel("Time")
    ax_qs.set_title(
        f"Longshore sediment transport - CERC (orientation={shoreline_orientation_deg:g}°)"
    )
    ax_qs.grid(True, alpha=0.3)
    ax_qs.legend(fontsize=8)
    if xlim is not None:
        ax_qs.set_xlim(pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1]))
    qs_path = os.path.join(save_folder, f"Qs_timeseries_bulk_vs_emulator{suffix}.png")
    fig_qs.savefig(qs_path, dpi=150, bbox_inches="tight")
    print("Saved:", qs_path)
    plt.show()

    fig_qs_cum, ax_qs_cum = plt.subplots(1, 1, figsize=(14, 4), sharex=True)
    ax_qs_cum.plot(
        Qs_bulk_cum["time"].values,
        Qs_bulk_cum.values,
        color="k",
        lw=1.0,
        label=f"{hist_label} (cumulative)",
    )
    ax_qs_cum.plot(
        Qs_emu_cum["time"].values,
        Qs_emu_cum.values,
        color="fuchsia",
        lw=1.0,
        label=f"{sim_label} (cumulative)",
    )
    ax_qs_cum.set_ylabel("Cumulative Qs")
    ax_qs_cum.set_xlabel("Time")
    ax_qs_cum.set_title("Cumulative longshore sediment transport (Historical vs Simulated)")
    ax_qs_cum.grid(True, alpha=0.3)
    ax_qs_cum.legend(fontsize=8)
    if xlim is not None:
        ax_qs_cum.set_xlim(pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1]))
    qs_cum_path = os.path.join(save_folder, f"Qs_cumulative_bulk_vs_emulator{suffix}.png")
    fig_qs_cum.savefig(qs_cum_path, dpi=150, bbox_inches="tight")
    print("Saved:", qs_cum_path)
    plt.show()

    return {"Qs_historical": Qs_bulk_plot, "Qs_simulated": Qs_emu_plot}


# Backward-compatible alias
plot_from_century_outputs = plot_from_folder


def run_bulk_bootstrap_plots(
    simulated_daily_bmus: xr.Dataset,
    point_lat: float,
    point_lon: float,
    project_root: str | None = None,
    variables: Sequence[str] = ("hs", "tp", "dp"),
    sim_idx: int = 0,
    seed: int = 0,
    bulk_files: dict | None = None,
    historical_folder: str | None = None,
    historical_filename_template: str = "{var}_NorthCarolina.nc",
    save_subfolder: str = "outputs/Figures/ALR_bulk_bootstrap_month",
    figure_suffix: str | None = None,
    monthly_conditioning: bool = True,
    plot_non_overlap_timeseries: bool = False,
    original_band: str | None = "std",
    band_color: str = "0.7",
    band_alpha: float = 0.25,
):
    """
    Bootstrap-and-plot bulk historical vs simulated daily series at the nearest
    grid point to (``point_lat``, ``point_lon``).

    Prefer ``plot_from_folder`` when bootstrap NetCDFs already exist: that path
    reads the saved series instead of re-sampling, so figures match the final files.

    Historical-data location:
    - ``historical_folder`` selects the folder containing the historical NetCDF
      files. If ``None`` (default), falls back to ``outputs/merged_500m_binwaves_bmus``.
      Relative paths are resolved against ``project_root`` / cwd (``..`` allowed);
      absolute paths are used as-is. Examples:
        * ``historical_folder=None`` → ``outputs/merged_500m_binwaves_bmus``
        * ``historical_folder="outputs/merged_500m_binwaves_bmus"`` → same
        * ``historical_folder="/abs/path/to/data"`` → arbitrary absolute path
    - ``historical_filename_template`` builds the per-variable filename when not
      explicitly given via ``bulk_files``. ``{var}`` is replaced by each variable
      name in ``variables``. Defaults to ``"{var}_NorthCarolina.nc"``. For the
      merged 500 m product use ``"{var}_500m.nc"``.
    - ``bulk_files`` (optional ``dict``) overrides individual filenames per
      variable; entries here take precedence over the template.
    """
    fname_for = lambda var: historical_filename_template.format(var=var)

    if bulk_files is None:
        bulk_files = {var: fname_for(var) for var in ("hs", "tp", "dp")}
    else:
        bulk_files = dict(bulk_files)
    for var in variables:
        bulk_files.setdefault(var, fname_for(var))

    if historical_folder is None:
        historical_folder = "outputs/merged_500m_binwaves_bmus"
    project_root, bulk_folder = _resolve_existing_folder(project_root, historical_folder)
    save_folder = _ensure_folder(os.path.join(project_root, _with_suffix(save_subfolder, figure_suffix)))
    rng = np.random.default_rng(seed)

    bmus_da = simulated_daily_bmus["evbmus_sims"].isel(n_sim=sim_idx).astype(int)
    if "time" not in bmus_da.dims:
        raise ValueError("simulated_daily_bmus['evbmus_sims'] must include 'time' dimension.")
    bmus_da = _ensure_datetime64_time(bmus_da)
    bmus_da = bmus_da.rename("bmu")

    daily_pairs = {}
    loaded_variables: list[str] = []
    for var in variables:
        if var not in bulk_files:
            print(f"Warning: no bulk file mapping for '{var}'. Skipping.")
            continue
        path = os.path.join(bulk_folder, bulk_files[var])
        if not os.path.exists(path):
            print(f"Warning: bulk file not found for '{var}': {path}. Skipping.")
            continue
        ds = xr.open_dataset(path)
        nc_var = _resolve_bulk_nc_var_name(var, ds)
        if nc_var is None:
            print(
                f"Warning: no variable for '{var}' (family '{_split_var_family(var)[0]}') in '{path}'. "
                f"Available vars={list(ds.data_vars)}. Skipping."
            )
            ds.close()
            continue
        if nc_var != var:
            print(f"Note: reading NetCDF variable '{nc_var}' for logical variable '{var}' ({path}).")
        lats = ds["lat"].values
        lons = ds["lon"].values
        dists_km = haversine_km(point_lat, point_lon, lats, lons)
        site_idx = int(np.argmin(dists_km))
        nearest_dist_km = float(dists_km[site_idx])
        nearest_lat = float(lats[site_idx])
        nearest_lon = float(lons[site_idx])
        print(
            f"{var.upper()}: target ({point_lat:.4f}, {point_lon:.4f}) -> "
            f"nearest site {site_idx} ({nearest_lat:.4f}, {nearest_lon:.4f}), "
            f"distance={nearest_dist_km:.3f} km"
        )
        bulk_daily_full = _daily_agg(ds[nc_var].isel(site=site_idx), name_hint=var)
        ds.close()
        bulk_daily_full = _ensure_datetime64_time(bulk_daily_full)
        bmus_al, bulk_daily = xr.align(bmus_da, bulk_daily_full, join="inner")
        if bmus_al.sizes.get("time", 0) == 0:
            raise ValueError(
                f"No common time overlap between simulated BMUs and bulk series for variable '{var}'."
            )
        cluster_ids = np.asarray(bmus_al.values).astype(int) - 1
        values = np.asarray(bulk_daily.values)
        months = pd.to_datetime(bulk_daily["time"].values).month

        cluster_month_to_vals = {}
        cluster_to_vals = {}
        for cid in np.unique(cluster_ids):
            cid_mask = cluster_ids == cid
            pool_all = values[cid_mask]
            pool_all = pool_all[np.isfinite(pool_all)]
            if pool_all.size > 0:
                cluster_to_vals[int(cid)] = pool_all
            if monthly_conditioning:
                for m in range(1, 13):
                    mask = cid_mask & (months == m)
                    pool = values[mask]
                    pool = pool[np.isfinite(pool)]
                    if pool.size > 0:
                        cluster_month_to_vals[(int(cid), int(m))] = pool

        ts_boot = np.full(cluster_ids.shape, np.nan, dtype=float)
        for i, cid in enumerate(cluster_ids):
            pool = None
            if monthly_conditioning:
                m = int(months[i])
                pool = cluster_month_to_vals.get((int(cid), m))
            if pool is None:
                pool = cluster_to_vals.get(int(cid))
            if pool is not None and len(pool) > 0:
                ts_boot[i] = rng.choice(pool)

        emu = xr.DataArray(ts_boot, coords={"time": bulk_daily["time"]}, dims=["time"], name=f"{var}_boot")
        daily_pairs[var] = (emu, bulk_daily)
        loaded_variables.append(var)

        emu_plot = emu
        hist_plot = bulk_daily
        if plot_non_overlap_timeseries:
            sim_time = pd.to_datetime(bmus_da["time"].values)
            sim_months = sim_time.month
            cluster_ids_full = np.asarray(bmus_da.values).astype(int) - 1
            ts_boot_full = np.full(cluster_ids_full.shape, np.nan, dtype=float)
            for i, cid in enumerate(cluster_ids_full):
                pool = None
                if monthly_conditioning:
                    pool = cluster_month_to_vals.get((int(cid), int(sim_months[i])))
                if pool is None:
                    pool = cluster_to_vals.get(int(cid))
                if pool is not None and len(pool) > 0:
                    ts_boot_full[i] = rng.choice(pool)

            emu_full = xr.DataArray(
                ts_boot_full,
                coords={"time": bmus_da["time"]},
                dims=["time"],
                name=f"{var}_boot_full",
            )
            emu_plot, hist_plot = xr.align(emu_full, bulk_daily_full, join="outer")

        fig_ts, ax = plt.subplots(1, 1, figsize=(14, 4), sharex=True)
        ax.plot(hist_plot["time"].values, hist_plot.values, lw=1.5, color="k", label=f"Historical {var}")
        ax.plot(emu_plot["time"].values, emu_plot.values, lw=1.2, color="fuchsia", alpha=0.8, label=f"Simulated {var}")
        ax.set_ylabel(var.upper())
        ax.set_xlabel("Time")
        ax.set_title(f"{var.upper()} - Historical vs Simulated")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        if plot_non_overlap_timeseries:
            # Show the full union range when requested (including non-overlap segments).
            tvals = pd.to_datetime(hist_plot["time"].values)
            if tvals.size > 0:
                ax.set_xlim(tvals.min(), tvals.max())
        else:
            ax.set_xlim(pd.to_datetime("1980-01-01"), pd.to_datetime("2024-12-31"))
        out = os.path.join(
            save_folder, f"Daily_mean_{var}_Historical_vs_Simulated{'' if not figure_suffix else '_' + figure_suffix}.png"
        )
        fig_ts.savefig(out, dpi=150, bbox_inches="tight")
        print("Saved:", out)
        plt.show()

    if not loaded_variables:
        raise ValueError("No variables were loaded. Check 'variables' and bulk files.")

    _render_bulk_daily_pair_figures(
        daily_pairs,
        loaded_variables,
        save_folder,
        figure_suffix=figure_suffix,
        original_band=original_band,
        band_color=band_color,
        band_alpha=band_alpha,
    )
    return daily_pairs, save_folder


