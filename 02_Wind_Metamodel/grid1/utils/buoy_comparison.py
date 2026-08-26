"""Buoy vs hindcast bulk validation (Hs, Tp, Dir) for grid1 BinWaves_BMUS outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_BUOY_DATA_DIR = Path("/nfs/home/geocean/montanoj/ShoreShop2026/inputs/buoy_data")
_BUOY_TP_COLUMNS = ("Tp_Buoy", "TP_Buoy", "Tp", "PeakPeriod_Buoy", "Tm_Buoy")
BUOY_COLOR = "black"
HINDCAST_COLOR = "fuchsia"


@dataclass(frozen=True)
class SiteSelection:
    ids: list[str]
    lats: np.ndarray
    lons: np.ndarray
    src_index: np.ndarray
    distance_m: np.ndarray


@dataclass(frozen=True)
class BuoyCatalogEntry:
    lat: float
    lon: float


@dataclass(frozen=True)
class BuoyHistoricalMatch:
    buoy_id: str
    buoy_lat: float
    buoy_lon: float
    coordinate: tuple[float, float]
    distance_km: float
    match_source: Literal["historical_grid"]
    site_index: int | None = None


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def _haversine_m(lat1, lon1, lat2, lon2):
    return _haversine_km(lat1, lon1, lat2, lon2) * 1000.0


def _lat_lon_from_pair(pair: Sequence[float]) -> tuple[float, float]:
    a, b = float(pair[0]), float(pair[1])
    if abs(a) > abs(b):
        lon, lat = a, b
    else:
        lat, lon = a, b
    return lat, lon


def parse_buoy_entry(value: Any) -> BuoyCatalogEntry:
    if isinstance(value, Mapping):
        if "coord" in value:
            return BuoyCatalogEntry(*_lat_lon_from_pair(value["coord"]))
        if "lat" in value and "lon" in value:
            return BuoyCatalogEntry(float(value["lat"]), float(value["lon"]))
    return BuoyCatalogEntry(*_lat_lon_from_pair(value))


def parse_buoy_catalog(buoys: Mapping[str, Any]) -> dict[str, BuoyCatalogEntry]:
    return {str(bid): parse_buoy_entry(entry) for bid, entry in buoys.items()}


def _resolve_buoy_location(
    buoy_id: str,
    buoys: Mapping[str, Any],
    *,
    buoy_data_dir: Path,
) -> tuple[str, BuoyCatalogEntry]:
    catalog = parse_buoy_catalog(buoys)
    key = str(buoy_id).lower()
    lookup = {k.lower(): (k, v) for k, v in catalog.items()}
    if key not in lookup:
        raise KeyError(
            f"Unknown buoy {buoy_id!r}. Add it to buoys_geo as {{id: (lon, lat)}}."
        )
    return lookup[key]


def _buoy_pkl_path(buoy_id: str, buoy_data_dir: Path) -> Path | None:
    for key in (str(buoy_id), str(buoy_id).lower()):
        path = buoy_data_dir / f"buoy_{key}_bulk_parameters.pkl"
        if path.is_file():
            return path
    return None


def load_buoy_bulk_dataframe(buoy_id: str, buoy_data_dir: Path) -> pd.DataFrame | None:
    pkl_path = _buoy_pkl_path(buoy_id, buoy_data_dir)
    if pkl_path is None:
        print(f"SKIP buoy {buoy_id}: no pickle in {buoy_data_dir}")
        return None

    df = pd.read_pickle(pkl_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        elif "time" in df.columns:
            df = df.set_index("time")
        else:
            print(f"SKIP buoy {buoy_id}: no datetime index in {pkl_path.name}")
            return None
    df = df.sort_index()
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index), name=None)
    return df


def get_buoy_observation_period(
    buoy_id: str,
    buoy_data_dir: Path,
) -> tuple[pd.Timestamp, pd.Timestamp] | tuple[None, None]:
    df = load_buoy_bulk_dataframe(buoy_id, buoy_data_dir)
    if df is None:
        return None, None
    for col in ("Hs_Buoy", "Dir_Buoy"):
        if col not in df.columns:
            print(f"SKIP buoy {buoy_id}: missing column {col}")
            return None, None

    valid = (
        df["Hs_Buoy"].notna()
        & df["Dir_Buoy"].notna()
        & (df["Hs_Buoy"] > 0)
        & (df["Dir_Buoy"] >= 0)
        & (df["Dir_Buoy"] <= 360)
    )
    if not bool(valid.any()):
        print(f"SKIP buoy {buoy_id}: no valid Hs/Dir observations")
        return None, None

    times = pd.to_datetime(df.index[valid])
    return pd.Timestamp(times.min()), pd.Timestamp(times.max())


def _series_to_dataarray(series: pd.Series, *, name: str) -> xr.DataArray:
    series = series.sort_index()
    time_coord = pd.DatetimeIndex(pd.to_datetime(series.index), name=None)
    return xr.DataArray(
        series.astype(float).values,
        coords=[("time", time_coord)],
        dims="time",
        name=name,
    )


def _is_directional_name(name: str) -> bool:
    n = str(name).lower()
    return n.startswith(("dp", "dm", "spr", "pdp", "pdm", "pspr", "dir"))


def _circular_mean_deg_np(a: np.ndarray, axis: int) -> np.ndarray:
    rad = np.deg2rad(a)
    sinm = np.nanmean(np.sin(rad), axis=axis)
    cosm = np.nanmean(np.cos(rad), axis=axis)
    ang = np.rad2deg(np.arctan2(sinm, cosm))
    return (ang + 360.0) % 360.0


def _daily_agg(da: xr.DataArray, *, name_hint: str) -> xr.DataArray:
    if _is_directional_name(name_hint):
        return da.resample(time="1D").reduce(_circular_mean_deg_np, dim="time")
    return da.resample(time="1D").mean()


def apply_aggregation(da: xr.DataArray, variable: str, aggregation: str) -> xr.DataArray:
    aggregation = str(aggregation).lower()
    if aggregation in ("native", "hourly"):
        return da
    if aggregation == "daily":
        return _daily_agg(da, name_hint=variable)
    if aggregation == "monthly":
        daily = _daily_agg(da, name_hint=variable)
        if _is_directional_name(variable):
            return daily.resample(time="MS").reduce(_circular_mean_deg_np, dim="time")
        return daily.resample(time="MS").mean()
    raise ValueError(
        f"aggregation must be 'hourly', 'native', 'daily', or 'monthly'; got {aggregation!r}"
    )


def _guess_lat_lon(ds: xr.Dataset) -> tuple[str, str]:
    for lat_name in ("lat", "latitude"):
        if lat_name in ds.coords or lat_name in ds:
            for lon_name in ("lon", "longitude"):
                if lon_name in ds.coords or lon_name in ds:
                    return lat_name, lon_name
    raise KeyError(f"Could not find lat/lon in dataset. coords={list(ds.coords)}")


def _guess_site_dim(ds: xr.Dataset, lat_name: str, lon_name: str) -> str:
    lat = ds[lat_name] if lat_name in ds else ds.coords[lat_name]
    lon = ds[lon_name] if lon_name in ds else ds.coords[lon_name]
    if lat.ndim == 1 and lon.ndim == 1 and lat.dims == lon.dims:
        return lat.dims[0]
    for cand in ("site", "seapoint", "point", "stations"):
        if cand in ds.dims:
            return cand
    raise ValueError("Could not infer point/site dimension")


def select_nearest_site(nc_path: Path, coordinate: tuple[float, float]) -> SiteSelection:
    with xr.open_dataset(nc_path) as ds:
        lat_name, lon_name = _guess_lat_lon(ds)
        _ = _guess_site_dim(ds, lat_name, lon_name)
        src_lat = (ds[lat_name] if lat_name in ds else ds.coords[lat_name]).values.astype(float)
        src_lon = (ds[lon_name] if lon_name in ds else ds.coords[lon_name]).values.astype(float)

    if src_lat.ndim != 1 or src_lon.ndim != 1 or src_lat.shape != src_lon.shape:
        raise ValueError(f"Expected 1-D lat/lon in {nc_path}")

    sel_lat, sel_lon = float(coordinate[0]), float(coordinate[1])
    d = _haversine_m(sel_lat, sel_lon, src_lat, src_lon)
    i_pick = int(np.argmin(d))
    return SiteSelection(
        ids=[str(i_pick)],
        lats=np.asarray([src_lat[i_pick]], dtype=float),
        lons=np.asarray([src_lon[i_pick]], dtype=float),
        src_index=np.asarray([i_pick], dtype=int),
        distance_m=np.asarray([float(d[i_pick])], dtype=float),
    )


def resolve_hindcast_nc(hindcast_folder: Path, variable: str, *, grid_id: int = 1) -> Path:
    folder = Path(hindcast_folder)
    gid = f"grid{grid_id}"
    candidates = (
        folder / f"{variable}_{gid}_BinWaves_BMUS.nc",
        folder / f"{variable}_merged_all.nc",
        folder / f"{variable}_500m.nc",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No NetCDF for {variable!r} in {folder} (tried {[p.name for p in candidates]})"
    )


def load_timeseries_at_coordinate(
    nc_path: Path,
    coordinate: tuple[float, float],
    *,
    variable: str,
    aggregation: str = "hourly",
    time_start=None,
    time_end=None,
) -> tuple[xr.DataArray, dict]:
    selection = select_nearest_site(nc_path, coordinate)
    site_idx = int(selection.src_index[0])

    with xr.open_dataset(nc_path) as ds:
        lat_name, lon_name = _guess_lat_lon(ds)
        site_dim = _guess_site_dim(ds, lat_name, lon_name)
        nc_var = variable if variable in ds.data_vars else variable
        if nc_var not in ds.data_vars:
            for name in ds.data_vars:
                if str(name).lower() == variable.lower():
                    nc_var = str(name)
                    break
        da = ds[nc_var].isel({site_dim: site_idx})
        da = apply_aggregation(da, variable, aggregation)
        if time_start is not None or time_end is not None:
            da = da.sel(time=slice(time_start, time_end))

    info = {
        "site_index": site_idx,
        "lat": float(selection.lats[0]),
        "lon": float(selection.lons[0]),
        "distance_km": float(selection.distance_m[0] / 1000.0),
        "path": str(nc_path),
    }
    return da, info


def match_buoy_to_hindcast(
    buoy_id: str,
    *,
    buoys: Mapping[str, Any],
    hindcast_folder: Path,
    grid_id: int = 1,
    buoy_data_dir: Path = DEFAULT_BUOY_DATA_DIR,
) -> BuoyHistoricalMatch:
    bid, entry = _resolve_buoy_location(buoy_id, buoys, buoy_data_dir=buoy_data_dir)
    hs_nc = resolve_hindcast_nc(hindcast_folder, "hs", grid_id=grid_id)
    selection = select_nearest_site(hs_nc, (entry.lat, entry.lon))
    coord = (float(selection.lats[0]), float(selection.lons[0]))
    dist_km = float(selection.distance_m[0] / 1000.0)
    return BuoyHistoricalMatch(
        buoy_id=bid,
        buoy_lat=entry.lat,
        buoy_lon=entry.lon,
        coordinate=coord,
        distance_km=dist_km,
        match_source="historical_grid",
        site_index=int(selection.src_index[0]),
    )


def load_buoy_hs_tp_dp(
    buoy_id: str,
    *,
    buoy_data_dir: Path,
    aggregation: str = "hourly",
    time_start=None,
    time_end=None,
) -> tuple[dict[str, xr.DataArray | None], dict]:
    df = load_buoy_bulk_dataframe(buoy_id, buoy_data_dir)
    if df is None:
        return {}, {}

    pkl_path = _buoy_pkl_path(buoy_id, buoy_data_dir)
    if "Hs_Buoy" not in df.columns or "Dir_Buoy" not in df.columns:
        print(f"SKIP buoy {buoy_id}: missing Hs_Buoy/Dir_Buoy in {pkl_path.name}")
        return {}, {}

    hs = _series_to_dataarray(df["Hs_Buoy"], name="hs")
    dp = _series_to_dataarray(df["Dir_Buoy"], name="dp")
    if time_start is not None or time_end is not None:
        hs = hs.sel(time=slice(time_start, time_end))
        dp = dp.sel(time=slice(time_start, time_end))

    hs = apply_aggregation(hs, "hs", aggregation)
    dp = apply_aggregation(dp, "dp", aggregation)

    tp_da: xr.DataArray | None = None
    tp_col = next((c for c in _BUOY_TP_COLUMNS if c in df.columns), None)
    if tp_col is not None:
        tp = _series_to_dataarray(df[tp_col], name="tp")
        if time_start is not None or time_end is not None:
            tp = tp.sel(time=slice(time_start, time_end))
        tp_da = apply_aggregation(tp, "tp", aggregation)

    hs_a, dp_a = xr.align(hs, dp, join="inner")
    if hs_a.sizes.get("time", 0) == 0:
        print(f"SKIP buoy {buoy_id}: no overlapping Hs/Dir after aggregation")
        return {}, {}

    if tp_da is not None:
        hs_a, dp_a, tp_a = xr.align(hs_a, dp_a, tp_da, join="inner")
        if hs_a.sizes.get("time", 0) == 0:
            print(f"SKIP buoy {buoy_id}: no overlapping Hs/Dir/Tp after aggregation")
            return {}, {}
        tp_da = tp_a

    info = {
        "buoy_id": str(buoy_id),
        "path": str(pkl_path) if pkl_path else "",
        "aggregation": str(aggregation).lower(),
        "n_points": int(hs_a.sizes.get("time", hs_a.size)),
        "tp_column": tp_col,
    }
    return {"hs": hs_a, "tp": tp_da, "dp": dp_a}, info


def _align_buoy_hindcast_series(
    buoy_bulk: Mapping[str, xr.DataArray | None],
    hindcast_bulk: Mapping[str, xr.DataArray],
) -> dict[str, xr.DataArray | None]:
    arrays: list[xr.DataArray] = [buoy_bulk["hs"], hindcast_bulk["hs"], buoy_bulk["dp"], hindcast_bulk["dp"]]  # type: ignore[index]
    keys = ["hs_b", "hs_m", "dp_b", "dp_m"]
    if buoy_bulk.get("tp") is not None and "tp" in hindcast_bulk:
        arrays.extend([buoy_bulk["tp"], hindcast_bulk["tp"]])  # type: ignore[list-item]
        keys.extend(["tp_b", "tp_m"])
    aligned = xr.align(*arrays, join="inner")
    return dict(zip(keys, aligned))


def validation_dataframe(aligned: Mapping[str, xr.DataArray | None]) -> pd.DataFrame:
    hs_b = aligned["hs_b"]
    assert hs_b is not None
    n = int(hs_b.sizes.get("time", 0))
    nan_col = np.full(n, np.nan, dtype=float)
    tp_b = aligned.get("tp_b")
    tp_m = aligned.get("tp_m")
    df = pd.DataFrame(
        {
            "Hs_Buoy": np.asarray(hs_b.values, dtype=float),
            "Hs_Hindcast": np.asarray(aligned["hs_m"].values, dtype=float),
            "Tp_Buoy": np.asarray(tp_b.values, dtype=float) if tp_b is not None else nan_col,
            "Tp_Hindcast": np.asarray(tp_m.values, dtype=float) if tp_m is not None else nan_col,
            "Dir_Buoy": np.asarray(aligned["dp_b"].values, dtype=float),
            "Dir_Hindcast": np.asarray(aligned["dp_m"].values, dtype=float),
        },
        index=pd.to_datetime(hs_b["time"].values),
    )
    return df[
        ["Hs_Buoy", "Hs_Hindcast", "Tp_Buoy", "Tp_Hindcast", "Dir_Buoy", "Dir_Hindcast"]
    ]


def _point_density_from_hist2d(x: np.ndarray, y: np.ndarray, bins: int = 65) -> np.ndarray:
    if x.size == 0:
        return np.array([])
    h, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    x_idx = np.clip(np.digitize(x, x_edges) - 1, 0, h.shape[0] - 1)
    y_idx = np.clip(np.digitize(y, y_edges) - 1, 0, h.shape[1] - 1)
    return h[x_idx, y_idx]


def _validation_metrics_text(obs: np.ndarray, model: np.ndarray) -> str:
    finite = np.isfinite(obs) & np.isfinite(model)
    if finite.sum() < 2:
        return "MAE: nan\nRMSE: nan\nR²: nan\nSI: nan\nBias: nan"
    obs_c = obs[finite]
    mod_c = model[finite]
    mae = float(np.mean(np.abs(obs_c - mod_c)))
    rmse = float(np.sqrt(np.mean((obs_c - mod_c) ** 2)))
    r2 = float(np.corrcoef(obs_c, mod_c)[0, 1] ** 2)
    bias = float(np.mean(mod_c - obs_c))
    si = rmse / float(np.mean(obs_c)) if np.mean(obs_c) != 0 else float("nan")
    return (
        f"MAE: {mae:.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"R²: {r2:.2f}\n"
        f"SI: {si:.2f}\n"
        f"Bias: {bias:.2f}"
    )


def _plot_validation_scatter_panel(
    ax,
    obs: np.ndarray,
    model: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    lim: tuple[float, float],
    scatter_cmap: str = "plasma",
    scatter_alpha: float = 0.6,
) -> None:
    valid = np.isfinite(obs) & np.isfinite(model)
    ox = obs[valid]
    oy = model[valid]
    if ox.size == 0:
        ax.text(0.5, 0.5, "no overlap", transform=ax.transAxes, ha="center", color="white")
        ax.set_xlabel(xlabel, color="white")
        ax.set_ylabel(ylabel, color="white")
        return

    density = _point_density_from_hist2d(ox, oy)
    order = np.argsort(density)
    ax.scatter(
        ox[order],
        oy[order],
        c=density[order],
        cmap=scatter_cmap,
        s=1,
        alpha=scatter_alpha,
        edgecolors="none",
    )

    lo, hi = lim
    ax.plot([lo, hi], [lo, hi], c="white", linestyle="--", lw=1)
    ax.text(
        0.95,
        0.05,
        _validation_metrics_text(obs, model),
        transform=ax.transAxes,
        color="white",
        fontsize=10,
        va="bottom",
        ha="right",
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel, color="white")
    ax.set_ylabel(ylabel, color="white")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("white")
    ax.spines["bottom"].set_color("white")
    ax.set_facecolor((0, 0, 0, 0))
    ax.patch.set_alpha(0.0)


def _plot_timeseries_on_ax(
    ax,
    times,
    values,
    *,
    directional: bool,
    label: str | None = None,
    **style,
) -> None:
    plot_kw = dict(style)
    if label is not None:
        plot_kw["label"] = label
    if directional:
        plot_kw = {k: v for k, v in plot_kw.items() if k not in ("lw", "ls")}
        ax.plot(
            times,
            values,
            linestyle="none",
            marker=".",
            markersize=1.8,
            **plot_kw,
        )
    else:
        ax.plot(times, values, **plot_kw)


def _plot_buoy_bulk_validation_timeseries(
    aligned: Mapping[str, xr.DataArray | None],
    *,
    model_label: str,
) -> plt.Figure:
    """Three-row Hs/Tp/Dir time series: buoy vs hindcast (Dir as dots only)."""
    panel_cfg = (
        ("hs", "hs_b", "hs_m", "Hs [m]"),
        ("tp", "tp_b", "tp_m", "Tp [s]"),
        ("dp", "dp_b", "dp_m", "Dir [°]"),
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for ax, (var, obs_key, mod_key, ylab) in zip(axes, panel_cfg):
        obs = aligned.get(obs_key)
        mod = aligned.get(mod_key)
        if obs is None or mod is None:
            ax.text(
                0.5,
                0.5,
                f"no {var.upper()} data",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_ylabel(ylab)
            ax.grid(True, alpha=0.3)
            continue

        times = obs["time"].values
        directional = _is_directional_name(var)
        _plot_timeseries_on_ax(
            ax,
            times,
            obs.values,
            directional=directional,
            color=BUOY_COLOR,
            lw=1.0,
            alpha=0.9,
            label="Buoy",
        )
        _plot_timeseries_on_ax(
            ax,
            times,
            mod.values,
            directional=directional,
            color=HINDCAST_COLOR,
            lw=1.0,
            alpha=0.85,
            label=model_label,
        )
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("time")
    fig.tight_layout()
    return fig


def plot_buoy_bulk_validation_scatter(
    buoy_ids: str | Sequence[str],
    *,
    buoys: Mapping[str, Any],
    hindcast_folder: str | Path,
    grid_id: int = 1,
    buoy_data_dir: str | Path = DEFAULT_BUOY_DATA_DIR,
    use_buoy_time_range: bool = True,
    time_start=None,
    time_end=None,
    aggregation: str = "hourly",
    model_label: str = "BinWaves + BMUS",
    output_folder: str | Path | None = None,
    csv_output_folder: str | Path | None = None,
    save: bool = True,
    save_timeseries: bool = True,
    show: bool = False,
    dpi: int = 150,
    scatter_cmap: str = "plasma",
    scatter_alpha: float = 0.6,
    max_match_distance_km: float | None = None,
) -> list[BuoyHistoricalMatch]:
    """
    Buoy vs hindcast bulk validation scatter plots (Hs, Tp, Dir).

    Writes one 1×3 scatter PNG per buoy, optional time-series PNG, and CSV.
    """
    ids = [buoy_ids] if isinstance(buoy_ids, str) else [str(b) for b in buoy_ids]
    if not ids:
        raise ValueError("buoy_ids must contain at least one buoy ID")

    hindcast_folder = Path(hindcast_folder)
    buoy_data_dir = Path(buoy_data_dir)

    out_dir: Path | None = None
    if save or save_timeseries:
        if output_folder is None:
            raise ValueError("output_folder is required when save=True or save_timeseries=True")
        out_dir = Path(output_folder)
        out_dir.mkdir(parents=True, exist_ok=True)

    csv_dir: Path | None = None
    if csv_output_folder is not None:
        csv_dir = Path(csv_output_folder)
        csv_dir.mkdir(parents=True, exist_ok=True)
    elif save and out_dir is not None:
        csv_dir = out_dir

    print(f"Buoy bulk validation — hindcast: {hindcast_folder} | model: {model_label}")

    results: list[BuoyHistoricalMatch] = []
    panel_cfg = (
        ("hs", f"Hs - Buoy [m]", f"Hs - {model_label} [m]", (0.0, 6.0)),
        ("tp", f"Tp - Buoy [s]", f"Tp - {model_label} [s]", (0.0, 25.0)),
        ("dp", f"Dir - Buoy [°]", f"Dir - {model_label} [°]", (0.0, 350.0)),
    )

    for bid in ids:
        try:
            match = match_buoy_to_hindcast(
                bid,
                buoys=buoys,
                hindcast_folder=hindcast_folder,
                grid_id=grid_id,
                buoy_data_dir=buoy_data_dir,
            )
        except KeyError as exc:
            print(f"SKIP buoy {bid}: {exc}")
            continue
        results.append(match)

        if max_match_distance_km is not None and match.distance_km > max_match_distance_km:
            print(
                f"SKIP buoy {bid}: nearest site is {match.distance_km:.1f} km away "
                f"(max {max_match_distance_km:.1f} km)"
            )
            continue

        if use_buoy_time_range:
            t0, t1 = get_buoy_observation_period(bid, buoy_data_dir)
            if t0 is None:
                print(f"SKIP buoy {bid}: no valid buoy observations")
                continue
            period_start, period_end = t0, t1
        else:
            if time_start is None or time_end is None:
                raise ValueError("time_start and time_end required when use_buoy_time_range=False")
            period_start = pd.Timestamp(time_start)
            period_end = pd.Timestamp(time_end)

        buoy_bulk, _ = load_buoy_hs_tp_dp(
            bid,
            buoy_data_dir=buoy_data_dir,
            aggregation=aggregation,
            time_start=period_start,
            time_end=period_end,
        )
        if not buoy_bulk:
            print(f"SKIP buoy {bid}: could not load buoy bulk parameters")
            continue

        lat, lon = match.coordinate
        hindcast_bulk: dict[str, xr.DataArray] = {}
        for key in ("hs", "tp", "dp"):
            nc_path = resolve_hindcast_nc(hindcast_folder, key, grid_id=grid_id)
            da, _ = load_timeseries_at_coordinate(
                nc_path,
                (lat, lon),
                variable=key,
                aggregation=aggregation,
                time_start=period_start,
                time_end=period_end,
            )
            hindcast_bulk[key] = da

        aligned = _align_buoy_hindcast_series(buoy_bulk, hindcast_bulk)
        hs_b = aligned["hs_b"]
        if hs_b is None or hs_b.sizes.get("time", 0) == 0:
            print(f"SKIP buoy {bid}: no overlapping Hs samples")
            continue

        if csv_dir is not None:
            csv_path = csv_dir / f"buoy_validation_{bid}.csv"
            validation_dataframe(aligned).to_csv(csv_path)
            print(f"Saved: {csv_path}")

        if save_timeseries and out_dir is not None:
            ts_fig = _plot_buoy_bulk_validation_timeseries(
                aligned,
                model_label=model_label,
            )
            ts_path = out_dir / f"validation_timeseries_{bid}.png"
            ts_fig.savefig(ts_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved: {ts_path}")
            if show:
                plt.show()
            else:
                plt.close(ts_fig)

        if not save:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor("black")
        fig.patch.set_alpha(0.0)

        obs_map = {
            "hs": np.asarray(aligned["hs_b"].values),
            "tp": np.asarray(aligned["tp_b"].values) if aligned.get("tp_b") is not None else None,
            "dp": np.asarray(aligned["dp_b"].values),
        }
        mod_map = {
            "hs": np.asarray(aligned["hs_m"].values),
            "tp": np.asarray(aligned["tp_m"].values) if aligned.get("tp_m") is not None else None,
            "dp": np.asarray(aligned["dp_m"].values),
        }

        for ax, (key, xlab, ylab, lim) in zip(axes, panel_cfg):
            obs = obs_map.get(key)
            mod = mod_map.get(key)
            if obs is None or mod is None:
                ax.text(0.5, 0.5, f"no {key.upper()} data", transform=ax.transAxes, ha="center", color="white")
                ax.set_facecolor((0, 0, 0, 0))
                continue
            _plot_validation_scatter_panel(
                ax, obs, mod,
                xlabel=xlab, ylabel=ylab, lim=lim,
                scatter_cmap=scatter_cmap, scatter_alpha=scatter_alpha,
            )

        fig.tight_layout()

        if out_dir is not None:
            out_path = out_dir / f"validation_scatter_{bid}.png"
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="black")
            print(f"Saved: {out_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return results
