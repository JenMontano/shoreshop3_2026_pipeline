"""
GCM wave output comparison utilities (merged_500m / GCM scenario folders).

Used by ``GCMs_comparison.ipynb`` — call the high-level plot functions with
explicit parameters so each notebook cell can use a different configuration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from utils.alr_plotting import (
    _circular_mean_deg_np,
    _daily_agg,
    _is_directional_name,
    _point_density_from_hist2d,
    _resolve_bulk_nc_var_name,
)
from utils.bmu_bootstrap_timeseries import (
    _guess_lat_lon,
    _guess_site_dim,
    select_all_sites_from_merged_file,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    make_subplots = None  # type: ignore[misc, assignment]

# Default catalogue of runs (toggle with ``enabled`` or pass ``models=[...]``)
DEFAULT_SCENARIOS: dict[str, dict] = {
    "historical": {
        "folder": "outputs/merged_500m",
        "enabled": True,
        "color": "0.35",
        "lw": 0.8,
        "ls": "-",
        "zorder": 1,
        "alpha": 0.85,
    },
    "ACCESS SSP2-4.5": {"folder": "outputs/access_245", "enabled": True},
    "ACCESS SSP5-4.5": {"folder": "outputs/access_545", "enabled": True},
    "E3SM SSP2-4.5": {"folder": "outputs/earth3_veg_ssp245", "enabled": True},
    "E3SM SSP5-8.5": {"folder": "outputs/earth3_veg_ssp585", "enabled": True},
    "MIROC6 SSP2-4.5": {"folder": "outputs/miroc6_ssp245", "enabled": True},
    "MIROC6 SSP5-8.5": {"folder": "outputs/miroc6_ssp585", "enabled": True},
    "MPI-ESM1-2 SSP2-4.5": {"folder": "outputs/mpi_esm1_245", "enabled": True},
    "Century": {"folder": "outputs/century", "enabled": True},
}

DEFAULT_HISTORICAL_FOLDER = "outputs/merged_500m"
DEFAULT_SCATTER_VARIABLES = ("hs", "tp", "dp")

# Default line styling when not overridden per scenario
DEFAULT_MODEL_ALPHA = 0.72
DEFAULT_MODEL_LW = 1.4
DEFAULT_MODEL_ZORDER = 10

# CERC-like longshore transport: Q = (H**5/2) * sin(2*alpha) * K  (matches ALR notebooks)
DEFAULT_LONGSHORE_K = 0.39

# Natural Earth coastline clip for North Carolina (lon_min, lat_min, lon_max, lat_max)
NC_COAST_BBOX = (-79.5, 33.0, -74.0, 37.0)
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MERGED_GRIDS_FOLDER = str(_REPO_ROOT / "01_BinWaves" / "outputs" / "merged_grids")
DEFAULT_HISTORICAL_PARTITIONS_FOLDER = DEFAULT_MERGED_GRIDS_FOLDER
DEFAULT_BUOY_DATA_DIR = str(_REPO_ROOT / "01_BinWaves" / "inputs" / "buoy_data")
DEFAULT_SWAN_TIMESERIES_FOLDER = "outputs/wind_simulations"
SWAN_BULK_LABEL = "swan bulk"
_SWAN_BULK_NC_VARS = {"hs": "Hsig", "tp": "Tps", "dp": "pdir"}
BINWAVES_OVERLAY_COLOR = "fuchsia"
BINWAVES_BULK_LABEL = "BinWaves"
SWAN_WIND_OVERLAY_COLOR = "turquoise"
BUOY_OVERLAY_COLOR = "k"
BINWAVES_OVERLAY_COLOR_PLOTLY = "rgb(255,0,255)"
SWAN_WIND_OVERLAY_COLOR_PLOTLY = "rgb(64,224,208)"
BUOY_OVERLAY_COLOR_PLOTLY = "rgb(0,0,0)"

# Preset historical wave folders (auto-picks *_500m.nc vs *_merged_all.nc per folder).
HistoricalDataset = Literal["merged_500m", "merged_grids"]
HISTORICAL_DATASET_FOLDERS: dict[HistoricalDataset, str] = {
    "merged_500m": DEFAULT_HISTORICAL_FOLDER,
    "merged_grids": DEFAULT_MERGED_GRIDS_FOLDER,
}

# NDBC / NC buoy locations as ``{id: (lon, lat)}`` (converted internally to lat/lon).
# Matches buoy_data pickles under 01_BinWaves/inputs/buoy_data.
DEFAULT_NC_BUOYS: dict[str, tuple[float, float]] = {
    "44088": (-74.839, 36.612),
    "44014": (-74.837, 36.603),
    "44100": (-75.593, 36.258),
    "44079": (-75.593, 36.175),
    "44056": (-75.714, 36.200),
    "44006": (-75.400, 36.300),
    "44019": (-75.200, 36.400),
    "44086": (-75.330, 35.750),
    "44095": (-75.330, 35.750),
    "jprn7": (-75.587, 35.912),
    "41017": (-75.100, 35.400),
    "41015": (-75.300, 35.400),
    "41120": (-75.258, 35.258),
    "dsln7": (-75.297, 35.153),
    "41025": (-75.454, 35.010),
    "41159": (-76.944, 34.211),
    "41036": (-76.949, 34.207),
    "41007": (-76.5, 34.2),
    "41110": (-77.715, 34.142),
    "41109": (-77.300, 34.484),
    "41035": (-77.281, 34.476),
    "41013": (-77.764, 33.441),
    "41108": (-78.016, 33.721),
    "ocpn7": (-78.147, 33.911),
    "ssbn7": (-78.484, 33.838),
    "41119": (-78.483, 33.842),
}

DEFAULT_BUOY_SCENARIO_STYLE: dict = {
    "folder": "buoy",
    "color": "#e6550d",
    "lw": 1.6,
    "ls": "--",
    "alpha": 0.95,
    "zorder": 20,
}


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, project_root: str | Path | None = None) -> Path:
    project_root = Path(project_root) if project_root is not None else _default_project_root()
    path = Path(path)
    return path if path.is_absolute() else project_root / path


def historical_dataset_folder(
    dataset: HistoricalDataset,
    project_root: str | Path | None = None,
) -> Path:
    """Resolved path for a named historical dataset preset."""
    preset = HISTORICAL_DATASET_FOLDERS[dataset]
    if dataset == "merged_grids":
        return Path(preset)
    return resolve_path(preset, project_root)


def resolve_historical_folder(
    *,
    historical_folder: str | Path | None = None,
    historical_dataset: HistoricalDataset | None = "merged_grids",
    project_root: str | Path | None = None,
) -> Path:
    """
    Resolve the historical wave folder.

    - If ``historical_folder`` is set, use it (absolute or relative to ``project_root``).
    - Else use ``historical_dataset`` (``\"merged_500m\"`` or ``\"merged_grids\"``).
    """
    if historical_folder is not None:
        return resolve_path(historical_folder, project_root)
    if historical_dataset is None:
        return resolve_path(DEFAULT_HISTORICAL_FOLDER, project_root)
    return historical_dataset_folder(historical_dataset, project_root)


def resolve_partitions_folder(
    *,
    partitions_folder: str | Path | None = None,
    partitions_dataset: HistoricalDataset | None = "merged_grids",
    project_root: str | Path | None = None,
) -> Path:
    """
    Folder for wave partitions (``phs*``, ``dp*``).

    Defaults to ``merged_grids`` (partitions live there even when bulk historical
    uses ``merged_500m``).
    """
    if partitions_folder is not None:
        return resolve_path(partitions_folder, project_root)
    if partitions_dataset is None:
        return Path(DEFAULT_MERGED_GRIDS_FOLDER)
    return historical_dataset_folder(partitions_dataset, project_root)


DEFAULT_KMA_MERGED_GRIDS_FOLDER = "outputs/BinWaves_BMUS"


def _uses_kma_merged_grids_folder(
    historical_folder: str | Path | None,
    project_root: str | Path | None = None,
) -> bool:
    """True when ``historical_folder`` points at pre-built BinWaves+KMA NetCDFs."""
    if historical_folder is None:
        return False
    folder = Path(historical_folder)
    if not folder.is_absolute() and project_root is not None:
        folder = resolve_path(folder, project_root)
    folder = folder.resolve()
    name = folder.name.lower()
    if name in ("binwaves_bmus", "merged_grids_binwaves_kma"):
        return True
    if "binwaves_bmus" in name or "merged_grids_binwaves_kma" in name:
        return True
    if project_root is not None:
        default = resolve_path(DEFAULT_KMA_MERGED_GRIDS_FOLDER, project_root).resolve()
        if folder == default:
            return True
    return False


def _historical_bulk_series_label(
    historical_folder: str | Path | None,
    project_root: str | Path | None = None,
) -> str:
    """Legend/column key for hindcast bulk (KMA merged grids vs plain merged_grids)."""
    if _uses_kma_merged_grids_folder(historical_folder, project_root):
        from utils import kma_cluster_swan as kcs

        return kcs.BINWAVES_PLUS_KMA_LABEL
    return "historical bulk"


def _resolve_binwaves_reference_folder(
    *,
    historical_folder: str | Path | None,
    historical_dataset: HistoricalDataset = "merged_grids",
    partitions_folder: str | Path | None = None,
    partitions_dataset: HistoricalDataset = "merged_grids",
    binwaves_reference_folder: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path | None:
    """Original BinWaves ``merged_grids`` when primary hindcast is KMA merged grids."""
    hist_folder = resolve_historical_folder(
        historical_folder=historical_folder,
        historical_dataset=historical_dataset,
        project_root=project_root,
    )
    if not _uses_kma_merged_grids_folder(hist_folder, project_root):
        return None
    if binwaves_reference_folder is not None:
        ref_folder = resolve_path(binwaves_reference_folder, project_root).resolve()
    else:
        ref_folder = resolve_partitions_folder(
            partitions_folder=partitions_folder,
            partitions_dataset=partitions_dataset,
            project_root=project_root,
        ).resolve()
    if ref_folder == hist_folder.resolve():
        ref_folder = Path(DEFAULT_MERGED_GRIDS_FOLDER).resolve()
    if ref_folder == hist_folder.resolve():
        return None
    return ref_folder


def _bulk_series_keys_in_bundles(
    bundles: Mapping[str, Mapping[str, tuple[xr.DataArray, dict]]],
    variables: Sequence[str] = ("hs", "tp", "dp"),
) -> list[str]:
    """Primary bulk column keys (excludes BinWaves reference overlay series)."""
    from utils import kma_cluster_swan as kcs

    keys: list[str] = []
    for label in (kcs.BINWAVES_PLUS_KMA_LABEL, "historical bulk"):
        if any(label in bundles[v] for v in variables):
            keys.append(label)
    return keys


def _binwaves_reference_in_bundles(
    bundles: Mapping[str, Mapping[str, tuple[xr.DataArray, dict]]],
    variables: Sequence[str] = ("hs", "tp", "dp"),
) -> bool:
    return any(BINWAVES_BULK_LABEL in bundles[v] for v in variables)


def _is_overlay_bulk_key(key: str) -> bool:
    """Bulk column that receives buoy / SWAN overlays (primary hindcast bulk)."""
    from utils import kma_cluster_swan as kcs

    return key in (kcs.BINWAVES_PLUS_KMA_LABEL, "historical bulk")


def _overlay_bulk_key(series_keys: Sequence[str]) -> str | None:
    for key in series_keys:
        if _is_overlay_bulk_key(key):
            return key
    return None


def _lat_lon_from_pair(pair: Sequence[float]) -> tuple[float, float]:
    """
    Return ``(lat, lon)`` from ``(lat, lon)`` or ``(lon, lat)``.

    Uses the same US East Coast heuristic as ``normalize_buoy_locations``:
    when ``|first| > |second|``, treat the pair as ``(lon, lat)``.
    """
    a, b = float(pair[0]), float(pair[1])
    if abs(a) > abs(b):
        lon, lat = a, b
    else:
        lat, lon = a, b
    return (lat, lon)


def normalize_coordinates(coords: Sequence[float] | Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """Accept one ``(lat, lon)`` / ``(lon, lat)`` pair or a list of pairs (auto-detected)."""
    if (
        isinstance(coords, (list, tuple))
        and len(coords) == 2
        and isinstance(coords[0], (int, float))
    ):
        return [_lat_lon_from_pair(coords)]
    return [_lat_lon_from_pair(c) for c in coords]


def _infer_bmus_grid_id(folder: Path) -> int | None:
    """Infer ShoreShop grid id from ``{var}_grid{N}_BinWaves_BMUS.nc`` files in ``folder``."""
    for path in sorted(folder.glob("*_grid*_BinWaves_BMUS.nc")):
        match = re.match(r".*_grid(\d+)_BinWaves_BMUS$", path.stem)
        if match:
            return int(match.group(1))
    return None


def _resolve_variable_nc_path(
    folder: str | Path,
    variable: str,
    project_root: str | Path | None = None,
    *,
    grid_id: int | None = None,
) -> Path:
    """Return path to a variable NetCDF in ``folder`` (merged grids or BinWaves_BMUS)."""
    folder = resolve_path(folder, project_root)
    candidates: list[Path] = []
    if grid_id is None:
        grid_id = _infer_bmus_grid_id(folder)
    if grid_id is not None:
        gid = f"grid{grid_id}"
        candidates.append(folder / f"{variable}_{gid}_BinWaves_BMUS.nc")
    else:
        candidates.extend(sorted(folder.glob(f"{variable}_grid*_BinWaves_BMUS.nc")))
    candidates.extend(
        folder / name for name in (f"{variable}_merged_all.nc", f"{variable}_500m.nc")
    )
    tried = [p.name for p in candidates]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No NetCDF for {variable!r} in {folder} (tried {tried})"
    )


def _variable_from_nc_path(nc_path: Path) -> str:
    stem = nc_path.stem
    for suffix in ("_merged_all", "_500m"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    match = re.match(r"^(.+)_grid\d+_BinWaves_BMUS$", stem)
    if match:
        return match.group(1)
    return stem.split("_", 1)[0]


def _apply_historical_folder_override(
    scenarios: Sequence[tuple[str, dict]],
    historical_folder: str | Path,
    project_root: str | Path | None = None,
) -> list[tuple[str, dict]]:
    """Point the ``historical`` scenario at ``historical_folder`` (e.g. merged_grids)."""
    hist_path = str(resolve_path(historical_folder, project_root))
    out: list[tuple[str, dict]] = []
    for label, cfg in scenarios:
        if label.strip().lower() == "historical":
            cfg = dict(cfg)
            cfg["folder"] = hist_path
        out.append((label, cfg))
    return out


def list_500m_variables(folder: str | Path, project_root: str | Path | None = None) -> list[str]:
    """Basename prefixes for all ``*_500m.nc`` files in a folder."""
    folder = resolve_path(folder, project_root)
    return sorted(
        p.name.replace("_500m.nc", "")
        for p in folder.glob("*_500m.nc")
        if p.is_file()
    )


def select_scenarios(
    scenarios: Mapping[str, dict] | None = None,
    *,
    models: Sequence[str] | None = None,
    include_historical: bool = True,
    historical_folder: str | Path = DEFAULT_HISTORICAL_FOLDER,
    project_root: str | Path | None = None,
) -> list[tuple[str, dict]]:
    """
    Return ``[(label, cfg), ...]`` for plotting.

    - ``models``: explicit list of scenario labels; if ``None``, use entries with
      ``enabled=True`` in ``scenarios``.
    - ``include_historical=False``: drop the historical reference run.
    """
    scenarios = dict(scenarios or DEFAULT_SCENARIOS)
    if models is not None:
        picked = []
        for label in models:
            if label not in scenarios:
                raise KeyError(f"Unknown model {label!r}. Available: {list(scenarios)}")
            picked.append((label, scenarios[label]))
    else:
        picked = [(k, v) for k, v in scenarios.items() if v.get("enabled", True)]

    if not include_historical:
        hist_resolved = resolve_path(historical_folder, project_root)
        picked = [
            (label, cfg)
            for label, cfg in picked
            if label.strip().lower() != "historical"
            and resolve_path(cfg["folder"], project_root) != hist_resolved
        ]
    return picked


def _resolve_nc_var(ds: xr.Dataset, variable: str) -> str:
    name = _resolve_bulk_nc_var_name(variable, ds)
    if name is None:
        data_vars = [
            v
            for v in ds.data_vars
            if v not in {"latitude", "longitude", "lat", "lon", "projected_coordinate_system"}
        ]
        if not data_vars:
            raise KeyError(f"No data variables in dataset. vars={list(ds.data_vars)}")
        name = data_vars[0]
    return name


def _monthly_agg(da: xr.DataArray, *, name_hint: str) -> xr.DataArray:
    if _is_directional_name(name_hint):
        return da.resample(time="MS").reduce(_circular_mean_deg_np, dim="time")
    return da.resample(time="MS").mean()


def apply_aggregation(da: xr.DataArray, variable: str, aggregation: str) -> xr.DataArray:
    aggregation = str(aggregation).lower()
    # native / hourly: no resampling (buoy pickles and merged_grids are hourly)
    if aggregation in ("native", "hourly"):
        return da
    if aggregation == "daily":
        return _daily_agg(da, name_hint=variable)
    if aggregation == "monthly":
        return _monthly_agg(_daily_agg(da, name_hint=variable), name_hint=variable)
    raise ValueError(
        f"aggregation must be 'hourly', 'native', 'daily', or 'monthly'; got {aggregation!r}"
    )


def load_timeseries_at_coordinate(
    nc_path: str | Path,
    coordinate: tuple[float, float],
    *,
    variable: str | None = None,
    aggregation: str = "daily",
    time_start=None,
    time_end=None,
    project_root: str | Path | None = None,
) -> tuple[xr.DataArray, dict]:
    """Load aggregated time series at the nearest site to ``(lat, lon)``."""
    nc_path = resolve_path(nc_path, project_root)
    if not nc_path.is_file():
        raise FileNotFoundError(nc_path)

    variable = variable or _variable_from_nc_path(Path(nc_path))
    selection = select_all_sites_from_merged_file(str(nc_path), selected_coordinate=coordinate)
    site_idx = int(selection.src_index[0])

    with xr.open_dataset(nc_path) as ds:
        lat_name, lon_name = _guess_lat_lon(ds)
        site_dim = _guess_site_dim(ds, lat_name, lon_name)
        nc_var = _resolve_nc_var(ds, variable)
        da = ds[nc_var].isel({site_dim: site_idx})
        da = apply_aggregation(da, variable, aggregation)
        if time_start is not None or time_end is not None:
            da = da.sel(time=slice(time_start, time_end))

    info = {
        "site_index": site_idx,
        "lat": float(selection.lats[0]),
        "lon": float(selection.lons[0]),
        "distance_km": float(selection.distance_m[0] / 1000.0),
        "nc_var": nc_var,
        "path": str(nc_path),
        "aggregation": str(aggregation).lower(),
        "n_points": int(da.sizes.get("time", da.size)),
    }
    return da, info


def load_scenario_bundle(
    coordinate: tuple[float, float],
    variable: str,
    scenarios: Sequence[tuple[str, dict]],
    *,
    aggregation: str = "daily",
    time_start=None,
    time_end=None,
    project_root: str | Path | None = None,
) -> tuple[dict[str, tuple[xr.DataArray, dict]], str]:
    """Load one variable for all scenarios at a coordinate."""
    series: dict[str, tuple[xr.DataArray, dict]] = {}
    site_note = ""

    for label, cfg in scenarios:
        folder = resolve_path(cfg["folder"], project_root)
        try:
            nc_path = _resolve_variable_nc_path(folder, variable, project_root=None)
        except FileNotFoundError:
            print(f"SKIP {label}: missing {variable} NetCDF in {folder}")
            continue
        da, info = load_timeseries_at_coordinate(
            nc_path,
            coordinate,
            variable=variable,
            aggregation=aggregation,
            time_start=time_start,
            time_end=time_end,
            project_root=project_root,
        )
        series[label] = (da, info)
        if not site_note:
            lat, lon = coordinate
            site_note = (
                f"site {info['site_index']} ({info['lat']:.3f}, {info['lon']:.3f}), "
                f"{info['distance_km']:.2f} km from ({lat:.3f}, {lon:.3f})"
            )
    return series, site_note


def longshore_transport_index(
    H: xr.DataArray,
    Dp_deg: xr.DataArray,
    shoreline_orientation_deg: float,
    *,
    K: float = DEFAULT_LONGSHORE_K,
) -> xr.DataArray:
    """
    CERC-like longshore sediment transport index (relative units).

    Uses Hs and wave direction at the point (no breaking transformation):
    ``Q = (H**5/2) * sin(2*alpha) * K`` with ``alpha = Dp - shoreline_orientation`` (degrees).

    Apply to daily- or monthly-aggregated Hs/Dp (same ``aggregation`` as other variables).
    """
    alpha_rad = np.deg2rad(Dp_deg - shoreline_orientation_deg)
    Q = (H ** 5 / 2) * np.sin(2.0 * alpha_rad) * K
    return Q


def _qs_timeseries_for_bundle_key(
    bundles: dict[str, dict[str, tuple[xr.DataArray, dict]]],
    key: str,
    shoreline_orientation_deg: float,
    *,
    K: float = DEFAULT_LONGSHORE_K,
    qs_formula: str = "cerc",
) -> xr.DataArray | None:
    """Instantaneous Qs for one bulk/partition column in Hs/Tp/Dp grids."""
    formula = str(qs_formula).lower()
    if formula not in ("cerc", "deep"):
        raise ValueError(f"qs_formula must be 'cerc' or 'deep'; got {qs_formula!r}")
    if key not in bundles.get("hs", {}) or key not in bundles.get("dp", {}):
        return None
    hs, _ = bundles["hs"][key]
    dp, _ = bundles["dp"][key]
    hs_a, dp_a = xr.align(hs, dp, join="inner")
    if hs_a.sizes.get("time", 0) == 0:
        return None
    if formula == "deep":
        if key not in bundles.get("tp", {}):
            print(f"SKIP Qs {key}: missing Tp for deep-water formula")
            return None
        tp, _ = bundles["tp"][key]
        hs_a, dp_a, tp_a = xr.align(hs_a, dp_a, tp, join="inner")
        if hs_a.sizes.get("time", 0) == 0:
            return None
        from utils.sediment_transport_NC import _compute_qs_series

        return _compute_qs_series(
            hs_a, dp_a, shoreline_orientation_deg, tp=tp_a, K=K, qs_formula="deep"
        )
    return longshore_transport_index(hs_a, dp_a, shoreline_orientation_deg, K=K)


def load_qs_scenario_bundle(
    coordinate: tuple[float, float],
    scenarios: Sequence[tuple[str, dict]],
    shoreline_orientation_deg: float,
    *,
    aggregation: str = "daily",
    time_start=None,
    time_end=None,
    K: float = DEFAULT_LONGSHORE_K,
    project_root: str | Path | None = None,
) -> tuple[dict[str, tuple[xr.DataArray, dict]], str]:
    """Load Hs + Dp per scenario, align times, return longshore transport index Qs."""
    hs_bundle, site_note = load_scenario_bundle(
        coordinate,
        "hs",
        scenarios,
        aggregation=aggregation,
        time_start=time_start,
        time_end=time_end,
        project_root=project_root,
    )
    dp_bundle, _ = load_scenario_bundle(
        coordinate,
        "dp",
        scenarios,
        aggregation=aggregation,
        time_start=time_start,
        time_end=time_end,
        project_root=project_root,
    )

    series: dict[str, tuple[xr.DataArray, dict]] = {}
    for label, (hs_da, info) in hs_bundle.items():
        if label not in dp_bundle:
            print(f"SKIP {label}: missing dp for Qs")
            continue
        dp_da, _ = dp_bundle[label]
        hs_a, dp_a = xr.align(hs_da, dp_da, join="inner")
        if hs_a.sizes.get("time", 0) == 0:
            print(f"SKIP {label}: no overlapping hs/dp times for Qs")
            continue
        qs = longshore_transport_index(hs_a, dp_a, shoreline_orientation_deg, K=K)
        qs.name = "Qs"
        info = dict(info)
        info["n_points"] = int(qs.sizes.get("time", qs.size))
        series[label] = (qs, info)
    return series, site_note


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_earth_km = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return float(2 * r_earth_km * np.arcsin(np.sqrt(a)))


@dataclass(frozen=True)
class BuoyCatalogEntry:
    """
    Buoy location, optional hindcast wave site, and shore-normal for Qs.

    ``wave_lat`` / ``wave_lon``: merged-grid site where hindcast Hs/Dp are extracted
    (defaults to nearest grid point to the buoy when omitted).
    """

    lat: float
    lon: float
    shore_normal_deg: float | None = None
    wave_lat: float | None = None
    wave_lon: float | None = None
    wave_site_index: int | None = None


def _wave_location_from_mapping(data: Mapping) -> tuple[float | None, float | None, int | None]:
    """Parse optional hindcast wave site from catalog keys."""
    site_index: int | None = None
    for key in ("site_index", "wave_site_index", "hist_site_index", "grid_site_index"):
        if key in data and data[key] is not None:
            site_index = int(data[key])
            break

    coord_raw: Any = None
    for key in (
        "waves",
        "wave_coord",
        "wave_location",
        "wave",
        "hist_coord",
        "historical_coord",
        "hindcast_coord",
    ):
        if key in data and data[key] is not None:
            coord_raw = data[key]
            break

    if coord_raw is None:
        return None, None, site_index

    lat, lon = _lat_lon_from_pair(coord_raw)
    return lat, lon, site_index


def _shore_normal_from_mapping(data: Mapping) -> float | None:
    for key in ("sn", "shore_normal", "shore_normal_deg", "shoreline_orientation_deg"):
        if key in data and data[key] is not None:
            return float(data[key])
    return None


def parse_buoy_entry(value: Any) -> BuoyCatalogEntry:
    """
  Parse a buoy catalog entry.

  Coordinates (``(lon, lat)`` or ``(lat, lon)`` auto-detected):

  - ``(-75.7, 36.2)``
  - ``[(-75.7, 36.2), 69]`` or ``((-75.7, 36.2), 69)`` — second value is shore-normal ``sn``
  - ``{"coord": (-75.7, 36.2), "sn": 69}``
  - ``{"coord": (-75.7, 36.2), "waves": (-75.71, 36.20), "site_index": 1669, "sn": 69}``
    (``waves`` = merged hindcast extraction point, ``(lon, lat)`` or ``(lat, lon)``)
  - ``{"lon": -75.7, "lat": 36.2, "sn": 69}``
    """
    sn: float | None = None
    wave_lat: float | None = None
    wave_lon: float | None = None
    wave_site_index: int | None = None
    coord_raw: Any = value

    if isinstance(value, Mapping):
        sn = _shore_normal_from_mapping(value)
        wave_lat, wave_lon, wave_site_index = _wave_location_from_mapping(value)
        if "coord" in value:
            coord_raw = value["coord"]
        elif "coordinates" in value:
            coord_raw = value["coordinates"]
        elif "lat" in value and "lon" in value:
            lat, lon = float(value["lat"]), float(value["lon"])
            return BuoyCatalogEntry(
                lat, lon, sn, wave_lat, wave_lon, wave_site_index
            )
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        first, second = value[0], value[1]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            if isinstance(second, (int, float)):
                coord_raw, sn = first, float(second)
            elif isinstance(second, Mapping):
                coord_raw, sn = first, _shore_normal_from_mapping(second)
            else:
                coord_raw = value
        elif isinstance(first, Mapping):
            return parse_buoy_entry(first)
        else:
            coord_raw = value

    lat, lon = _lat_lon_from_pair(coord_raw)
    return BuoyCatalogEntry(lat, lon, sn, wave_lat, wave_lon, wave_site_index)


def parse_buoy_catalog(buoys: Mapping[str, Any]) -> dict[str, BuoyCatalogEntry]:
    """Return ``{buoy_id: BuoyCatalogEntry}`` from a user buoy mapping."""
    return {str(bid): parse_buoy_entry(entry) for bid, entry in buoys.items()}


def normalize_buoy_locations(
    buoys: Mapping[str, Any],
) -> dict[str, tuple[float, float]]:
    """
    Return ``{buoy_id: (lat, lon)}`` from catalog entries (see ``parse_buoy_entry``).
    """
    return {bid: (e.lat, e.lon) for bid, e in parse_buoy_catalog(buoys).items()}


def buoy_wave_locations_from_catalog(
    buoys: Mapping[str, Any] | None,
    *,
    include_defaults: bool = True,
) -> dict[str, dict[str, float | int | None]]:
    """
    Return buoy vs hindcast wave sites per catalog entry.

    Keys: ``buoy_lat``, ``buoy_lon``, ``wave_lat``, ``wave_lon``, ``wave_site_index``
    (wave fields ``None`` when not set in the catalog).
    """
    catalog = merge_buoy_catalog(buoys, include_defaults=include_defaults)
    out: dict[str, dict[str, float | int | None]] = {}
    for bid, e in catalog.items():
        out[bid] = {
            "buoy_lat": e.lat,
            "buoy_lon": e.lon,
            "wave_lat": e.wave_lat,
            "wave_lon": e.wave_lon,
            "wave_site_index": e.wave_site_index,
            "shore_normal_deg": e.shore_normal_deg,
        }
    return out


def buoy_shore_normals_from_catalog(
    buoys: Mapping[str, Any] | None,
    *,
    include_defaults: bool = True,
) -> dict[str, float]:
    """``{buoy_id: shore_normal_deg}`` for entries that define ``sn`` (case-insensitive keys)."""
    catalog = merge_buoy_catalog(buoys, include_defaults=include_defaults)
    return {
        bid: float(entry.shore_normal_deg)
        for bid, entry in catalog.items()
        if entry.shore_normal_deg is not None
    }


def list_buoy_ids_in_data_dir(buoy_data_dir: str | Path = DEFAULT_BUOY_DATA_DIR) -> list[str]:
    """Buoy IDs with ``buoy_<id>_bulk_parameters.pkl`` in ``buoy_data_dir``."""
    buoy_data_dir = Path(buoy_data_dir)
    ids: list[str] = []
    for path in sorted(buoy_data_dir.glob("buoy_*_bulk_parameters.pkl")):
        stem = path.stem  # buoy_44056_bulk_parameters
        bid = stem.replace("buoy_", "").replace("_bulk_parameters", "")
        if bid:
            ids.append(bid)
    return ids


def merge_buoy_catalog(
    buoys: Mapping[str, Any] | None = None,
    *,
    include_defaults: bool = True,
) -> dict[str, BuoyCatalogEntry]:
    """Merge user ``buoys`` with ``DEFAULT_NC_BUOYS`` (user entries override)."""
    catalog: dict[str, BuoyCatalogEntry] = {}
    if include_defaults:
        for bid, pair in DEFAULT_NC_BUOYS.items():
            catalog[str(bid)] = parse_buoy_entry(pair)
    if buoys:
        for bid, entry in buoys.items():
            catalog[str(bid)] = parse_buoy_entry(entry)
    return catalog


def _resolve_buoy_location(
    buoy_id: str,
    buoys: Mapping[str, Any],
    *,
    buoy_data_dir: str | Path | None = None,
) -> tuple[str, BuoyCatalogEntry]:
    """Return ``(canonical_id, BuoyCatalogEntry)`` for ``buoy_id`` (case-insensitive)."""
    catalog = merge_buoy_catalog(buoys, include_defaults=True)
    lookup = {k.lower(): (k, v) for k, v in catalog.items()}
    key = str(buoy_id).lower()
    if key not in lookup:
        has_pkl = (
            buoy_data_dir is not None
            and _buoy_pkl_path(buoy_id, Path(buoy_data_dir)) is not None
        )
        in_dir = list_buoy_ids_in_data_dir(buoy_data_dir) if buoy_data_dir else []
        msg = f"Unknown buoy {buoy_id!r}. Known IDs: {sorted(catalog)}"
        if has_pkl:
            msg += (
                f". Pickle exists in {buoy_data_dir} — add coordinates via "
                f"buoys={{'{buoy_id}': (lon, lat)}}."
            )
        elif in_dir:
            msg += f". IDs with data files (no coords): {[b for b in in_dir if b.lower() not in lookup]}"
        raise KeyError(msg)
    bid, entry = lookup[key]
    return bid, entry


def find_nearest_coordinate(
    target: tuple[float, float],
    candidates: Sequence[float] | Sequence[Sequence[float]],
) -> tuple[tuple[float, float], float, int]:
    """
    Return ``((lat, lon), distance_km, candidate_index)`` for the candidate nearest ``target``.
    """
    coords = normalize_coordinates(candidates)
    if not coords:
        raise ValueError("candidates must contain at least one (lat, lon) pair")

    tlat, tlon = target
    best_i = 0
    best_km = float("inf")
    for i, (lat, lon) in enumerate(coords):
        dist_km = _haversine_km(tlat, tlon, lat, lon)
        if dist_km < best_km:
            best_i, best_km = i, dist_km
    return coords[best_i], best_km, best_i


def find_nearest_historical_coordinate(
    target: tuple[float, float],
    *,
    historical_folder: str | Path | None = None,
    historical_dataset: HistoricalDataset = "merged_grids",
    project_root: str | Path | None = None,
) -> tuple[tuple[float, float], float, dict]:
    """
    Nearest site in the historical hs NetCDF to ``target`` ``(lat, lon)``.

    Returns ``((lat, lon), distance_km, info)`` with ``info['site_index']``.
    """
    folder = resolve_historical_folder(
        historical_folder=historical_folder,
        historical_dataset=historical_dataset,
        project_root=project_root,
    )
    hist_nc = _resolve_variable_nc_path(folder, "hs", project_root=None)

    selection = select_all_sites_from_merged_file(
        str(hist_nc),
        selected_coordinate=target,
    )
    lat = float(selection.lats[0])
    lon = float(selection.lons[0])
    dist_km = float(selection.distance_m[0] / 1000.0)
    info = {
        "site_index": int(selection.src_index[0]),
        "path": str(hist_nc),
        "distance_km": dist_km,
        "historical_dataset": historical_dataset,
        "historical_folder": str(folder),
    }
    return (lat, lon), dist_km, info


def match_buoy_to_historical(
    buoy_id: str,
    *,
    coordinates: Sequence[float] | Sequence[Sequence[float]] | None = None,
    buoys: Mapping[str, Any] | None = None,
    buoy_data_dir: str | Path = DEFAULT_BUOY_DATA_DIR,
    historical_folder: str | Path | None = None,
    historical_dataset: HistoricalDataset = "merged_grids",
    project_root: str | Path | None = None,
) -> BuoyHistoricalMatch:
    """
    Map a buoy ID to the nearest model/historical coordinate.

    - If ``coordinates`` is given, search only among those ``(lat, lon)`` points
      (e.g. your coastal transect list).
    - If ``coordinates`` is ``None``, search all sites in the historical hs NetCDF
      (``hs_merged_all.nc`` in ``merged_grids`` by default).
    """
    bid, entry = _resolve_buoy_location(buoy_id, buoys or {}, buoy_data_dir=buoy_data_dir)
    blat, blon = entry.lat, entry.lon

    if entry.wave_lat is not None and entry.wave_lon is not None:
        coord = (float(entry.wave_lat), float(entry.wave_lon))
        dist_km = _haversine_km(blat, blon, coord[0], coord[1])
        site_index = entry.wave_site_index
        if site_index is None:
            _, _, info = find_nearest_historical_coordinate(
                coord,
                historical_folder=historical_folder,
                historical_dataset=historical_dataset,
                project_root=project_root,
            )
            site_index = info.get("site_index")
        source: Literal["coordinates", "historical_grid", "catalog_waves"] = "catalog_waves"
    elif coordinates is not None:
        coord, dist_km, _idx = find_nearest_coordinate((blat, blon), coordinates)
        source = "coordinates"
        site_index = None
    else:
        coord, dist_km, info = find_nearest_historical_coordinate(
            (blat, blon),
            historical_folder=historical_folder,
            historical_dataset=historical_dataset,
            project_root=project_root,
        )
        source = "historical_grid"
        site_index = info.get("site_index")

    return BuoyHistoricalMatch(
        buoy_id=bid,
        buoy_lat=blat,
        buoy_lon=blon,
        coordinate=coord,
        distance_km=dist_km,
        match_source=source,
        site_index=site_index,
    )


def find_nearest_buoy(
    coordinate: tuple[float, float],
    buoys: Mapping[str, tuple[float, float]],
) -> tuple[str, float]:
    """Return ``(buoy_id, distance_km)`` for the buoy nearest ``(lat, lon)``."""
    lat, lon = coordinate
    best_id = ""
    best_km = float("inf")
    for buoy_id, (blat, blon) in buoys.items():
        dist_km = _haversine_km(lat, lon, float(blat), float(blon))
        if dist_km < best_km:
            best_id, best_km = buoy_id, dist_km
    return best_id, best_km


def _buoy_pkl_path(buoy_id: str, buoy_data_dir: Path) -> Path | None:
    for key in (str(buoy_id), str(buoy_id).lower()):
        path = buoy_data_dir / f"buoy_{key}_bulk_parameters.pkl"
        if path.is_file():
            return path
    return None


def load_buoy_bulk_dataframe(
    buoy_id: str,
    buoy_data_dir: str | Path,
) -> pd.DataFrame | None:
    """Load raw buoy bulk parameters (datetime index, ``Hs_Buoy``, ``Dir_Buoy``, …)."""
    buoy_data_dir = Path(buoy_data_dir)
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
    # Unnamed DatetimeIndex (some pickles use index name 'datetime' → xarray dim mismatch).
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index), name=None)
    return df


def get_buoy_observation_period(
    buoy_id: str,
    buoy_data_dir: str | Path = DEFAULT_BUOY_DATA_DIR,
) -> tuple[pd.Timestamp, pd.Timestamp] | tuple[None, None]:
    """
    Start/end of valid buoy observations (finite ``Hs_Buoy`` and ``Dir_Buoy``, Hs > 0).
    """
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


def load_buoy_qs(
    buoy_id: str,
    shoreline_orientation_deg: float,
    *,
    buoy_data_dir: str | Path,
    aggregation: str = "daily",
    time_start=None,
    time_end=None,
    K: float = DEFAULT_LONGSHORE_K,
) -> tuple[xr.DataArray | None, dict]:
    """
    Load NDBC buoy bulk parameters and compute CERC longshore transport index Qs.

    Expects pickle files ``buoy_<id>_bulk_parameters.pkl`` with columns
    ``Hs_Buoy`` and ``Dir_Buoy`` (hourly index).
    """
    df = load_buoy_bulk_dataframe(buoy_id, buoy_data_dir)
    if df is None:
        return None, {}

    pkl_path = _buoy_pkl_path(buoy_id, Path(buoy_data_dir))
    missing = [c for c in ("Hs_Buoy", "Dir_Buoy") if c not in df.columns]
    if missing:
        print(f"SKIP buoy {buoy_id}: missing columns {missing} in {pkl_path.name}")
        return None, {}

    hs = _series_to_dataarray(df["Hs_Buoy"], name="hs")
    dp = _series_to_dataarray(df["Dir_Buoy"], name="dp")
    if time_start is not None or time_end is not None:
        hs = hs.sel(time=slice(time_start, time_end))
        dp = dp.sel(time=slice(time_start, time_end))

    hs = apply_aggregation(hs, "hs", aggregation)
    dp = apply_aggregation(dp, "dp", aggregation)
    hs_a, dp_a = xr.align(hs, dp, join="inner")
    if hs_a.sizes.get("time", 0) == 0:
        print(f"SKIP buoy {buoy_id}: no overlapping Hs/Dir after aggregation")
        return None, {}

    qs = longshore_transport_index(hs_a, dp_a, shoreline_orientation_deg, K=K)
    qs.name = "Qs"
    info = {
        "buoy_id": str(buoy_id),
        "path": str(pkl_path) if pkl_path else "",
        "aggregation": str(aggregation).lower(),
        "n_points": int(qs.sizes.get("time", qs.size)),
        "time_start": str(time_start) if time_start is not None else None,
        "time_end": str(time_end) if time_end is not None else None,
    }
    return qs, info


def load_buoy_hs_dp(
    buoy_id: str,
    *,
    buoy_data_dir: str | Path,
    aggregation: str = "daily",
    time_start=None,
    time_end=None,
) -> tuple[tuple[xr.DataArray, xr.DataArray] | None, dict]:
    """
    Load aligned buoy bulk Hs and wave direction (``Hs_Buoy``, ``Dir_Buoy``).

    Same pickles and aggregation rules as ``load_buoy_qs`` (hourly native data by default).
    """
    df = load_buoy_bulk_dataframe(buoy_id, buoy_data_dir)
    if df is None:
        return None, {}

    pkl_path = _buoy_pkl_path(buoy_id, Path(buoy_data_dir))
    missing = [c for c in ("Hs_Buoy", "Dir_Buoy") if c not in df.columns]
    if missing:
        print(f"SKIP buoy {buoy_id}: missing columns {missing} in {pkl_path.name}")
        return None, {}

    hs = _series_to_dataarray(df["Hs_Buoy"], name="hs")
    dp = _series_to_dataarray(df["Dir_Buoy"], name="dp")
    if time_start is not None or time_end is not None:
        hs = hs.sel(time=slice(time_start, time_end))
        dp = dp.sel(time=slice(time_start, time_end))

    hs = apply_aggregation(hs, "hs", aggregation)
    dp = apply_aggregation(dp, "dp", aggregation)
    hs_a, dp_a = xr.align(hs, dp, join="inner")
    if hs_a.sizes.get("time", 0) == 0:
        print(f"SKIP buoy {buoy_id}: no overlapping Hs/Dir after aggregation")
        return None, {}

    info = {
        "buoy_id": str(buoy_id),
        "path": str(pkl_path) if pkl_path else "",
        "aggregation": str(aggregation).lower(),
        "n_points": int(hs_a.sizes.get("time", hs_a.size)),
        "time_start": str(time_start) if time_start is not None else None,
        "time_end": str(time_end) if time_end is not None else None,
    }
    return (hs_a, dp_a), info


_BUOY_TP_COLUMNS = ("Tp_Buoy", "TP_Buoy", "Tp", "PeakPeriod_Buoy", "Tm_Buoy")


def load_buoy_hs_tp_dp(
    buoy_id: str,
    *,
    buoy_data_dir: str | Path = DEFAULT_BUOY_DATA_DIR,
    aggregation: str = "daily",
    time_start=None,
    time_end=None,
) -> tuple[dict[str, xr.DataArray | None], dict]:
    """
    Load NDBC buoy Hs, Dp, and Tp (if present) for partition-style grid plots.

    Returns ``({"hs": da, "tp": da|None, "dp": da}, info)`` or ``({}, {})`` on failure.
    """
    df = load_buoy_bulk_dataframe(buoy_id, buoy_data_dir)
    if df is None:
        return {}, {}

    pkl_path = _buoy_pkl_path(buoy_id, Path(buoy_data_dir))
    missing = [c for c in ("Hs_Buoy", "Dir_Buoy") if c not in df.columns]
    if missing:
        print(f"SKIP buoy {buoy_id}: missing columns {missing} in {pkl_path.name}")
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
        "time_start": str(time_start) if time_start is not None else None,
        "time_end": str(time_end) if time_end is not None else None,
    }
    return {"hs": hs_a, "tp": tp_da, "dp": dp_a}, info


def _swan_timeseries_filename(lat: float, lon: float) -> str:
    return f"swan_point_timeseries_lon{lon:.3f}_lat{lat:.3f}.nc"


def resolve_swan_timeseries_path(
    coordinate: tuple[float, float],
    *,
    swan_timeseries_path: str | Path | None = None,
    swan_timeseries_folder: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path | None:
    """Resolve a SWAN point bulk-parameter NetCDF for ``(lat, lon)``."""
    if swan_timeseries_path is not None:
        path = resolve_path(swan_timeseries_path, project_root)
        return path if path.is_file() else None
    if swan_timeseries_folder is None:
        return None
    lat, lon = coordinate
    folder = resolve_path(swan_timeseries_folder, project_root)
    path = folder / _swan_timeseries_filename(lat, lon)
    return path if path.is_file() else None


def _swan_bulk_in_bundles(bundles: dict[str, dict[str, tuple[xr.DataArray, dict]]]) -> bool:
    return any(SWAN_BULK_LABEL in bundles[v] for v in ("hs", "tp", "dp"))


def _resolve_swan_overlay_label(
    bundles: dict[str, dict[str, tuple[xr.DataArray, dict]]],
    swan_bulk_label: str | None = None,
) -> str | None:
    """Return the bundle key used for SWAN bulk overlay, if present."""
    if swan_bulk_label and any(swan_bulk_label in bundles[v] for v in ("hs", "tp", "dp")):
        return swan_bulk_label
    if _swan_bulk_in_bundles(bundles):
        return SWAN_BULK_LABEL
    return None


def load_swan_bulk_hs_tp_dp(
    nc_path: str | Path,
    *,
    aggregation: str = "daily",
    time_start=None,
    time_end=None,
    project_root: str | Path | None = None,
) -> tuple[dict[str, xr.DataArray | None], dict]:
    """
    Load SWAN bulk Hs/Tp/Dp from a point timeseries NetCDF (no partitions).

    Expects variables ``Hsig``, ``Tps``, and ``pdir`` on a ``time`` dimension.
    """
    nc_path = resolve_path(nc_path, project_root)
    if not nc_path.is_file():
        print(f"SKIP SWAN bulk: not found {nc_path}")
        return {}, {}

    panels: dict[str, xr.DataArray | None] = {}
    lat = float("nan")
    lon = float("nan")
    try:
        with xr.open_dataset(nc_path) as ds:
            if "latitude" in ds:
                lat = float(ds["latitude"].values)
            if "longitude" in ds:
                lon = float(ds["longitude"].values)
            for panel_var, nc_var in _SWAN_BULK_NC_VARS.items():
                if nc_var not in ds:
                    print(f"SKIP SWAN bulk {panel_var}: missing {nc_var} in {nc_path.name}")
                    panels[panel_var] = None
                    continue
                da = ds[nc_var].load()
                if "time" not in da.dims:
                    print(f"SKIP SWAN bulk {panel_var}: no time dim in {nc_path.name}")
                    panels[panel_var] = None
                    continue
                da = apply_aggregation(da, panel_var, aggregation)
                if time_start is not None or time_end is not None:
                    da = da.sel(time=slice(time_start, time_end))
                panels[panel_var] = da.rename(panel_var)
    except Exception as exc:
        print(f"SKIP SWAN bulk: failed to load {nc_path}: {exc}")
        return {}, {}

    if panels.get("hs") is None or panels.get("dp") is None:
        print(f"SKIP SWAN bulk: missing Hsig/pdir after load from {nc_path.name}")
        return {}, {}

    hs_a, dp_a = xr.align(panels["hs"], panels["dp"], join="inner")
    if hs_a.sizes.get("time", 0) == 0:
        print(f"SKIP SWAN bulk: no overlapping Hsig/pdir times in {nc_path.name}")
        return {}, {}
    panels["hs"], panels["dp"] = hs_a, dp_a

    tp_da = panels.get("tp")
    if tp_da is not None:
        hs_a, dp_a, tp_a = xr.align(panels["hs"], panels["dp"], tp_da, join="inner")
        if hs_a.sizes.get("time", 0) == 0:
            print(f"SKIP SWAN bulk: no overlapping Hsig/Tps/pdir times in {nc_path.name}")
            return {}, {}
        panels["hs"], panels["dp"], panels["tp"] = hs_a, dp_a, tp_a

    info = {
        "path": str(nc_path),
        "lat": lat,
        "lon": lon,
        "aggregation": str(aggregation).lower(),
        "n_points": int(panels["hs"].sizes.get("time", panels["hs"].size)),
        "time_start": str(time_start) if time_start is not None else None,
        "time_end": str(time_end) if time_end is not None else None,
    }
    return panels, info


def _resolve_buoy_id_for_coordinate(
    coordinate: tuple[float, float],
    *,
    buoy_id: str | None = None,
    buoys: Mapping[str, Any] | None = None,
    buoy_data_dir: str | Path = DEFAULT_BUOY_DATA_DIR,
    max_distance_km: float | None = 75.0,
) -> str | None:
    """Pick explicit ``buoy_id`` or nearest catalog buoy with a pickle file."""
    if buoy_id is not None:
        bid = str(buoy_id)
        if _buoy_pkl_path(bid, Path(buoy_data_dir)) is None:
            print(f"SKIP buoy {bid}: no pickle in {buoy_data_dir}")
            return None
        return bid

    if not buoys:
        return None

    locs = normalize_buoy_locations(buoys)
    if not locs:
        return None

    bid, dist_km = find_nearest_buoy(coordinate, locs)
    if not bid:
        return None
    if max_distance_km is not None and dist_km > max_distance_km:
        print(
            f"SKIP buoy: nearest {bid} is {dist_km:.1f} km from "
            f"({coordinate[0]:.3f}, {coordinate[1]:.3f}) "
            f"(max {max_distance_km:.1f} km)"
        )
        return None
    if _buoy_pkl_path(bid, Path(buoy_data_dir)) is None:
        print(f"SKIP buoy {bid}: no pickle in {buoy_data_dir}")
        return None
    print(f"Using nearest buoy {bid} ({dist_km:.1f} km from target)")
    return bid


def _buoy_labels_in_bundles(
    bundles: dict[str, dict[str, tuple[xr.DataArray, dict]]],
) -> list[str]:
    """Sorted ``buoy <id>`` column labels present in ``bundles['hs']``."""
    return sorted(k for k in bundles.get("hs", {}) if k.startswith("buoy "))


def _primary_buoy_label(
    bundles: dict[str, dict[str, tuple[xr.DataArray, dict]]],
) -> str | None:
    labels = _buoy_labels_in_bundles(bundles)
    return labels[0] if labels else None


def _partition_hs_tp_dp_series_keys(
    bundles: dict[str, dict[str, tuple[xr.DataArray, dict]]],
    partition_ids: Sequence[int],
    *,
    include_bulk: bool,
    overlay_buoy_on_bulk: bool = False,
) -> list[str]:
    """Column order: bulk columns (+ buoy overlay on primary bulk), then partitions."""
    variables = ("hs", "tp", "dp")
    keys: list[str] = []
    if include_bulk:
        keys.extend(_bulk_series_keys_in_bundles(bundles, variables))
    if not overlay_buoy_on_bulk:
        keys.extend(_buoy_labels_in_bundles(bundles))
    for pid in partition_ids:
        key = f"partition {pid}"
        if any(key in bundles[v] for v in variables):
            keys.append(key)
    return keys


def _bulk_column_title(
    bundles: dict[str, dict[str, tuple[xr.DataArray, dict]]],
    *,
    bulk_key: str | None = None,
    overlay_buoy_label: str | None = None,
    overlay_swan_on_bulk: bool = False,
) -> str:
    from utils import kma_cluster_swan as kcs

    if bulk_key == kcs.BINWAVES_PLUS_KMA_LABEL:
        parts = ["bulk+KMA"]
    elif bulk_key == BINWAVES_BULK_LABEL:
        parts = ["BinWaves"]
    else:
        parts = ["bulk"]
    if overlay_buoy_label and bulk_key is not None and _is_overlay_bulk_key(bulk_key):
        parts.append(overlay_buoy_label.replace("buoy ", ""))
    if overlay_swan_on_bulk and bulk_key is not None and _is_overlay_bulk_key(bulk_key):
        if _swan_bulk_in_bundles(bundles):
            parts.append("SWAN_WIND")
    if bulk_key is not None and _is_overlay_bulk_key(bulk_key) and _binwaves_reference_in_bundles(bundles):
        parts.append("BinWaves")
    return "+".join(parts)


def _partition_series_key_style(
    key: str,
    column_index: int,
    *,
    overlay_buoy_label: str | None = None,
    overlay_swan_on_bulk: bool = False,
    bundles: dict[str, dict[str, tuple[xr.DataArray, dict]]] | None = None,
) -> tuple[dict[str, Any], str]:
    """Matplotlib line style and short column title for grid plots."""
    from utils import kma_cluster_swan as kcs

    pid = _partition_id_from_qs_label(key)
    if _is_overlay_bulk_key(key):
        if bundles is not None:
            title = _bulk_column_title(
                bundles,
                bulk_key=key,
                overlay_buoy_label=overlay_buoy_label,
                overlay_swan_on_bulk=overlay_swan_on_bulk,
            )
        elif overlay_buoy_label:
            bid = overlay_buoy_label.replace("buoy ", "")
            title = f"bulk+{bid}"
        else:
            title = "bulk+KMA" if key == kcs.BINWAVES_PLUS_KMA_LABEL else "bulk"
        color = SWAN_WIND_OVERLAY_COLOR if key == kcs.BINWAVES_PLUS_KMA_LABEL else BINWAVES_OVERLAY_COLOR
        return {"color": color, "lw": 1.2, "alpha": 0.95}, title
    if key.startswith("buoy "):
        return {"color": BUOY_OVERLAY_COLOR, "lw": 1.1, "alpha": 0.9}, key.replace("buoy ", "buoy\n")
    if pid is not None:
        return {
            "color": _qs_partition_color(pid, column_index),
            "lw": 0.9,
            "alpha": 0.85,
        }, f"h{pid}"
    return {"lw": 0.9, "alpha": 0.85}, key


def _buoy_overlay_line_style() -> dict[str, Any]:
    return {"color": BUOY_OVERLAY_COLOR, "lw": 1.0, "alpha": 0.85}


def _swan_overlay_line_style() -> dict[str, Any]:
    return {"color": SWAN_WIND_OVERLAY_COLOR, "lw": 1.1, "alpha": 0.9}


def _binwaves_reference_overlay_line_style() -> dict[str, Any]:
    return {"color": BINWAVES_OVERLAY_COLOR, "lw": 1.0, "alpha": 0.85, "ls": "--"}


def _plot_buoy_overlay_on_ax(
    ax,
    bundles: dict[str, dict[str, tuple[xr.DataArray, dict]]],
    buoy_label: str,
    var: str,
    *,
    qs_da: xr.DataArray | None = None,
) -> None:
    """Draw buoy series on an axes that already shows hindcast bulk."""
    if var == "qs":
        if qs_da is None:
            return
        _plot_timeseries_on_ax(
            ax,
            qs_da["time"].values,
            qs_da.values,
            directional=False,
            **_buoy_overlay_line_style(),
        )
        return
    panel = bundles.get(var, {})
    if buoy_label not in panel:
        return
    da, _ = panel[buoy_label]
    _plot_timeseries_on_ax(
        ax,
        da["time"].values,
        da.values,
        directional=_is_directional_name(var),
        **_buoy_overlay_line_style(),
    )


def _plot_swan_overlay_on_ax(
    ax,
    bundles: dict[str, dict[str, tuple[xr.DataArray, dict]]],
    var: str,
    *,
    qs_da: xr.DataArray | None = None,
    swan_label: str = SWAN_BULK_LABEL,
) -> None:
    """Draw SWAN bulk series on an axes that already shows hindcast bulk."""
    if var == "qs":
        if qs_da is None:
            return
        _plot_timeseries_on_ax(
            ax,
            qs_da["time"].values,
            qs_da.values,
            directional=False,
            **_swan_overlay_line_style(),
        )
        return
    panel = bundles.get(var, {})
    if swan_label not in panel:
        return
    da, _ = panel[swan_label]
    _plot_timeseries_on_ax(
        ax,
        da["time"].values,
        da.values,
        directional=_is_directional_name(var),
        **_swan_overlay_line_style(),
    )


def _plot_binwaves_reference_overlay_on_ax(
    ax,
    bundles: dict[str, dict[str, tuple[xr.DataArray, dict]]],
    var: str,
    *,
    qs_da: xr.DataArray | None = None,
) -> None:
    """Draw original BinWaves bulk on the primary hindcast bulk axes."""
    if var == "qs":
        if qs_da is None:
            return
        _plot_timeseries_on_ax(
            ax,
            qs_da["time"].values,
            qs_da.values,
            directional=False,
            **_binwaves_reference_overlay_line_style(),
        )
        return
    panel = bundles.get(var, {})
    if BINWAVES_BULK_LABEL not in panel:
        return
    da, _ = panel[BINWAVES_BULK_LABEL]
    _plot_timeseries_on_ax(
        ax,
        da["time"].values,
        da.values,
        directional=_is_directional_name(var),
        **_binwaves_reference_overlay_line_style(),
    )


def _append_nearest_buoy_qs(
    series: dict[str, tuple[xr.DataArray, dict]],
    coordinate: tuple[float, float],
    shoreline_orientation_deg: float,
    *,
    buoy_data_dir: str | Path,
    buoys: Mapping[str, Sequence[float]],
    buoy_max_distance_km: float | None,
    aggregation: str,
    time_start,
    time_end,
    K: float,
) -> tuple[dict[str, tuple[xr.DataArray, dict]], list[tuple[str, dict]], str]:
    """
    Add nearest-buoy Qs to ``series``.

    Returns ``(series, extra_scenarios_cfg, note_suffix)``.
    """
    buoy_dir = Path(buoy_data_dir)
    if not buoy_dir.is_dir():
        print(f"SKIP buoy overlay: directory not found: {buoy_dir}")
        return series, [], ""

    buoys_latlon = normalize_buoy_locations(merge_buoy_catalog(buoys))
    buoy_id, dist_km = find_nearest_buoy(coordinate, buoys_latlon)
    if not buoy_id:
        return series, [], ""

    if buoy_max_distance_km is not None and dist_km > buoy_max_distance_km:
        lat, lon = coordinate
        print(
            f"SKIP buoy: nearest {buoy_id} is {dist_km:.1f} km from "
            f"({lat:.3f}, {lon:.3f}) (max {buoy_max_distance_km:.1f} km)"
        )
        return series, [], ""

    qs, info = load_buoy_qs(
        buoy_id,
        shoreline_orientation_deg,
        buoy_data_dir=buoy_dir,
        aggregation=aggregation,
        time_start=time_start,
        time_end=time_end,
        K=K,
    )
    if qs is None:
        return series, [], ""

    blat, blon = buoys_latlon[buoy_id]
    info = dict(info)
    info["lat"] = blat
    info["lon"] = blon
    info["distance_km"] = dist_km

    label = f"buoy {buoy_id}"
    series = dict(series)
    series[label] = (qs, info)
    extra_cfg = [(label, dict(DEFAULT_BUOY_SCENARIO_STYLE))]
    note = f"buoy {buoy_id} @ ({blat:.3f}, {blon:.3f}), {dist_km:.1f} km"
    return series, extra_cfg, note


def _partition_ids_from_folder(folder: str | Path, project_root: str | Path | None = None) -> list[int]:
    folder = resolve_path(folder, project_root)
    ids = set()
    for p in folder.glob("phs*_merged_all.nc"):
        suffix = p.stem.replace("phs", "").replace("_merged_all", "")
        if suffix.isdigit():
            ids.add(int(suffix))
    for p in folder.glob("phs*_grid*_BinWaves_BMUS.nc"):
        prefix = p.stem.split("_grid", 1)[0].replace("phs", "")
        if prefix.isdigit():
            ids.add(int(prefix))
    return sorted(ids)


def _site_note_from_info(coordinate: tuple[float, float], info: dict) -> str:
    lat, lon = coordinate
    path_bit = ""
    if info.get("path"):
        path_bit = f" | {Path(info['path']).name}"
    return (
        f"site {info['site_index']} ({info['lat']:.3f}, {info['lon']:.3f}), "
        f"{info['distance_km']:.2f} km from ({lat:.3f}, {lon:.3f}){path_bit}"
    )


def _warn_distant_site_match(
    coordinate: tuple[float, float],
    info: dict,
    *,
    label: str,
    max_distance_km: float = 50.0,
) -> None:
    dist = float(info.get("distance_km", 0.0))
    if dist <= max_distance_km:
        return
    lat, lon = coordinate
    print(
        f"WARNING [{label}]: nearest grid site {info.get('site_index')} is {dist:.1f} km from "
        f"requested ({lat:.3f}, {lon:.3f}) — check coordinate order (lat, lon) and NetCDF path:\n"
        f"  {info.get('path', '?')}"
    )


def load_qs_historical_with_partitions(
    coordinate: tuple[float, float],
    shoreline_orientation_deg: float,
    *,
    aggregation: str = "daily",
    time_start=None,
    time_end=None,
    K: float = DEFAULT_LONGSHORE_K,
    historical_folder: str | Path = DEFAULT_HISTORICAL_FOLDER,
    partitions_folder: str | Path = DEFAULT_HISTORICAL_PARTITIONS_FOLDER,
    partition_ids: Sequence[int] | None = None,
    include_bulk: bool = True,
    include_partitions_sum: bool = True,
    project_root: str | Path | None = None,
) -> tuple[dict[str, tuple[xr.DataArray, dict]], str]:
    """
    Historical-only CERC Qs from bulk and wave partitions at one coordinate.

    Returns a ``series`` dict with:
    - ``historical bulk`` (optional)
    - ``partition <id>`` for each available partition
    - ``partitions sum`` (optional; aligned inner-time sum of partition Qs)
    """
    series: dict[str, tuple[xr.DataArray, dict]] = {}
    site_note = ""

    if include_bulk:
        hs_nc = _resolve_variable_nc_path(historical_folder, "hs", project_root)
        dp_nc = _resolve_variable_nc_path(historical_folder, "dp", project_root)
        hs_bulk, info_hs = load_timeseries_at_coordinate(
            hs_nc,
            coordinate,
            variable="hs",
            aggregation=aggregation,
            time_start=time_start,
            time_end=time_end,
            project_root=project_root,
        )
        dp_bulk, _ = load_timeseries_at_coordinate(
            dp_nc,
            coordinate,
            variable="dp",
            aggregation=aggregation,
            time_start=time_start,
            time_end=time_end,
            project_root=project_root,
        )
        hs_a, dp_a = xr.align(hs_bulk, dp_bulk, join="inner")
        if hs_a.sizes.get("time", 0) > 0:
            qs_bulk = longshore_transport_index(hs_a, dp_a, shoreline_orientation_deg, K=K)
            qs_bulk.name = "Qs"
            info_bulk = dict(info_hs)
            info_bulk["n_points"] = int(qs_bulk.sizes.get("time", qs_bulk.size))
            series["historical bulk"] = (qs_bulk, info_bulk)
            site_note = _site_note_from_info(coordinate, info_hs)
        else:
            print("SKIP historical bulk: no overlapping hs/dp times for Qs")

    pids = [int(i) for i in partition_ids] if partition_ids is not None else _partition_ids_from_folder(
        partitions_folder, project_root
    )
    if partition_ids is not None:
        print(f"Using requested partition_ids: {pids}")
    partition_qs: list[xr.DataArray] = []
    for pid in pids:
        hs_name = f"phs{pid}"
        dp_name = f"dp{pid}"
        part_root = resolve_path(partitions_folder, project_root)
        try:
            hs_path = _resolve_variable_nc_path(part_root, hs_name, project_root)
            dp_path = _resolve_variable_nc_path(part_root, dp_name, project_root)
        except FileNotFoundError as exc:
            print(f"SKIP partition {pid}: {exc}")
            continue

        hs_da, info_part = load_timeseries_at_coordinate(
            hs_path,
            coordinate,
            variable=hs_name,
            aggregation=aggregation,
            time_start=time_start,
            time_end=time_end,
            project_root=project_root,
        )
        dp_da, _ = load_timeseries_at_coordinate(
            dp_path,
            coordinate,
            variable=dp_name,
            aggregation=aggregation,
            time_start=time_start,
            time_end=time_end,
            project_root=project_root,
        )
        hs_a, dp_a = xr.align(hs_da, dp_da, join="inner")
        if hs_a.sizes.get("time", 0) == 0:
            print(f"SKIP partition {pid}: no overlapping hs/dp times for Qs")
            continue

        qs_part = longshore_transport_index(hs_a, dp_a, shoreline_orientation_deg, K=K)
        qs_part.name = "Qs"
        info_p = dict(info_part)
        info_p["n_points"] = int(qs_part.sizes.get("time", qs_part.size))
        label = f"partition {pid}"
        series[label] = (qs_part, info_p)
        partition_qs.append(qs_part.rename(label))
        if not site_note:
            site_note = _site_note_from_info(coordinate, info_part)

    if include_partitions_sum and partition_qs:
        aligned = xr.align(*partition_qs, join="inner")
        if aligned and aligned[0].sizes.get("time", 0) > 0:
            summed = aligned[0].copy()
            for da in aligned[1:]:
                summed = summed + da
            summed.name = "Qs"
            sum_info = {
                "site_index": series[next(iter(series))][1]["site_index"],
                "lat": series[next(iter(series))][1]["lat"],
                "lon": series[next(iter(series))][1]["lon"],
                "distance_km": series[next(iter(series))][1]["distance_km"],
                "n_points": int(summed.sizes.get("time", summed.size)),
                "path": str(resolve_path(partitions_folder, project_root)),
                "aggregation": str(aggregation).lower(),
                "nc_var": "sum(partitions)",
            }
            series["partitions sum"] = (summed, sum_info)
        else:
            print("SKIP partitions sum: no overlapping time among partitions")

    # Safety filter: if explicit partition_ids were requested, keep only those
    # partition traces (+ optional bulk and partitions sum).
    if partition_ids is not None:
        allowed = {f"partition {int(i)}" for i in partition_ids}
        keep = {"historical bulk", "partitions sum"} | allowed
        series = {label: payload for label, payload in series.items() if label in keep}

    return series, site_note


_QS_PARTITION_LW = 1
_QS_PARTITION_FIXED_COLORS: dict[int, str] = {0: "olive", 1: "darkviolet"}
_QS_PARTITION_EXTRA_COLORS: tuple[str, ...] = (
    "coral",
    "darkorange",
    "turquoise",
    "gold",
    "magenta",
    "olive",
    "slateblue",
    "chocolate",
    "darkcyan",
    "salmon",
    "peru",
    "steelblue",
)


def _partition_id_from_qs_label(label: str) -> int | None:
    if not label.startswith("partition "):
        return None
    try:
        return int(label.rsplit(maxsplit=1)[-1])
    except ValueError:
        return None


def _qs_partition_color(partition_id: int, fallback_index: int) -> str:
    if partition_id in _QS_PARTITION_FIXED_COLORS:
        return _QS_PARTITION_FIXED_COLORS[partition_id]
    if partition_id >= 2:
        return _QS_PARTITION_EXTRA_COLORS[(partition_id - 2) % len(_QS_PARTITION_EXTRA_COLORS)]
    return _QS_PARTITION_EXTRA_COLORS[fallback_index % len(_QS_PARTITION_EXTRA_COLORS)]


def _scenarios_cfg_for_qs_plot(
    series: Mapping[str, tuple[xr.DataArray, dict]],
    *,
    gcm_scenarios: Sequence[tuple[str, dict]] = (),
    historical_folder: str | Path = DEFAULT_HISTORICAL_FOLDER,
    project_root: str | Path | None = None,
    bulk_series_folders: Mapping[str, str | Path] | None = None,
) -> list[tuple[str, dict]]:
    """Line styles for Qs plots: partitions, historical bulk, GCMs, buoys."""
    from utils import kma_cluster_swan as kcs

    gcm_by_label = {label: cfg for label, cfg in gcm_scenarios}
    hist_path = str(resolve_path(historical_folder, project_root))
    folder_by_label = {
        label: str(resolve_path(folder, project_root))
        for label, folder in (bulk_series_folders or {}).items()
    }
    cfg: list[tuple[str, dict]] = []
    part_i = 0
    for label in series:
        if label in gcm_by_label:
            cfg.append((label, gcm_by_label[label]))
        elif label == BINWAVES_BULK_LABEL:
            cfg.append(
                (
                    label,
                    {
                        "folder": folder_by_label.get(label, hist_path),
                        "color": BINWAVES_OVERLAY_COLOR,
                        "lw": 1.5,
                        "alpha": 0.95,
                        "zorder": 1,
                    },
                )
            )
        elif (
            label == "historical bulk"
            or label == kcs.BINWAVES_PLUS_KMA_LABEL
            or label.strip().lower() == "historical"
        ):
            cfg.append(
                (
                    label,
                    {
                        "folder": folder_by_label.get(label, hist_path),
                        "color": "fuchsia",
                        "lw": 1.5,
                        "alpha": 0.95,
                        "zorder": 1,
                    },
                )
            )
        elif label == "partitions sum":
            cfg.append(
                (label, {"folder": "partition-sum", "color": "dodgerblue", "lw": 1.5, "alpha": 0.9})
            )
        elif label.startswith("partition "):
            pid = _partition_id_from_qs_label(label)
            color = _qs_partition_color(pid if pid is not None else -1, part_i)
            cfg.append(
                (
                    label,
                    {
                        "folder": f"partition-{part_i}",
                        "color": color,
                        "lw": _QS_PARTITION_LW,
                        "alpha": 0.75,
                    },
                )
            )
            part_i += 1
        elif label.startswith("buoy "):
            cfg.append(
                (label, {"folder": "buoy", "color": "k", "lw": 1.5, "ls": "-", "alpha": 0.95, "zorder": 20})
            )
        else:
            cfg.append((label, {"folder": label, "lw": 1.2, "alpha": 0.75}))
    return cfg


def plot_longshore_transport_historical_partitions(
    coordinates: Sequence[float] | Sequence[Sequence[float]],
    shoreline_orientation_deg: float | Sequence[float] | Mapping[tuple[float, float], float],
    *,
    time_start=None,
    time_end=None,
    aggregation: str = "daily",
    K: float = DEFAULT_LONGSHORE_K,
    partition_ids: Sequence[int] | None = None,
    include_bulk: bool = True,
    include_partitions_sum: bool = True,
    interactive: bool = False,
    static: bool = True,
    plot_cumulative: bool = False,
    historical_folder: str | Path = DEFAULT_HISTORICAL_FOLDER,
    partitions_folder: str | Path = DEFAULT_HISTORICAL_PARTITIONS_FOLDER,
    project_root: str | Path | None = None,
) -> None:
    """
    Historical-only Qs plots from bulk + partitioned waves.

    Produces the same Qs/Cumulative-Qs figures as ``plot_longshore_transport`` but overlays:
    - bulk historical Qs
    - one line per partition
    - optional line with sum of all partition Qs
    """
    _warn_native(aggregation)
    coords = normalize_coordinates(coordinates)
    coord_angles = normalize_shoreline_orientations(coords, shoreline_orientation_deg)

    for (lat, lon), shore_deg in coord_angles:
        series, site_note = load_qs_historical_with_partitions(
            (lat, lon),
            shore_deg,
            aggregation=aggregation,
            time_start=time_start,
            time_end=time_end,
            K=K,
            historical_folder=historical_folder,
            partitions_folder=partitions_folder,
            partition_ids=partition_ids,
            include_bulk=include_bulk,
            include_partitions_sum=include_partitions_sum,
            project_root=project_root,
        )
        if not series:
            print(f"No historical/partition Qs series loaded for ({lat:.3f}, {lon:.3f})")
            continue

        scenarios_cfg = _scenarios_cfg_for_qs_plot(
            series,
            historical_folder=historical_folder,
            project_root=project_root,
        )

        title = (
            f"Qs CERC ({aggregation}) @ ({lat:.3f}, {lon:.3f}), "
            f"shore={shore_deg:.1f}° — {site_note}"
        )
        if static:
            plot_timeseries_matplotlib(
                series,
                variable="Qs",
                title=title,
                scenarios_cfg=scenarios_cfg,
                historical_folder=historical_folder,
                project_root=project_root,
            )
            if plot_cumulative:
                cum_series = {
                    label: (da.cumsum("time"), info) for label, (da, info) in series.items()
                }
                plot_timeseries_matplotlib(
                    cum_series,
                    variable="Cumulative Qs",
                    title=f"Cumulative {title}",
                    scenarios_cfg=scenarios_cfg,
                    historical_folder=historical_folder,
                    project_root=project_root,
                )
        if interactive:
            n_pts = sum(info["n_points"] for _da, info in series.values())
            print(f"Interactive historical+partitions Qs: {len(series)} series, ~{n_pts:,} points ({aggregation})")
            plot_timeseries_plotly(
                series,
                variable="Qs",
                title=title,
                scenarios_cfg=scenarios_cfg,
                historical_folder=historical_folder,
                project_root=project_root,
            )


_TIME_HIGHLIGHT_COLORS = ("#f4a6a6", "#a6c8f4", "#a6f4b8", "#f4e0a6", "#d4a6f4", "#f4c2a6")
_DEFAULT_TIME_HIGHLIGHT_ALPHA = 0.35


def _normalize_time_highlights(
    time_highlights: Sequence[Any] | None,
) -> list[dict[str, Any]]:
    """
    Parse ``time_highlights`` into normalized dicts with start, end, color, alpha, label.

    Each entry may be:
    - ``(start, end)`` or ``(start, end, color)`` or ``(start, end, color, label)``
    - ``{"start": ..., "end": ..., "color": ..., "alpha": ..., "label": ...}``
    """
    if not time_highlights:
        return []
    out: list[dict[str, Any]] = []
    for i, item in enumerate(time_highlights):
        if isinstance(item, Mapping):
            start = item.get("start", item.get("time_start"))
            end = item.get("end", item.get("time_end"))
            if start is None or end is None:
                raise ValueError(
                    f"time_highlights[{i}] needs 'start' and 'end' (or 'time_start'/'time_end'): {item!r}"
                )
            color = item.get("color", _TIME_HIGHLIGHT_COLORS[i % len(_TIME_HIGHLIGHT_COLORS)])
            alpha = float(item.get("alpha", _DEFAULT_TIME_HIGHLIGHT_ALPHA))
            label = item.get("label")
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            start, end = item[0], item[1]
            color = (
                item[2]
                if len(item) >= 3 and not isinstance(item[2], (int, float))
                else _TIME_HIGHLIGHT_COLORS[i % len(_TIME_HIGHLIGHT_COLORS)]
            )
            label = item[3] if len(item) >= 4 else None
            alpha = (
                float(item[2])
                if len(item) >= 3 and isinstance(item[2], (int, float))
                else _DEFAULT_TIME_HIGHLIGHT_ALPHA
            )
        else:
            raise ValueError(
                f"time_highlights[{i}] must be a mapping or (start, end[, color[, label]]) tuple; got {item!r}"
            )
        t0 = pd.Timestamp(start)
        t1 = pd.Timestamp(end)
        if t1 < t0:
            t0, t1 = t1, t0
        out.append(
            {
                "start": t0,
                "end": t1,
                "color": color,
                "alpha": alpha,
                "label": label,
            }
        )
    return out


def _apply_time_highlights_matplotlib(axes, highlights: Sequence[Mapping[str, Any]]) -> None:
    """Shade vertical time ranges on every subplot (matplotlib).

    Draws filled spans plus dashed boundary lines and top labels so short events
    remain visible on multi-decade axes.
    """
    if not highlights:
        return
    from matplotlib.patches import Patch

    axes_flat = np.atleast_1d(axes).ravel()
    for hl in highlights:
        t0, t1 = hl["start"], hl["end"]
        xmid = t0 + (t1 - t0) / 2
        for ax in axes_flat:
            ax.axvspan(
                t0,
                t1,
                color=hl["color"],
                alpha=hl["alpha"],
                zorder=0,
                linewidth=0,
            )
            ax.axvline(t0, color=hl["color"], lw=1.4, alpha=0.9, zorder=2, ls="--")
            ax.axvline(t1, color=hl["color"], lw=1.4, alpha=0.9, zorder=2, ls="--")
            if hl.get("label"):
                ax.text(
                    xmid,
                    0.97,
                    hl["label"],
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=7,
                    color=hl["color"],
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        alpha=0.8,
                        edgecolor=hl["color"],
                    ),
                    zorder=5,
                    clip_on=False,
                )

    highlight_handles = [
        Patch(facecolor=hl["color"], alpha=hl["alpha"], label=hl["label"])
        for hl in highlights
        if hl.get("label")
    ]
    if not highlight_handles:
        return

    ax0 = axes_flat[0]
    existing_legend = ax0.get_legend()
    if existing_legend is not None:
        handles = list(existing_legend.legend_handles)
        labels = [t.get_text() for t in existing_legend.get_texts()]
        ax0.legend(
            handles=handles + highlight_handles,
            labels=labels + [h.get_label() for h in highlight_handles],
            loc="upper left",
            fontsize=7,
            ncol=2,
        )
    else:
        ax0.legend(handles=highlight_handles, loc="upper left", fontsize=7)


def _apply_time_highlights_plotly(
    fig,
    highlights: Sequence[Mapping[str, Any]],
    *,
    n_rows: int,
    n_cols: int,
) -> None:
    """Shade vertical time ranges on every subplot (Plotly)."""
    if not highlights:
        return
    for hl in highlights:
        xmid = hl["start"] + (hl["end"] - hl["start"]) / 2
        for r in range(1, n_rows + 1):
            for c in range(1, n_cols + 1):
                fig.add_vrect(
                    x0=hl["start"],
                    x1=hl["end"],
                    fillcolor=hl["color"],
                    opacity=hl["alpha"],
                    layer="below",
                    line_width=0,
                    row=r,
                    col=c,
                )
                fig.add_vline(
                    x=hl["start"],
                    line_color=hl["color"],
                    line_width=2,
                    line_dash="dash",
                    opacity=0.9,
                    row=r,
                    col=c,
                )
                fig.add_vline(
                    x=hl["end"],
                    line_color=hl["color"],
                    line_width=2,
                    line_dash="dash",
                    opacity=0.9,
                    row=r,
                    col=c,
                )
                if hl.get("label") and r == 1 and c == 1 and n_rows == 1 and n_cols == 1:
                    fig.add_annotation(
                        x=xmid,
                        y=1.03,
                        yref="paper",
                        text=hl["label"],
                        showarrow=False,
                        font=dict(color=hl["color"], size=9),
                        xanchor="center",
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor=hl["color"],
                        borderwidth=1,
                    )


def _plot_timeseries_on_ax(
    ax,
    times,
    values,
    *,
    directional: bool,
    label: str | None = None,
    **style,
) -> None:
    """Scalar series as lines; directional (Dp, etc.) as dot markers without connecting lines."""
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


def _plot_partition_hs_tp_dp_grid(
    bundles: dict[str, dict[str, tuple[xr.DataArray, dict]]],
    *,
    title: str,
    partition_ids: Sequence[int],
    include_bulk: bool,
    orientation: Literal["rows_are_partitions", "rows_are_variables"] = "rows_are_variables",
    include_sediment_transport_row: bool = False,
    shoreline_orientation_deg: float | None = None,
    K: float = DEFAULT_LONGSHORE_K,
    qs_formula: str = "cerc",
    overlay_buoy_on_bulk: bool = False,
    overlay_swan_on_bulk: bool = True,
    swan_bulk_label: str | None = None,
    axis_limits: Mapping[str, tuple[float, float]] | None = None,
    time_highlights: Sequence[Any] | None = None,
) -> None:
    """
    Grid layout for partition variables.

    - ``rows_are_partitions``: columns = Hs/Tp/Dp, rows = bulk + partitions.
    - ``rows_are_variables``: columns = bulk + partitions, rows = Hs/Tp/Dp
      (+ optional Qs row on top).
    - ``overlay_buoy_on_bulk``: plot NDBC buoy on the same axes as ``historical bulk``.
    - ``overlay_swan_on_bulk``: plot SWAN bulk on the same axes as ``historical bulk``.
    """
    variables = ("hs", "tp", "dp")
    highlights = _normalize_time_highlights(time_highlights)
    buoy_overlay_label = _primary_buoy_label(bundles) if overlay_buoy_on_bulk else None
    swan_label = _resolve_swan_overlay_label(bundles, swan_bulk_label)
    swan_overlay = overlay_swan_on_bulk and swan_label is not None
    binwaves_ref_overlay = _binwaves_reference_in_bundles(bundles, variables)
    if orientation == "rows_are_partitions":
        col_titles = ("Hs", "Tp", "Dp (°)")
        rows: list[tuple[str, str]] = []
        if include_bulk:
            for bulk_key in _bulk_series_keys_in_bundles(bundles, variables):
                row_title = _bulk_column_title(
                    bundles,
                    bulk_key=bulk_key,
                    overlay_buoy_label=buoy_overlay_label if _is_overlay_bulk_key(bulk_key) else None,
                    overlay_swan_on_bulk=swan_overlay and _is_overlay_bulk_key(bulk_key),
                )
                rows.append((row_title, bulk_key))
        if not overlay_buoy_on_bulk:
            for buoy_key in _buoy_labels_in_bundles(bundles):
                rows.append((buoy_key.replace("buoy ", "buoy "), buoy_key))
        for pid in partition_ids:
            label = f"partition {pid}"
            if any(label in bundles[v] for v in variables):
                rows.append((label, label))
        if not rows:
            return

        n_rows = len(rows)
        fig, axes = plt.subplots(
            n_rows,
            3,
            figsize=(14, max(2.2 * n_rows, 6)),
            sharex=True,
            squeeze=False,
        )
        for r, (row_title, series_key) in enumerate(rows):
            overlay_label = buoy_overlay_label if _is_overlay_bulk_key(series_key) else None
            style, _ = _partition_series_key_style(
                series_key, r, overlay_buoy_label=overlay_label, overlay_swan_on_bulk=swan_overlay, bundles=bundles
            )
            for c, var in enumerate(variables):
                ax = axes[r, c]
                panel = bundles[var]
                if series_key in panel:
                    da, _info = panel[series_key]
                    _plot_timeseries_on_ax(
                        ax,
                        da["time"].values,
                        da.values,
                        directional=_is_directional_name(var),
                        **style,
                    )
                    if overlay_label and _is_overlay_bulk_key(series_key):
                        _plot_buoy_overlay_on_ax(ax, bundles, overlay_label, var)
                    if swan_overlay and _is_overlay_bulk_key(series_key):
                        _plot_swan_overlay_on_ax(ax, bundles, var, swan_label=swan_label)
                    if binwaves_ref_overlay and _is_overlay_bulk_key(series_key):
                        _plot_binwaves_reference_overlay_on_ax(ax, bundles, var)
                else:
                    ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
                ax.grid(True, alpha=0.3)
                if r == 0:
                    ax.set_title(col_titles[c])
                if c == 0:
                    ax.set_ylabel(row_title, fontsize=9)
                if r == n_rows - 1:
                    ax.set_xlabel("Time")

        hs_limits = axis_limits.get("hs") if axis_limits else None
        if hs_limits is not None:
            for r in range(n_rows):
                axes[r, 0].set_ylim(*hs_limits)
        else:
            hs_panel = bundles["hs"]
            bulk_keys = _bulk_series_keys_in_bundles(bundles, variables)
            hs_keys = list(bulk_keys)
            if binwaves_ref_overlay and BINWAVES_BULK_LABEL in hs_panel:
                hs_keys.append(BINWAVES_BULK_LABEL)
            if hs_keys:
                hs_ymax = float(
                    np.nanmax(
                        [np.nanmax(hs_panel[k][0].values) for k in hs_keys if k in hs_panel]
                    )
                )
            elif hs_panel:
                hs_ymax = float(np.nanmax([np.nanmax(da.values) for da, _ in hs_panel.values()]))
            else:
                hs_ymax = None
            if hs_ymax is not None and np.isfinite(hs_ymax):
                for r in range(n_rows):
                    axes[r, 0].set_ylim(0, hs_ymax)
        tp_limits = axis_limits.get("tp") if axis_limits else None
        for r in range(n_rows):
            axes[r, 1].set_ylim(*(tp_limits if tp_limits is not None else (0, 20)))

        _apply_time_highlights_matplotlib(axes, highlights)
        fig.suptitle(title, y=1.01, fontsize=11)
        plt.tight_layout()
        plt.show()
        return

    series_keys = _partition_hs_tp_dp_series_keys(
        bundles,
        partition_ids,
        include_bulk=include_bulk,
        overlay_buoy_on_bulk=overlay_buoy_on_bulk,
    )
    if not series_keys:
        return

    overlay_bulk = _overlay_bulk_key(series_keys)

    qs_keys = list(series_keys)
    if overlay_buoy_on_bulk and buoy_overlay_label and buoy_overlay_label not in qs_keys:
        qs_keys.append(buoy_overlay_label)
    if swan_overlay and swan_label not in qs_keys:
        qs_keys.append(swan_label)
    if binwaves_ref_overlay and BINWAVES_BULK_LABEL not in qs_keys:
        qs_keys.append(BINWAVES_BULK_LABEL)

    qs_by_key: dict[str, xr.DataArray] = {}
    if include_sediment_transport_row and shoreline_orientation_deg is not None:
        for key in qs_keys:
            qs = _qs_timeseries_for_bundle_key(
                bundles,
                key,
                shoreline_orientation_deg,
                K=K,
                qs_formula=qs_formula,
            )
            if qs is not None:
                qs_by_key[key] = qs

    row_vars = ["hs", "tp", "dp"]
    if include_sediment_transport_row and qs_by_key:
        row_vars = ["qs", "hs", "tp", "dp"]
    qs_label = "Qs (deep)" if str(qs_formula).lower() == "deep" else "Qs"
    row_titles = {"qs": qs_label, "hs": "Hs (m)", "tp": "Tp (s)", "dp": "Dp (°)"}

    n_rows = len(row_vars)
    n_cols = len(series_keys)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(14, 2.8 * n_cols), max(7, 2.2 * n_rows)),
        sharex=True,
        squeeze=False,
    )
    for c, key in enumerate(series_keys):
        overlay_label = buoy_overlay_label if key == overlay_bulk else None
        style, col_title = _partition_series_key_style(
            key,
            c,
            overlay_buoy_label=overlay_label,
            overlay_swan_on_bulk=swan_overlay,
            bundles=bundles,
        )
        for r, var in enumerate(row_vars):
            ax = axes[r, c]
            if var == "qs":
                da = qs_by_key.get(key)
                if da is not None:
                    _plot_timeseries_on_ax(ax, da["time"].values, da.values, directional=False, **style)
                if overlay_label and overlay_label in qs_by_key:
                    _plot_buoy_overlay_on_ax(
                        ax, bundles, overlay_label, var, qs_da=qs_by_key[overlay_label]
                    )
                if swan_overlay and key == overlay_bulk and swan_label in qs_by_key:
                    _plot_swan_overlay_on_ax(
                        ax, bundles, var, qs_da=qs_by_key[swan_label], swan_label=swan_label
                    )
                if binwaves_ref_overlay and key == overlay_bulk and BINWAVES_BULK_LABEL in qs_by_key:
                    _plot_binwaves_reference_overlay_on_ax(
                        ax, bundles, var, qs_da=qs_by_key[BINWAVES_BULK_LABEL]
                    )
                elif da is None:
                    ax.text(0.5, 0.5, "no Qs", transform=ax.transAxes, ha="center", va="center")
            else:
                panel = bundles[var]
                if key in panel:
                    da, _info = panel[key]
                    _plot_timeseries_on_ax(
                        ax, da["time"].values, da.values, directional=_is_directional_name(var), **style
                    )
                    if overlay_label and key == overlay_bulk:
                        _plot_buoy_overlay_on_ax(ax, bundles, overlay_label, var)
                    if swan_overlay and key == overlay_bulk:
                        _plot_swan_overlay_on_ax(ax, bundles, var, swan_label=swan_label)
                    if binwaves_ref_overlay and key == overlay_bulk:
                        _plot_binwaves_reference_overlay_on_ax(ax, bundles, var)
                else:
                    ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
            ax.grid(True, alpha=0.3)
            if r == 0:
                ax.set_title(col_title)
            if c == 0:
                ax.set_ylabel(row_titles[var], fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel("Time")

    if "hs" in row_vars:
        hs_row = row_vars.index("hs")
        hs_limits = axis_limits.get("hs") if axis_limits else None
        if hs_limits is not None:
            for c in range(n_cols):
                axes[hs_row, c].set_ylim(*hs_limits)
        else:
            hs_vals: list[np.ndarray] = []
            hs_panel = bundles["hs"]
            for da, _ in hs_panel.values():
                v = np.asarray(da.values, dtype=float)
                v = v[np.isfinite(v)]
                if v.size:
                    hs_vals.append(v)
            if hs_vals:
                hs_ymax = float(np.nanmax(np.concatenate(hs_vals)))
                for c in range(n_cols):
                    axes[hs_row, c].set_ylim(0, hs_ymax)
    if "tp" in row_vars:
        tp_row = row_vars.index("tp")
        tp_limits = axis_limits.get("tp") if axis_limits else None
        for c in range(n_cols):
            axes[tp_row, c].set_ylim(*(tp_limits if tp_limits is not None else (0, 20)))
    if "dp" in row_vars and axis_limits and "dp" in axis_limits:
        dp_row = row_vars.index("dp")
        for c in range(n_cols):
            axes[dp_row, c].set_ylim(*axis_limits["dp"])
    if "qs" in row_vars and axis_limits and "qs" in axis_limits:
        qs_row = row_vars.index("qs")
        for c in range(n_cols):
            axes[qs_row, c].set_ylim(*axis_limits["qs"])

    if (buoy_overlay_label or swan_overlay or binwaves_ref_overlay) and n_cols > 0 and overlay_bulk is not None:
        bulk_col = series_keys.index(overlay_bulk)
        from utils import kma_cluster_swan as kcs

        if overlay_bulk == kcs.BINWAVES_PLUS_KMA_LABEL:
            axes[0, bulk_col].plot([], [], color=SWAN_WIND_OVERLAY_COLOR, lw=1.2, label=kcs.BINWAVES_PLUS_KMA_LABEL)
        else:
            axes[0, bulk_col].plot([], [], color=BINWAVES_OVERLAY_COLOR, lw=1.2, label="BinWaves")
        if binwaves_ref_overlay:
            axes[0, bulk_col].plot([], [], color=BINWAVES_OVERLAY_COLOR, lw=1.0, ls="--", label=BINWAVES_BULK_LABEL)
        if buoy_overlay_label:
            axes[0, bulk_col].plot([], [], color=BUOY_OVERLAY_COLOR, lw=1.0, label="buoy")
        if swan_overlay:
            axes[0, bulk_col].plot([], [], color=SWAN_WIND_OVERLAY_COLOR, lw=1.1, label="SWAN_WIND")
        axes[0, bulk_col].legend(loc="upper right", fontsize=7)

    _apply_time_highlights_matplotlib(axes, highlights)
    fig.suptitle(title, y=1.01, fontsize=11)
    plt.tight_layout()
    plt.show()


def _plot_partition_hs_tp_dp_grid_plotly(
    bundles: dict[str, dict[str, tuple[xr.DataArray, dict]]],
    *,
    title: str,
    partition_ids: Sequence[int],
    include_bulk: bool,
    orientation: Literal["rows_are_partitions", "rows_are_variables"] = "rows_are_variables",
    include_sediment_transport_row: bool = False,
    shoreline_orientation_deg: float | None = None,
    K: float = DEFAULT_LONGSHORE_K,
    qs_formula: str = "cerc",
    overlay_buoy_on_bulk: bool = False,
    overlay_swan_on_bulk: bool = True,
    swan_bulk_label: str | None = None,
    axis_limits: Mapping[str, tuple[float, float]] | None = None,
    time_highlights: Sequence[Any] | None = None,
) -> None:
    """Interactive Plotly version of partition Hs/Tp/Dp grid."""
    if not PLOTLY_AVAILABLE:
        print("Interactive plotting requires plotly. Install plotly or use static=True.")
        return
    if make_subplots is None:
        print("Interactive plotting unavailable: plotly.subplots not found.")
        return

    highlights = _normalize_time_highlights(time_highlights)
    variables = ("hs", "tp", "dp")
    buoy_overlay_label = _primary_buoy_label(bundles) if overlay_buoy_on_bulk else None
    swan_label = _resolve_swan_overlay_label(bundles, swan_bulk_label)
    swan_overlay = overlay_swan_on_bulk and swan_label is not None
    binwaves_ref_overlay = _binwaves_reference_in_bundles(bundles, variables)
    series_keys = _partition_hs_tp_dp_series_keys(
        bundles,
        partition_ids,
        include_bulk=include_bulk,
        overlay_buoy_on_bulk=overlay_buoy_on_bulk,
    )
    if not series_keys:
        return

    overlay_bulk = _overlay_bulk_key(series_keys)

    qs_keys = list(series_keys)
    if overlay_buoy_on_bulk and buoy_overlay_label and buoy_overlay_label not in qs_keys:
        qs_keys.append(buoy_overlay_label)
    if swan_overlay and swan_label not in qs_keys:
        qs_keys.append(swan_label)
    if binwaves_ref_overlay and BINWAVES_BULK_LABEL not in qs_keys:
        qs_keys.append(BINWAVES_BULK_LABEL)

    qs_by_key: dict[str, xr.DataArray] = {}
    if include_sediment_transport_row and shoreline_orientation_deg is not None:
        for key in qs_keys:
            qs = _qs_timeseries_for_bundle_key(
                bundles,
                key,
                shoreline_orientation_deg,
                K=K,
                qs_formula=qs_formula,
            )
            if qs is not None:
                qs_by_key[key] = qs

    # Build panel structure matching static behavior.
    if orientation == "rows_are_partitions":
        row_defs: list[tuple[str, str]] = []
        if include_bulk:
            for bulk_key in _bulk_series_keys_in_bundles(bundles, variables):
                row_defs.append(
                    (
                        _bulk_column_title(
                            bundles,
                            bulk_key=bulk_key,
                            overlay_buoy_label=buoy_overlay_label if _is_overlay_bulk_key(bulk_key) else None,
                            overlay_swan_on_bulk=swan_overlay and _is_overlay_bulk_key(bulk_key),
                        ),
                        bulk_key,
                    )
                )
        for pid in partition_ids:
            label = f"partition {pid}"
            if any(label in bundles[v] for v in variables):
                row_defs.append((label, label))
        if not row_defs:
            return
        row_vars = ["hs", "tp", "dp"]
        n_rows = len(row_defs)
        n_cols = len(row_vars)
        subplot_titles = []
        for row_title, _ in row_defs:
            for var in row_vars:
                subplot_titles.append(f"{row_title} - {var.upper()}")
        fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=True, subplot_titles=subplot_titles)

        for r, (_row_title, series_key) in enumerate(row_defs, start=1):
            from utils import kma_cluster_swan as kcs

            pid = _partition_id_from_qs_label(series_key)
            if series_key == kcs.BINWAVES_PLUS_KMA_LABEL:
                color = SWAN_WIND_OVERLAY_COLOR_PLOTLY
            elif series_key == BINWAVES_BULK_LABEL or series_key == "historical bulk":
                color = BINWAVES_OVERLAY_COLOR_PLOTLY
            elif series_key == swan_label:
                color = SWAN_WIND_OVERLAY_COLOR_PLOTLY
            elif pid is not None:
                color = _to_plotly_color(_qs_partition_color(pid, r - 1))
            else:
                color = "rgb(50,100,180)"
            for c, var in enumerate(row_vars, start=1):
                panel = bundles[var]
                if series_key not in panel:
                    continue
                da, _info = panel[series_key]
                directional = _is_directional_name(var)
                fig.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(da["time"].values),
                        y=da.values,
                        mode="markers" if directional else "lines",
                        marker=dict(size=3, color=color) if directional else None,
                        line=dict(color=color, width=1.2) if not directional else None,
                        name=f"{series_key} {var}",
                        showlegend=False,
                    ),
                    row=r,
                    col=c,
                )
                if _is_overlay_bulk_key(series_key) and buoy_overlay_label and buoy_overlay_label in panel:
                    bda, _ = panel[buoy_overlay_label]
                    fig.add_trace(
                        go.Scatter(
                            x=pd.to_datetime(bda["time"].values),
                            y=bda.values,
                            mode="markers" if directional else "lines",
                            marker=dict(size=3, color=BUOY_OVERLAY_COLOR_PLOTLY) if directional else None,
                            line=dict(color=BUOY_OVERLAY_COLOR_PLOTLY, width=1.0) if not directional else None,
                            name=f"{buoy_overlay_label} {var}",
                            showlegend=False,
                        ),
                        row=r,
                        col=c,
                    )
                if _is_overlay_bulk_key(series_key) and swan_overlay and swan_label in panel:
                    sda, _ = panel[swan_label]
                    fig.add_trace(
                        go.Scatter(
                            x=pd.to_datetime(sda["time"].values),
                            y=sda.values,
                            mode="markers" if directional else "lines",
                            marker=dict(size=3, color=SWAN_WIND_OVERLAY_COLOR_PLOTLY) if directional else None,
                            line=dict(color=SWAN_WIND_OVERLAY_COLOR_PLOTLY, width=1.1) if not directional else None,
                            name=f"SWAN_WIND {var}",
                            showlegend=False,
                        ),
                        row=r,
                        col=c,
                    )
                if binwaves_ref_overlay and _is_overlay_bulk_key(series_key) and BINWAVES_BULK_LABEL in panel:
                    bda, _ = panel[BINWAVES_BULK_LABEL]
                    fig.add_trace(
                        go.Scatter(
                            x=pd.to_datetime(bda["time"].values),
                            y=bda.values,
                            mode="markers" if directional else "lines",
                            marker=dict(size=3, color=BINWAVES_OVERLAY_COLOR_PLOTLY) if directional else None,
                            line=dict(color=BINWAVES_OVERLAY_COLOR_PLOTLY, width=1.0, dash="dash") if not directional else None,
                            name=f"{BINWAVES_BULK_LABEL} {var}",
                            showlegend=False,
                        ),
                        row=r,
                        col=c,
                    )
        _apply_time_highlights_plotly(fig, highlights, n_rows=n_rows, n_cols=n_cols)
        fig.update_layout(height=max(420, 220 * n_rows), width=max(900, 320 * n_cols), title=title)
        fig.show()
        return

    row_vars = ["hs", "tp", "dp"]
    if include_sediment_transport_row and qs_by_key:
        row_vars = ["qs", "hs", "tp", "dp"]
    n_rows = len(row_vars)
    n_cols = len(series_keys)
    from utils import kma_cluster_swan as kcs

    subplot_titles = []
    for var in row_vars:
        for key in series_keys:
            if _is_overlay_bulk_key(key):
                short_key = _bulk_column_title(
                    bundles,
                    bulk_key=key,
                    overlay_buoy_label=buoy_overlay_label,
                    overlay_swan_on_bulk=swan_overlay,
                )
            elif key.startswith("buoy "):
                short_key = key.replace("buoy ", "buoy ")
            else:
                short_key = f"h{_partition_id_from_qs_label(key)}"
            subplot_titles.append(f"{var.upper()} - {short_key}")
    fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=True, subplot_titles=subplot_titles)

    for c, key in enumerate(series_keys, start=1):
        pid = _partition_id_from_qs_label(key)
        if key == kcs.BINWAVES_PLUS_KMA_LABEL:
            color = SWAN_WIND_OVERLAY_COLOR_PLOTLY
        elif key == BINWAVES_BULK_LABEL or key == "historical bulk":
            color = BINWAVES_OVERLAY_COLOR_PLOTLY
        elif key == swan_label:
            color = SWAN_WIND_OVERLAY_COLOR_PLOTLY
        elif key.startswith("buoy "):
            color = BUOY_OVERLAY_COLOR_PLOTLY
        elif pid is not None:
            color = _to_plotly_color(_qs_partition_color(pid, c - 1))
        else:
            color = "rgb(50,100,180)"
        overlay_label = buoy_overlay_label if key == overlay_bulk else None
        for r, var in enumerate(row_vars, start=1):
            if var == "qs":
                da = qs_by_key.get(key)
                if da is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=pd.to_datetime(da["time"].values),
                            y=da.values,
                            mode="lines",
                            line=dict(color=color, width=1.2),
                            name=f"{key} Qs hindcast",
                            showlegend=False,
                        ),
                        row=r,
                        col=c,
                    )
                if overlay_label and overlay_label in qs_by_key:
                    bqs = qs_by_key[overlay_label]
                    fig.add_trace(
                        go.Scatter(
                            x=pd.to_datetime(bqs["time"].values),
                            y=bqs.values,
                            mode="lines",
                            line=dict(color=BUOY_OVERLAY_COLOR_PLOTLY, width=1.0),
                            name=f"{overlay_label} Qs buoy",
                            showlegend=False,
                        ),
                        row=r,
                        col=c,
                    )
                if swan_overlay and key == overlay_bulk and swan_label in qs_by_key:
                    sqs = qs_by_key[swan_label]
                    fig.add_trace(
                        go.Scatter(
                            x=pd.to_datetime(sqs["time"].values),
                            y=sqs.values,
                            mode="lines",
                            line=dict(color=SWAN_WIND_OVERLAY_COLOR_PLOTLY, width=1.1),
                            name="SWAN_WIND Qs",
                            showlegend=False,
                        ),
                        row=r,
                        col=c,
                    )
                if binwaves_ref_overlay and key == overlay_bulk and BINWAVES_BULK_LABEL in qs_by_key:
                    bqs = qs_by_key[BINWAVES_BULK_LABEL]
                    fig.add_trace(
                        go.Scatter(
                            x=pd.to_datetime(bqs["time"].values),
                            y=bqs.values,
                            mode="lines",
                            line=dict(color=BINWAVES_OVERLAY_COLOR_PLOTLY, width=1.0, dash="dash"),
                            name=f"{BINWAVES_BULK_LABEL} Qs",
                            showlegend=False,
                        ),
                        row=r,
                        col=c,
                    )
                continue
            panel = bundles[var]
            if key not in panel:
                continue
            da, _info = panel[key]
            directional = _is_directional_name(var)
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(da["time"].values),
                    y=da.values,
                    mode="markers" if directional else "lines",
                    marker=dict(size=3, color=color) if directional else None,
                    line=dict(color=color, width=1.2) if not directional else None,
                    name=f"{key} {var} hindcast",
                    showlegend=False,
                ),
                row=r,
                col=c,
            )
            if overlay_label and overlay_label in panel:
                bda, _ = panel[overlay_label]
                fig.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(bda["time"].values),
                        y=bda.values,
                        mode="markers" if directional else "lines",
                        marker=dict(size=3, color=BUOY_OVERLAY_COLOR_PLOTLY) if directional else None,
                        line=dict(color=BUOY_OVERLAY_COLOR_PLOTLY, width=1.0) if not directional else None,
                        name=f"{overlay_label} {var} buoy",
                        showlegend=False,
                    ),
                    row=r,
                    col=c,
                )
            if swan_overlay and key == overlay_bulk and swan_label in panel:
                sda, _ = panel[swan_label]
                fig.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(sda["time"].values),
                        y=sda.values,
                        mode="markers" if directional else "lines",
                        marker=dict(size=3, color=SWAN_WIND_OVERLAY_COLOR_PLOTLY) if directional else None,
                        line=dict(color=SWAN_WIND_OVERLAY_COLOR_PLOTLY, width=1.1) if not directional else None,
                        name=f"SWAN_WIND {var}",
                        showlegend=False,
                    ),
                    row=r,
                    col=c,
                )
            if binwaves_ref_overlay and key == overlay_bulk and BINWAVES_BULK_LABEL in panel:
                bda, _ = panel[BINWAVES_BULK_LABEL]
                fig.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(bda["time"].values),
                        y=bda.values,
                        mode="markers" if directional else "lines",
                        marker=dict(size=3, color=BINWAVES_OVERLAY_COLOR_PLOTLY) if directional else None,
                        line=dict(color=BINWAVES_OVERLAY_COLOR_PLOTLY, width=1.0, dash="dash") if not directional else None,
                        name=f"{BINWAVES_BULK_LABEL} {var}",
                        showlegend=False,
                    ),
                    row=r,
                    col=c,
                )

    # Match static y-axis conventions where relevant.
    if "tp" in row_vars:
        tp_row = row_vars.index("tp") + 1
        tp_limits = axis_limits.get("tp") if axis_limits else None
        for c in range(1, n_cols + 1):
            fig.update_yaxes(range=list(tp_limits if tp_limits is not None else (0, 20)), row=tp_row, col=c)
    if "hs" in row_vars:
        hs_row = row_vars.index("hs") + 1
        hs_limits = axis_limits.get("hs") if axis_limits else None
        if hs_limits is not None:
            for c in range(1, n_cols + 1):
                fig.update_yaxes(range=list(hs_limits), row=hs_row, col=c)
        else:
            hs_panel = bundles["hs"]
            bulk_keys = _bulk_series_keys_in_bundles(bundles, variables)
            if bulk_keys:
                hs_ymax = float(
                    np.nanmax(
                        [np.nanmax(hs_panel[k][0].values) for k in bulk_keys if k in hs_panel]
                    )
                )
            elif hs_panel:
                hs_ymax = float(np.nanmax([np.nanmax(da.values) for da, _ in hs_panel.values()]))
            else:
                hs_ymax = None
            if hs_ymax is not None and np.isfinite(hs_ymax):
                for c in range(1, n_cols + 1):
                    fig.update_yaxes(range=[0, hs_ymax], row=hs_row, col=c)
    if "dp" in row_vars and axis_limits and "dp" in axis_limits:
        dp_row = row_vars.index("dp") + 1
        for c in range(1, n_cols + 1):
            fig.update_yaxes(range=list(axis_limits["dp"]), row=dp_row, col=c)
    if "qs" in row_vars and axis_limits and "qs" in axis_limits:
        qs_row = row_vars.index("qs") + 1
        for c in range(1, n_cols + 1):
            fig.update_yaxes(range=list(axis_limits["qs"]), row=qs_row, col=c)

    _apply_time_highlights_plotly(fig, highlights, n_rows=n_rows, n_cols=n_cols)
    fig.update_layout(
        title=title,
        height=max(450, 220 * n_rows),
        width=max(1000, 300 * n_cols),
    )
    fig.show()


def plot_partition_hs_tp_dp_timeseries(
    coordinates: Sequence[float] | Sequence[Sequence[float]],
    *,
    partition_ids: Sequence[int] | None = None,
    time_start=None,
    time_end=None,
    aggregation: str = "daily",
    include_bulk: bool = True,
    layout: Literal["grid", "overlay"] = "grid",
    grid_orientation: Literal["rows_are_partitions", "rows_are_variables"] = "rows_are_variables",
    include_sediment_transport_row: bool = False,
    shoreline_orientation_deg: float | Sequence[float] | Mapping[tuple[float, float], float] | None = None,
    K: float = DEFAULT_LONGSHORE_K,
    qs_formula: str = "cerc",
    include_buoy: bool = False,
    overlay_buoy_on_bulk: bool = True,
    buoy_id: str | None = None,
    buoys: Mapping[str, Any] | None = None,
    buoy_data_dir: str | Path = DEFAULT_BUOY_DATA_DIR,
    buoy_max_distance_km: float | None = 75.0,
    static: bool = True,
    interactive: bool = False,
    bulk_axis_scope: Literal["per_figure", "all_figures"] = "all_figures",
    time_highlights: Sequence[Any] | None = None,
    swan_timeseries_path: str | Path | None = None,
    swan_timeseries_folder: str | Path | None = None,
    overlay_swan_on_bulk: bool = True,
    swan_bulk_label: str | None = None,
    partitions_folder: str | Path | None = None,
    partitions_dataset: HistoricalDataset = "merged_grids",
    historical_folder: str | Path | None = None,
    historical_dataset: HistoricalDataset = "merged_grids",
    include_binwaves_reference: bool | None = None,
    binwaves_reference_folder: str | Path | None = None,
    project_root: str | Path | None = None,
) -> None:
    """
    Plot partitioned Hs/Tp/Dp time series for historical data at given coordinates.

    ``time_highlights``: optional shaded vertical bands on every subplot. Each entry is
    ``(start, end)``, ``(start, end, color)``, ``(start, end, color, label)``, or a dict with
    ``start``/``end`` (or ``time_start``/``time_end``), optional ``color``, ``alpha``, ``label``.

    ``qs_formula``: ``\"cerc\"`` (default) or ``\"deep\"`` for the optional Qs row
    (``include_sediment_transport_row=True``; deep requires Tp in each column).

    ``include_buoy=True``: load NDBC buoy from ``buoy_<id>_bulk_parameters.pkl`` when
    available. Pass ``buoy_id`` or nearest entry in ``buoys``.

    ``overlay_buoy_on_bulk=True`` (default when using buoy): plot buoy and BinWaves bulk
    on the **same** column (fuchsia = BinWaves, black = buoy). Set ``False`` for a separate
    buoy column.

    **layout=\"grid\"** (default): grid plot. ``grid_orientation`` controls arrangement:
    - ``\"rows_are_variables\"`` (default): columns = bulk, h0, h1, ... and rows = Hs/Tp/Dp.
      Optionally prepend sediment transport row with ``include_sediment_transport_row=True``.
    - ``\"rows_are_partitions\"``: legacy arrangement (rows are partitions, cols Hs/Tp/Dp).

    **layout=\"overlay\"**: legacy 3 stacked panels with all partitions on each panel.

    **Data folders** (default ``merged_grids``):
    - Partitions: ``{partitions_folder}/phs<i>_merged_all.nc``, ``ptp<i>_…``, ``dp<i>_…``
    - Bulk: ``hs_merged_all.nc``, ``tp_merged_all.nc``, ``dp_merged_all.nc`` in ``historical_folder``
    - SWAN bulk (optional): point file with ``Hsig``/``Tps``/``pdir`` via
      ``swan_timeseries_path`` or auto-resolve from ``swan_timeseries_folder``.
      By default overlaid on the BinWaves bulk column (turquoise line).

    Coordinates may be ``(lat, lon)`` or ``(lon, lat)`` (auto-detected for US East Coast).
    """
    if not static and not interactive:
        print("Enable at least one output mode: static=True and/or interactive=True.")
        return
    if interactive and not PLOTLY_AVAILABLE:
        print("Interactive plotting requested but plotly is not available. Use static=True or install plotly.")
        interactive = False
    if not static and not interactive:
        return

    _warn_native(aggregation)
    qs_formula = str(qs_formula).lower()
    if qs_formula not in ("cerc", "deep"):
        raise ValueError(f"qs_formula must be 'cerc' or 'deep'; got {qs_formula!r}")

    coords = normalize_coordinates(coordinates)
    part_folder = resolve_partitions_folder(
        partitions_folder=partitions_folder,
        partitions_dataset=partitions_dataset,
        project_root=project_root,
    )
    hist_folder = resolve_historical_folder(
        historical_folder=historical_folder,
        historical_dataset=historical_dataset,
        project_root=project_root,
    )
    pids = [int(i) for i in partition_ids] if partition_ids is not None else _partition_ids_from_folder(
        part_folder, project_root
    )
    if not pids:
        raise ValueError("No partition ids found. Check partitions_folder or pass partition_ids explicitly.")

    print(f"Partition Hs/Tp/Dp — partitions_folder: {part_folder}")
    if include_bulk:
        bulk_label = _historical_bulk_series_label(hist_folder, project_root)
        print(f"Partition Hs/Tp/Dp — historical_folder (bulk): {hist_folder} | {bulk_label!r}")
        if include_binwaves_reference is None:
            include_binwaves_reference = _uses_kma_merged_grids_folder(hist_folder, project_root)
        if include_binwaves_reference:
            ref_folder = _resolve_binwaves_reference_folder(
                historical_folder=hist_folder,
                historical_dataset=historical_dataset,
                partitions_folder=part_folder,
                partitions_dataset=partitions_dataset,
                binwaves_reference_folder=binwaves_reference_folder,
                project_root=project_root,
            )
            if ref_folder is not None:
                print(f"Partition Hs/Tp/Dp — BinWaves reference folder: {ref_folder}")

    def _axis_limits_for_bundle(
        bundles: dict[str, dict[str, tuple[xr.DataArray, dict]]],
        *,
        include_qs: bool,
        shore_deg: float | None,
    ) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for var in ("hs", "tp", "dp"):
            if not bundles[var]:
                continue
            vals_all: list[np.ndarray] = []
            for da, _ in bundles[var].values():
                vals = np.asarray(da.values, dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size > 0:
                    vals_all.append(vals)
            if vals_all:
                merged = np.concatenate(vals_all)
                out[var] = (float(np.nanmin(merged)), float(np.nanmax(merged)))
        if include_qs and shore_deg is not None:
            q_vals_all: list[np.ndarray] = []
            shared_keys = set(bundles["hs"].keys()) & set(bundles["dp"].keys())
            for key in shared_keys:
                qs = _qs_timeseries_for_bundle_key(
                    bundles, key, shore_deg, K=K, qs_formula=qs_formula
                )
                if qs is not None:
                    qv = np.asarray(qs.values, dtype=float)
                    qv = qv[np.isfinite(qv)]
                    if qv.size > 0:
                        q_vals_all.append(qv)
            if q_vals_all:
                q_merged = np.concatenate(q_vals_all)
                out["qs"] = (float(np.nanmin(q_merged)), float(np.nanmax(q_merged)))
        # Keep existing Tp behavior if bulk missing.
        if "tp" not in out:
            out["tp"] = (0.0, 20.0)
        return out

    prepared: list[tuple[float, float, dict[str, dict[str, tuple[xr.DataArray, dict]]], str, float | None, dict[str, tuple[float, float]]]] = []
    global_limits: dict[str, tuple[float, float]] = {}

    for lat, lon in coords:
        bundles: dict[str, dict[str, tuple[xr.DataArray, dict]]] = {"hs": {}, "tp": {}, "dp": {}}
        site_note = ""
        print(f"Coordinate (lat, lon) = ({lat:.3f}, {lon:.3f})")
        for pid in pids:
            part_info = None
            for panel_var, var_name in (
                ("hs", f"phs{pid}"),
                ("tp", f"ptp{pid}"),
                ("dp", f"dp{pid}"),
            ):
                try:
                    nc_path = _resolve_variable_nc_path(part_folder, var_name, project_root)
                except FileNotFoundError:
                    print(f"SKIP partition {pid} {panel_var}: missing {var_name} in {part_folder}")
                    continue
                da, info = load_timeseries_at_coordinate(
                    nc_path,
                    (lat, lon),
                    variable=var_name,
                    aggregation=aggregation,
                    time_start=time_start,
                    time_end=time_end,
                    project_root=project_root,
                )
                _warn_distant_site_match((lat, lon), info, label=f"partition {pid} {panel_var}")
                bundles[panel_var][f"partition {pid}"] = (da, info)
                if part_info is None:
                    part_info = info

            if part_info is not None and not site_note:
                site_note = _site_note_from_info((lat, lon), part_info)

        if include_bulk:
            bulk_key = _historical_bulk_series_label(hist_folder, project_root)
            for panel_var in ("hs", "tp", "dp"):
                try:
                    nc_path = _resolve_variable_nc_path(hist_folder, panel_var, project_root=None)
                except FileNotFoundError:
                    print(f"SKIP bulk {panel_var}: missing in {hist_folder}")
                    continue
                da, info = load_timeseries_at_coordinate(
                    nc_path,
                    (lat, lon),
                    variable=panel_var,
                    aggregation=aggregation,
                    time_start=time_start,
                    time_end=time_end,
                    project_root=project_root,
                )
                _warn_distant_site_match((lat, lon), info, label=f"bulk {panel_var}")
                bundles[panel_var][bulk_key] = (da, info)
                if not site_note:
                    site_note = _site_note_from_info((lat, lon), info)

            if include_binwaves_reference is None:
                include_binwaves_reference = _uses_kma_merged_grids_folder(hist_folder, project_root)
            if include_binwaves_reference:
                ref_folder = _resolve_binwaves_reference_folder(
                    historical_folder=hist_folder,
                    historical_dataset=historical_dataset,
                    partitions_folder=part_folder,
                    partitions_dataset=partitions_dataset,
                    binwaves_reference_folder=binwaves_reference_folder,
                    project_root=project_root,
                )
                if ref_folder is not None:
                    for panel_var in ("hs", "tp", "dp"):
                        try:
                            nc_path = _resolve_variable_nc_path(ref_folder, panel_var, project_root=None)
                        except FileNotFoundError:
                            print(f"SKIP {BINWAVES_BULK_LABEL} {panel_var}: missing in {ref_folder}")
                            continue
                        da, info = load_timeseries_at_coordinate(
                            nc_path,
                            (lat, lon),
                            variable=panel_var,
                            aggregation=aggregation,
                            time_start=time_start,
                            time_end=time_end,
                            project_root=project_root,
                        )
                        _warn_distant_site_match((lat, lon), info, label=f"{BINWAVES_BULK_LABEL} {panel_var}")
                        bundles[panel_var][BINWAVES_BULK_LABEL] = (da, info)

        if include_buoy:
            bid = _resolve_buoy_id_for_coordinate(
                (lat, lon),
                buoy_id=buoy_id,
                buoys=buoys,
                buoy_data_dir=buoy_data_dir,
                max_distance_km=buoy_max_distance_km,
            )
            if bid is not None:
                buoy_panels, buoy_info = load_buoy_hs_tp_dp(
                    bid,
                    buoy_data_dir=buoy_data_dir,
                    aggregation=aggregation,
                    time_start=time_start,
                    time_end=time_end,
                )
                if buoy_panels:
                    label = f"buoy {bid}"
                    for panel_var in ("hs", "dp"):
                        if buoy_panels.get(panel_var) is not None:
                            bundles[panel_var][label] = (buoy_panels[panel_var], buoy_info)
                    if buoy_panels.get("tp") is not None:
                        bundles["tp"][label] = (buoy_panels["tp"], buoy_info)
                    elif qs_formula == "deep" and include_sediment_transport_row:
                        print(
                            f"WARN buoy {bid}: no Tp column for deep-water Qs "
                            f"(tried {_BUOY_TP_COLUMNS})"
                        )

        if swan_timeseries_path is not None or swan_timeseries_folder is not None:
            swan_nc = resolve_swan_timeseries_path(
                (lat, lon),
                swan_timeseries_path=swan_timeseries_path,
                swan_timeseries_folder=(
                    swan_timeseries_folder
                    if swan_timeseries_folder is not None
                    else DEFAULT_SWAN_TIMESERIES_FOLDER
                ),
                project_root=project_root,
            )
            if swan_nc is None:
                if swan_timeseries_path is not None:
                    print(f"SKIP SWAN bulk: file not found ({swan_timeseries_path})")
                else:
                    print(
                        f"SKIP SWAN bulk: no file for ({lat:.3f}, {lon:.3f}) in "
                        f"{resolve_path(swan_timeseries_folder, project_root)}"
                    )
            else:
                swan_panels, swan_info = load_swan_bulk_hs_tp_dp(
                    swan_nc,
                    aggregation=aggregation,
                    time_start=time_start,
                    time_end=time_end,
                    project_root=project_root,
                )
                if swan_panels:
                    swan_label = swan_bulk_label or SWAN_BULK_LABEL
                    for panel_var in ("hs", "tp", "dp"):
                        if swan_panels.get(panel_var) is not None:
                            bundles[panel_var][swan_label] = (
                                swan_panels[panel_var],
                                swan_info,
                            )
                    print(f"SWAN bulk ({swan_label}) loaded from {swan_nc.name} ({swan_info['n_points']} points)")

        if not any(bundles[v] for v in ("hs", "tp", "dp")):
            print(f"No partition/bulk Hs-Tp-Dp series loaded for ({lat:.3f}, {lon:.3f})")
            continue

        formula_note = f" | Qs={qs_formula}" if include_sediment_transport_row else ""
        title = (
            f"Partition Hs/Tp/Dp ({aggregation}) @ ({lat:.3f}, {lon:.3f}) — {site_note}"
            f"{formula_note}"
        )
        if layout == "grid":
            shore_deg_for_qs: float | None = None
            if include_sediment_transport_row:
                if shoreline_orientation_deg is None:
                    shore_deg_for_qs = compute_shoreline_orientations_from_coastline(
                        [(lat, lon)], convention="seaward"
                    )[0].angle_deg
                else:
                    shore_deg_for_qs = normalize_shoreline_orientations(
                        [(lat, lon)], shoreline_orientation_deg
                    )[0][1]
            local_limits = _axis_limits_for_bundle(
                bundles,
                include_qs=include_sediment_transport_row,
                shore_deg=shore_deg_for_qs,
            )
            for k, (vmin, vmax) in local_limits.items():
                if k not in global_limits:
                    global_limits[k] = (vmin, vmax)
                else:
                    gvmin, gvmax = global_limits[k]
                    global_limits[k] = (min(gvmin, vmin), max(gvmax, vmax))
            prepared.append((lat, lon, bundles, title, shore_deg_for_qs, local_limits))
            continue

        scenarios_cfg: list[tuple[str, dict]] = []
        present_labels = []
        for panel_var in ("hs", "tp", "dp"):
            for label in bundles[panel_var].keys():
                if label not in present_labels:
                    present_labels.append(label)
        for i, label in enumerate(present_labels):
            from utils import kma_cluster_swan as kcs

            if label == kcs.BINWAVES_PLUS_KMA_LABEL:
                scenarios_cfg.append(
                    (
                        label,
                        {
                            "folder": str(hist_folder),
                            "color": SWAN_WIND_OVERLAY_COLOR,
                            "lw": 1.4,
                            "alpha": 0.95,
                        },
                    )
                )
            elif label == BINWAVES_BULK_LABEL or label == "historical bulk":
                scenarios_cfg.append(
                    (
                        label,
                        {
                            "folder": str(hist_folder),
                            "color": BINWAVES_OVERLAY_COLOR,
                            "lw": 1.4,
                            "alpha": 0.95,
                        },
                    )
                )
            elif label == SWAN_BULK_LABEL:
                scenarios_cfg.append(
                    (
                        label,
                        {
                            "folder": "swan-bulk",
                            "color": SWAN_WIND_OVERLAY_COLOR,
                            "lw": 1.1,
                            "alpha": 0.9,
                        },
                    )
                )
            else:
                scenarios_cfg.append((label, {"folder": f"partition-{i}", "lw": 1.0, "alpha": 0.78}))

        if static:
            # overlay plotting happens immediately; shared-axis normalization applies to grid layout.
            plot_timeseries_matplotlib_multi(
                bundles,
                title=title,
                scenarios_cfg=scenarios_cfg,
                historical_folder=hist_folder,
                project_root=project_root,
            )
        if interactive:
            plot_timeseries_plotly_multi(
                bundles,
                title=title,
                scenarios_cfg=scenarios_cfg,
                historical_folder=hist_folder,
                project_root=project_root,
            )

    if layout == "grid":
        for lat, lon, bundles, title, shore_deg_for_qs, local_limits in prepared:
            axis_limits = global_limits if bulk_axis_scope == "all_figures" else local_limits
            if static:
                _plot_partition_hs_tp_dp_grid(
                    bundles,
                    title=title,
                    partition_ids=pids,
                    include_bulk=include_bulk,
                    orientation=grid_orientation,
                    include_sediment_transport_row=include_sediment_transport_row,
                    shoreline_orientation_deg=shore_deg_for_qs,
                    K=K,
                    qs_formula=qs_formula,
                    overlay_buoy_on_bulk=overlay_buoy_on_bulk,
                    overlay_swan_on_bulk=overlay_swan_on_bulk,
                    swan_bulk_label=swan_bulk_label,
                    axis_limits=axis_limits,
                    time_highlights=time_highlights,
                )
            if interactive:
                _plot_partition_hs_tp_dp_grid_plotly(
                    bundles,
                    title=title,
                    partition_ids=pids,
                    include_bulk=include_bulk,
                    orientation=grid_orientation,
                    include_sediment_transport_row=include_sediment_transport_row,
                    shoreline_orientation_deg=shore_deg_for_qs,
                    K=K,
                    qs_formula=qs_formula,
                    overlay_buoy_on_bulk=overlay_buoy_on_bulk,
                    overlay_swan_on_bulk=overlay_swan_on_bulk,
                    swan_bulk_label=swan_bulk_label,
                    axis_limits=axis_limits,
                    time_highlights=time_highlights,
                )
            # no continue needed; this is the final plotting pass for grid mode


def _load_hs_dp_at_coordinate(
    coordinate: tuple[float, float],
    *,
    partition_ids: Sequence[int],
    aggregation: str = "daily",
    time_start=None,
    time_end=None,
    include_bulk: bool = True,
    partitions_folder: str | Path,
    historical_folder: str | Path,
    project_root: str | Path | None = None,
) -> tuple[dict[str, tuple[xr.DataArray, xr.DataArray]], str]:
    """
    Load aligned (Hs, Dp) pairs at one site for bulk and each partition.

    Returns ``{label: (hs, dp)}`` with labels ``historical bulk`` and ``partition <id>``.
    """
    lat, lon = coordinate
    series: dict[str, tuple[xr.DataArray, xr.DataArray]] = {}
    site_note = ""

    for pid in partition_ids:
        hs_name = f"phs{pid}"
        dp_name = f"dp{pid}"
        try:
            hs_path = _resolve_variable_nc_path(partitions_folder, hs_name, project_root)
            dp_path = _resolve_variable_nc_path(partitions_folder, dp_name, project_root)
        except FileNotFoundError as exc:
            print(f"SKIP partition {pid} Hs/Dp: {exc}")
            continue
        hs_da, info_hs = load_timeseries_at_coordinate(
            hs_path,
            (lat, lon),
            variable=f"phs{pid}",
            aggregation=aggregation,
            time_start=time_start,
            time_end=time_end,
            project_root=project_root,
        )
        dp_da, _ = load_timeseries_at_coordinate(
            dp_path,
            (lat, lon),
            variable=f"dp{pid}",
            aggregation=aggregation,
            time_start=time_start,
            time_end=time_end,
            project_root=project_root,
        )
        _warn_distant_site_match((lat, lon), info_hs, label=f"partition {pid} Hs")
        hs_a, dp_a = xr.align(hs_da, dp_da, join="inner")
        if hs_a.sizes.get("time", 0) == 0:
            print(f"SKIP partition {pid}: no overlapping Hs/Dp times")
            continue
        series[f"partition {pid}"] = (hs_a, dp_a)
        if not site_note:
            site_note = _site_note_from_info((lat, lon), info_hs)

    if include_bulk:
        try:
            hs_nc = _resolve_variable_nc_path(historical_folder, "hs", project_root=None)
            dp_nc = _resolve_variable_nc_path(historical_folder, "dp", project_root=None)
        except FileNotFoundError:
            print(f"SKIP bulk Hs/Dp: missing in {historical_folder}")
        else:
            hs_da, info_hs = load_timeseries_at_coordinate(
                hs_nc,
                (lat, lon),
                variable="hs",
                aggregation=aggregation,
                time_start=time_start,
                time_end=time_end,
                project_root=project_root,
            )
            dp_da, _ = load_timeseries_at_coordinate(
                dp_nc,
                (lat, lon),
                variable="dp",
                aggregation=aggregation,
                time_start=time_start,
                time_end=time_end,
                project_root=project_root,
            )
            _warn_distant_site_match((lat, lon), info_hs, label="bulk Hs")
            hs_a, dp_a = xr.align(hs_da, dp_da, join="inner")
            if hs_a.sizes.get("time", 0) == 0:
                print("SKIP bulk: no overlapping Hs/Dp times")
            else:
                series["historical bulk"] = (hs_a, dp_a)
                if not site_note:
                    site_note = _site_note_from_info((lat, lon), info_hs)

    return series, site_note


def _hs_dp_joint_row_order(
    series: Mapping[str, tuple[xr.DataArray, xr.DataArray]],
    *,
    partition_ids: Sequence[int],
    include_bulk: bool,
) -> list[str]:
    rows: list[str] = []
    if include_bulk and "historical bulk" in series:
        rows.append("historical bulk")
    for pid in partition_ids:
        label = f"partition {pid}"
        if label in series:
            rows.append(label)
    return rows


def _plot_hs_dp_joint_pdf_figure(
    series: Mapping[str, tuple[xr.DataArray, xr.DataArray]],
    *,
    title: str,
    row_labels: Sequence[str],
    bins_hs: int = 40,
    bins_dp: int = 36,
    hs_range: tuple[float, float] = (0.0, 6.0),
    dp_range: tuple[float, float] = (0.0, 360.0),
    cmap: str = "turbo",
    vmax_percentile: float = 92.0,
) -> None:
    """One row per label: 2D density of Dp (x) vs Hs (y)."""
    n_rows = len(row_labels)
    if n_rows == 0:
        return

    dp_edges = np.linspace(dp_range[0], dp_range[1], int(bins_dp) + 1)
    hs_edges = np.linspace(hs_range[0], hs_range[1], int(bins_hs) + 1)

    hist_stack = []
    for label in row_labels:
        hs_da, dp_da = series[label]
        h_vals = np.asarray(hs_da.values, dtype=float).ravel()
        d_vals = np.asarray(dp_da.values, dtype=float).ravel()
        valid = np.isfinite(h_vals) & np.isfinite(d_vals)
        h_vals = h_vals[valid]
        d_vals = d_vals[valid]
        if h_vals.size == 0:
            hist_stack.append(np.zeros((bins_hs, bins_dp)))
            continue
        h2d, _, _ = np.histogram2d(d_vals, h_vals, bins=[dp_edges, hs_edges], density=True)
        hist_stack.append(h2d.T)

    positive = np.concatenate([h.ravel() for h in hist_stack if np.any(h > 0)])
    positive = positive[positive > 0]
    if positive.size:
        vmax = float(np.percentile(positive, vmax_percentile))
        vmax = max(vmax, 1e-8)
    else:
        vmax = 1.0

    fig_h = max(2.2 * n_rows, 5)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8.0, fig_h), squeeze=False)
    axes = axes.ravel()

    for r, label in enumerate(row_labels):
        ax = axes[r]
        pid = _partition_id_from_qs_label(label)
        if label == "historical bulk":
            style_color = "0.2"
        elif label.startswith("buoy "):
            style_color = "k"
        elif pid is not None:
            style_color = _qs_partition_color(pid, r)
        else:
            style_color = None

        pcm = ax.pcolormesh(
            dp_edges,
            hs_edges,
            hist_stack[r],
            cmap=cmap,
            shading="auto",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_ylabel("Hs (m)")
        ax.set_xlim(dp_range)
        ax.set_ylim(hs_range)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        row_title = "bulk" if label == "historical bulk" else label
        ax.set_title(row_title, fontsize=10, loc="left", color=style_color or "k")
        if r == n_rows - 1:
            ax.set_xlabel("Dp (°)")
        else:
            ax.set_xlabel("")

    fig.subplots_adjust(left=0.12, right=0.82, top=0.94, bottom=0.06, hspace=0.28)
    cbar_ax = fig.add_axes([0.84, 0.12, 0.025, 0.76])
    cbar = fig.colorbar(pcm, cax=cbar_ax, label="joint PDF")
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle(title, y=0.99, fontsize=10)
    plt.show()


def plot_partition_hs_dp_joint_pdf(
    coordinates: Sequence[float] | Sequence[Sequence[float]],
    *,
    partition_ids: Sequence[int] | None = None,
    time_start=None,
    time_end=None,
    aggregation: str = "daily",
    include_bulk: bool = True,
    bins_hs: int = 40,
    bins_dp: int = 36,
    hs_range: tuple[float, float] = (0.0, 6.0),
    dp_range: tuple[float, float] = (0.0, 360.0),
    cmap: str = "turbo",
    vmax_percentile: float = 92.0,
    partitions_folder: str | Path | None = None,
    partitions_dataset: HistoricalDataset = "merged_grids",
    historical_folder: str | Path | None = None,
    historical_dataset: HistoricalDataset = "merged_grids",
    project_root: str | Path | None = None,
) -> None:
    """
    Joint probability density of wave direction (Dp) vs Hs for bulk and each partition.

    Uses the same ``merged_grids`` files and site selection as
    ``plot_partition_hs_tp_dp_timeseries`` (``phs*``/``dp*`` per partition;
    ``hs_merged_all`` / ``dp_merged_all`` for bulk).

    One figure per coordinate: rows = bulk (optional) + partitions; x = Dp (°), y = Hs (m).
    """
    _warn_native(aggregation)
    coords = normalize_coordinates(coordinates)
    part_folder = resolve_partitions_folder(
        partitions_folder=partitions_folder,
        partitions_dataset=partitions_dataset,
        project_root=project_root,
    )
    hist_folder = resolve_historical_folder(
        historical_folder=historical_folder,
        historical_dataset=historical_dataset,
        project_root=project_root,
    )
    pids = [int(i) for i in partition_ids] if partition_ids is not None else _partition_ids_from_folder(
        part_folder, project_root
    )
    if not pids:
        raise ValueError("No partition ids found. Check partitions_folder or pass partition_ids explicitly.")

    print(f"Hs–Dp joint PDF — partitions_folder: {part_folder}")
    if include_bulk:
        print(f"Hs–Dp joint PDF — historical_folder (bulk): {hist_folder}")

    for lat, lon in coords:
        print(f"Coordinate (lat, lon) = ({lat:.3f}, {lon:.3f})")
        series, site_note = _load_hs_dp_at_coordinate(
            (lat, lon),
            partition_ids=pids,
            aggregation=aggregation,
            time_start=time_start,
            time_end=time_end,
            include_bulk=include_bulk,
            partitions_folder=part_folder,
            historical_folder=hist_folder,
            project_root=project_root,
        )
        if not series:
            print(f"No Hs/Dp series loaded for ({lat:.3f}, {lon:.3f})")
            continue

        row_labels = _hs_dp_joint_row_order(series, partition_ids=pids, include_bulk=include_bulk)
        title = f"Hs–Dp joint PDF ({aggregation}) @ ({lat:.3f}, {lon:.3f}) — {site_note}"
        _plot_hs_dp_joint_pdf_figure(
            series,
            title=title,
            row_labels=row_labels,
            bins_hs=bins_hs,
            bins_dp=bins_dp,
            hs_range=hs_range,
            dp_range=dp_range,
            cmap=cmap,
            vmax_percentile=vmax_percentile,
        )


def plot_buoy_bulk_hs_dp_joint_pdf(
    buoy_ids: str | Sequence[str],
    *,
    coordinates: Sequence[float] | Sequence[Sequence[float]] | None = None,
    use_buoy_time_range: bool = True,
    time_start=None,
    time_end=None,
    aggregation: str = "daily",
    bins_hs: int = 40,
    bins_dp: int = 36,
    hs_range: tuple[float, float] = (0.0, 6.0),
    dp_range: tuple[float, float] = (0.0, 360.0),
    cmap: str = "turbo",
    vmax_percentile: float = 92.0,
    historical_folder: str | Path | None = None,
    historical_dataset: HistoricalDataset = "merged_grids",
    project_root: str | Path | None = None,
    buoy_data_dir: str | Path = DEFAULT_BUOY_DATA_DIR,
    buoys: Mapping[str, Any] | None = None,
    max_match_distance_km: float | None = None,
) -> list[BuoyHistoricalMatch]:
    """
    Joint PDF of wave direction vs Hs for buoy bulk parameters vs historical hindcast.

    One figure per buoy: **top** = NDBC buoy (``Hs_Buoy`` / ``Dir_Buoy``); **bottom** =
    nearest ``merged_grids`` bulk hindcast (``hs_merged_all`` / ``dp_merged_all``) on the
    **same time window** as the buoy observations.

    Site matching and time-range behaviour follow ``plot_longshore_transport_for_buoys``.
    Buoys without pickle data are skipped unless ``use_buoy_time_range=False`` and
    ``time_start`` / ``time_end`` are set (historical-only row is not plotted in that case).

    Returns ``BuoyHistoricalMatch`` records (coordinate, distance, time window).
    """
    _warn_native(aggregation)
    ids = [buoy_ids] if isinstance(buoy_ids, str) else [str(b) for b in buoy_ids]
    if not ids:
        raise ValueError("buoy_ids must contain at least one buoy ID")

    matches = [
        match_buoy_to_historical(
            bid,
            coordinates=coordinates,
            buoys=buoys,
            buoy_data_dir=buoy_data_dir,
            historical_folder=historical_folder,
            historical_dataset=historical_dataset,
            project_root=project_root,
        )
        for bid in ids
    ]

    hist_folder = resolve_historical_folder(
        historical_folder=historical_folder,
        historical_dataset=historical_dataset,
        project_root=project_root,
    )
    print(f"Buoy vs hindcast Hs–Dp joint PDF — historical_folder: {hist_folder}")

    results: list[BuoyHistoricalMatch] = []

    for match in matches:
        if max_match_distance_km is not None and match.distance_km > max_match_distance_km:
            print(
                f"SKIP buoy {match.buoy_id}: nearest "
                f"{'candidate' if match.match_source == 'coordinates' else 'grid'} point "
                f"is {match.distance_km:.1f} km away (max {max_match_distance_km:.1f} km)"
            )
            results.append(match)
            continue

        lat, lon = match.coordinate

        if use_buoy_time_range:
            t0, t1 = get_buoy_observation_period(match.buoy_id, buoy_data_dir)
            if t0 is None or t1 is None:
                if time_start is None:
                    print(
                        f"SKIP buoy {match.buoy_id}: no buoy observations and no time_start"
                    )
                    results.append(match)
                    continue
                period_start = pd.Timestamp(time_start)
                period_end = pd.Timestamp(time_end) if time_end is not None else None
                print(
                    f"INFO buoy {match.buoy_id}: no buoy observations; using fallback period "
                    f"{period_start.date()} – "
                    f"{(period_end.date() if period_end is not None else 'latest')}"
                )
            else:
                period_start, period_end = t0, t1
        else:
            if time_start is None or time_end is None:
                raise ValueError(
                    "time_start and time_end are required when use_buoy_time_range=False"
                )
            period_start = pd.Timestamp(time_start)
            period_end = pd.Timestamp(time_end)

        match = BuoyHistoricalMatch(
            buoy_id=match.buoy_id,
            buoy_lat=match.buoy_lat,
            buoy_lon=match.buoy_lon,
            coordinate=match.coordinate,
            distance_km=match.distance_km,
            match_source=match.match_source,
            site_index=match.site_index,
            time_start=period_start,
            time_end=period_end,
        )
        results.append(match)

        print(
            f"Buoy {match.buoy_id} @ ({match.buoy_lat:.3f}, {match.buoy_lon:.3f}) → "
            f"hindcast ({lat:.3f}, {lon:.3f}), {match.distance_km:.1f} km "
            f"[{match.match_source}"
            + (f", site {match.site_index}" if match.site_index is not None else "")
            + f"] | period {period_start.date()} – "
            + (f"{period_end.date()}" if period_end is not None else "latest")
        )

        buoy_pair, _buoy_info = load_buoy_hs_dp(
            match.buoy_id,
            buoy_data_dir=buoy_data_dir,
            aggregation=aggregation,
            time_start=period_start,
            time_end=period_end,
        )
        if buoy_pair is None:
            continue

        hist_series, site_note = _load_hs_dp_at_coordinate(
            (lat, lon),
            partition_ids=[],
            aggregation=aggregation,
            time_start=period_start,
            time_end=period_end,
            include_bulk=True,
            partitions_folder=hist_folder,
            historical_folder=hist_folder,
            project_root=project_root,
        )
        if "historical bulk" not in hist_series:
            print(f"No historical bulk Hs/Dp for buoy {match.buoy_id} at ({lat:.3f}, {lon:.3f})")
            continue

        buoy_label = f"buoy {match.buoy_id}"
        series = {
            buoy_label: buoy_pair,
            "historical bulk": hist_series["historical bulk"],
        }
        row_labels = [buoy_label, "historical bulk"]
        period_note = (
            f"{period_start.date()}–"
            f"{(period_end.date() if period_end is not None else 'latest')}"
        )
        title = (
            f"Hs–Dp joint PDF ({aggregation}) — buoy {match.buoy_id} vs hindcast "
            f"@ ({lat:.3f}, {lon:.3f}) | {period_note} — {site_note}"
        )
        _plot_hs_dp_joint_pdf_figure(
            series,
            title=title,
            row_labels=row_labels,
            bins_hs=bins_hs,
            bins_dp=bins_dp,
            hs_range=hs_range,
            dp_range=dp_range,
            cmap=cmap,
            vmax_percentile=vmax_percentile,
        )

    return results


def _aligned_hs_dp_numpy(
    hs_da: xr.DataArray,
    dp_da: xr.DataArray,
    *,
    hs_min: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite (Hs, Dp) pairs with ``hs >= hs_min`` and ``0 <= Dp <= 360``."""
    hs_a, dp_a = xr.align(hs_da, dp_da, join="inner")
    h_vals = np.asarray(hs_a.values, dtype=float).ravel()
    d_vals = np.asarray(dp_a.values, dtype=float).ravel()
    valid = (
        np.isfinite(h_vals)
        & np.isfinite(d_vals)
        & (h_vals >= hs_min)
        & (d_vals >= 0.0)
        & (d_vals <= 360.0)
    )
    return h_vals[valid], d_vals[valid]


def _direction_bin_edges(n_dir_bins: int = 36) -> np.ndarray:
    return np.linspace(0.0, 360.0, int(n_dir_bins) + 1)


def _direction_bin_centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def _count_fraction_by_direction(
    d_vals: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    counts, _ = np.histogram(d_vals, bins=edges)
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts, dtype=float)
    return counts.astype(float) / float(total)


def _hs_stats_in_direction_bins(
    h_vals: np.ndarray,
    d_vals: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per direction bin: count, mean Hs, median Hs, 90th percentile Hs, max Hs."""
    n = len(edges) - 1
    counts = np.zeros(n, dtype=int)
    mean_hs = np.full(n, np.nan)
    median_hs = np.full(n, np.nan)
    p90_hs = np.full(n, np.nan)
    max_hs = np.full(n, np.nan)
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        if i < n - 1:
            mask = (d_vals >= lo) & (d_vals < hi)
        else:
            mask = (d_vals >= lo) & (d_vals <= hi)
        sel = h_vals[mask]
        if sel.size == 0:
            continue
        counts[i] = int(sel.size)
        mean_hs[i] = float(np.mean(sel))
        median_hs[i] = float(np.median(sel))
        p90_hs[i] = float(np.percentile(sel, 90))
        max_hs[i] = float(np.max(sel))
    return counts, mean_hs, median_hs, p90_hs, max_hs


def _print_buoy_only_direction_gaps(
    *,
    buoy_id: str,
    edges: np.ndarray,
    buoy_frac: np.ndarray,
    model_frac: np.ndarray,
    buoy_counts: np.ndarray,
    model_counts: np.ndarray,
    buoy_mean_hs: np.ndarray,
    buoy_median_hs: np.ndarray,
    buoy_p90_hs: np.ndarray,
    buoy_max_hs: np.ndarray,
    buoy_min_frac: float,
    model_max_frac: float,
    model_max_count: int,
) -> None:
    """Stdout table: direction sectors in buoy but rare/absent in hindcast."""
    centers = _direction_bin_centers(edges)
    rows: list[str] = []
    for i, center in enumerate(centers):
        if buoy_frac[i] < buoy_min_frac:
            continue
        if model_frac[i] > model_max_frac and model_counts[i] > model_max_count:
            continue
        lo, hi = edges[i], edges[i + 1]
        rows.append(
            f"  {lo:5.0f}–{hi:5.0f}° ({center:5.0f}°): "
            f"n={buoy_counts[i]:5d} ({100 * buoy_frac[i]:4.2f}% buoy) | "
            f"model n={model_counts[i]:5d} ({100 * model_frac[i]:4.2f}%) | "
            f"Hs mean={buoy_mean_hs[i]:.2f} med={buoy_median_hs[i]:.2f} "
            f"p90={buoy_p90_hs[i]:.2f} max={buoy_max_hs[i]:.2f} m"
        )
    print(f"Buoy {buoy_id} — direction sectors present in buoy, weak/absent in hindcast:")
    if rows:
        print("\n".join(rows))
    else:
        print("  (none above thresholds)")


def _plot_buoy_hindcast_hs_dp_distribution_figure(
    h_buoy: np.ndarray,
    d_buoy: np.ndarray,
    h_hist: np.ndarray,
    d_hist: np.ndarray,
    *,
    title: str,
    buoy_label: str = "buoy",
    hist_label: str = "hindcast bulk",
    n_dir_bins: int = 36,
    n_hs_bins: int = 30,
    hs_range: tuple[float, float] = (0.0, 6.0),
    buoy_min_frac: float = 0.001,
    model_max_frac: float = 0.0005,
    model_max_count: int = 5,
) -> None:
    """
    Marginal and sector diagnostics for buoy vs hindcast Hs/Dp.

    Panels: direction PDF, Hs PDF, mean Hs vs direction, buoy−model direction occupancy
    (bars coloured by buoy mean Hs in that sector).
    """
    edges = _direction_bin_edges(n_dir_bins)
    centers = _direction_bin_centers(edges)
    buoy_frac = _count_fraction_by_direction(d_buoy, edges)
    hist_frac = _count_fraction_by_direction(d_hist, edges)
    b_counts, b_mean, b_median, b_p90, b_max = _hs_stats_in_direction_bins(h_buoy, d_buoy, edges)
    h_counts, h_mean, _, _, h_max = _hs_stats_in_direction_bins(h_hist, d_hist, edges)
    gap_mask = (buoy_frac >= buoy_min_frac) & (
        (hist_frac <= model_max_frac) | (h_counts <= model_max_count)
    )

    buoy_color = "k"
    hist_color = "fuchsia"

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))
    ax_dir, ax_hs, ax_mean, ax_gap = axes.ravel()

    width = 360.0 / n_dir_bins * 0.42
    ax_dir.bar(
        centers - width / 2,
        buoy_frac * 100,
        width=width,
        label=buoy_label,
        color=buoy_color,
        alpha=0.55,
        edgecolor="none",
    )
    ax_dir.bar(
        centers + width / 2,
        hist_frac * 100,
        width=width,
        label=hist_label,
        color=hist_color,
        alpha=0.75,
        edgecolor="none",
    )
    ax_dir.set_xlim(0, 360)
    ax_dir.set_xlabel("Dp (°)")
    ax_dir.set_ylabel("fraction of samples (%)")
    ax_dir.set_title("Direction distribution")
    ax_dir.legend(fontsize=8, loc="upper right")
    ax_dir.grid(True, alpha=0.25)

    hs_edges = np.linspace(hs_range[0], hs_range[1], int(n_hs_bins) + 1)
    ax_hs.hist(
        h_buoy,
        bins=hs_edges,
        density=True,
        histtype="step",
        linewidth=1.6,
        color=buoy_color,
        label=buoy_label,
    )
    ax_hs.hist(
        h_hist,
        bins=hs_edges,
        density=True,
        histtype="step",
        linewidth=1.4,
        color=hist_color,
        label=hist_label,
    )
    ax_hs.set_xlabel("Hs (m)")
    ax_hs.set_ylabel("density")
    ax_hs.set_title("Hs distribution (all directions)")
    ax_hs.set_xlim(hs_range)
    ax_hs.legend(fontsize=8)
    ax_hs.grid(True, alpha=0.25)

    ax_mean.plot(
        centers,
        b_mean,
        "-o",
        ms=3,
        color=buoy_color,
        lw=1.2,
        label=f"{buoy_label} mean",
    )
    ax_mean.plot(
        centers,
        b_max,
        "--",
        ms=0,
        color=buoy_color,
        lw=1.0,
        alpha=0.75,
        label=f"{buoy_label} max",
    )
    ax_mean.plot(
        centers,
        h_mean,
        "-o",
        ms=3,
        color=hist_color,
        lw=1.2,
        label=f"{hist_label} mean",
    )
    ax_mean.plot(
        centers,
        h_max,
        "--",
        ms=0,
        color=hist_color,
        lw=1.0,
        alpha=0.75,
        label=f"{hist_label} max",
    )
    ax_mean.set_xlim(0, 360)
    ax_mean.set_xlabel("Dp (°)")
    ax_mean.set_ylabel("Hs (m)")
    ax_mean.set_title("Mean and max Hs by direction sector")
    ax_mean.legend(fontsize=7, ncol=2, loc="upper right")
    ax_mean.grid(True, alpha=0.25)

    delta_pct = (buoy_frac - hist_frac) * 100
    bar_colors = np.where(np.isfinite(b_mean), b_mean, 0.0)
    bar_colors = bar_colors / (np.nanmax(bar_colors) if np.nanmax(bar_colors) > 0 else 1.0)
    bars = ax_gap.bar(
        centers,
        delta_pct,
        width=360.0 / n_dir_bins * 0.9,
        color=plt.cm.viridis(bar_colors),
        edgecolor="0.3",
        linewidth=0.3,
    )
    for i, bar in enumerate(bars):
        if gap_mask[i]:
            bar.set_edgecolor("crimson")
            bar.set_linewidth(1.8)
    ax_gap.axhline(0, color="0.3", lw=0.8)
    ax_gap.set_xlim(0, 360)
    ax_gap.set_xlabel("Dp (°)")
    ax_gap.set_ylabel("buoy % − hindcast %")
    ax_gap.set_title("Direction occupancy gap (red edge = buoy-only sectors)")
    ax_gap.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=10, y=1.01)
    fig.tight_layout()
    plt.show()


def plot_buoy_bulk_hs_dp_distribution_compare(
    buoy_ids: str | Sequence[str],
    *,
    coordinates: Sequence[float] | Sequence[Sequence[float]] | None = None,
    use_buoy_time_range: bool = True,
    time_start=None,
    time_end=None,
    aggregation: str = "daily",
    n_dir_bins: int = 36,
    n_hs_bins: int = 30,
    hs_range: tuple[float, float] = (0.0, 6.0),
    buoy_min_frac: float = 0.001,
    model_max_frac: float = 0.0005,
    model_max_count: int = 5,
    historical_folder: str | Path | None = None,
    historical_dataset: HistoricalDataset = "merged_grids",
    project_root: str | Path | None = None,
    buoy_data_dir: str | Path = DEFAULT_BUOY_DATA_DIR,
    buoys: Mapping[str, Any] | None = None,
    max_match_distance_km: float | None = None,
) -> list[BuoyHistoricalMatch]:
    """
    Compare buoy vs hindcast **marginal** Hs and Dp distributions (easier than joint PDFs
    for spotting missing directions).

    Per buoy, one 2×2 figure:

    - **Direction histogram** (% of samples per sector): overlaid buoy vs hindcast.
    - **Hs histogram** (all directions combined).
    - **Mean and max Hs vs direction** (10° sectors): solid = mean, dashed = max;
      black = buoy, fuchsia = hindcast.
    - **Occupancy gap** (buoy % − hindcast % per sector); red edges mark sectors where the
      buoy has data but the hindcast is nearly empty.

    Also prints a text table of those buoy-only sectors with count and Hs statistics.

    Same site matching and time window as ``plot_buoy_bulk_hs_dp_joint_pdf``.
    """
    _warn_native(aggregation)
    ids = [buoy_ids] if isinstance(buoy_ids, str) else [str(b) for b in buoy_ids]
    if not ids:
        raise ValueError("buoy_ids must contain at least one buoy ID")

    matches = [
        match_buoy_to_historical(
            bid,
            coordinates=coordinates,
            buoys=buoys,
            buoy_data_dir=buoy_data_dir,
            historical_folder=historical_folder,
            historical_dataset=historical_dataset,
            project_root=project_root,
        )
        for bid in ids
    ]

    hist_folder = resolve_historical_folder(
        historical_folder=historical_folder,
        historical_dataset=historical_dataset,
        project_root=project_root,
    )
    print(f"Buoy vs hindcast Hs/Dp distribution compare — historical_folder: {hist_folder}")

    results: list[BuoyHistoricalMatch] = []
    edges = _direction_bin_edges(n_dir_bins)

    for match in matches:
        if max_match_distance_km is not None and match.distance_km > max_match_distance_km:
            print(
                f"SKIP buoy {match.buoy_id}: nearest "
                f"{'candidate' if match.match_source == 'coordinates' else 'grid'} point "
                f"is {match.distance_km:.1f} km away (max {max_match_distance_km:.1f} km)"
            )
            results.append(match)
            continue

        lat, lon = match.coordinate

        if use_buoy_time_range:
            t0, t1 = get_buoy_observation_period(match.buoy_id, buoy_data_dir)
            if t0 is None or t1 is None:
                if time_start is None:
                    print(
                        f"SKIP buoy {match.buoy_id}: no buoy observations and no time_start"
                    )
                    results.append(match)
                    continue
                period_start = pd.Timestamp(time_start)
                period_end = pd.Timestamp(time_end) if time_end is not None else None
            else:
                period_start, period_end = t0, t1
        else:
            if time_start is None or time_end is None:
                raise ValueError(
                    "time_start and time_end are required when use_buoy_time_range=False"
                )
            period_start = pd.Timestamp(time_start)
            period_end = pd.Timestamp(time_end)

        match = BuoyHistoricalMatch(
            buoy_id=match.buoy_id,
            buoy_lat=match.buoy_lat,
            buoy_lon=match.buoy_lon,
            coordinate=match.coordinate,
            distance_km=match.distance_km,
            match_source=match.match_source,
            site_index=match.site_index,
            time_start=period_start,
            time_end=period_end,
        )
        results.append(match)

        buoy_pair, _ = load_buoy_hs_dp(
            match.buoy_id,
            buoy_data_dir=buoy_data_dir,
            aggregation=aggregation,
            time_start=period_start,
            time_end=period_end,
        )
        if buoy_pair is None:
            continue

        hist_series, site_note = _load_hs_dp_at_coordinate(
            (lat, lon),
            partition_ids=[],
            aggregation=aggregation,
            time_start=period_start,
            time_end=period_end,
            include_bulk=True,
            partitions_folder=hist_folder,
            historical_folder=hist_folder,
            project_root=project_root,
        )
        if "historical bulk" not in hist_series:
            print(f"No historical bulk Hs/Dp for buoy {match.buoy_id}")
            continue

        h_b, d_b = _aligned_hs_dp_numpy(*buoy_pair)
        h_h, d_h = _aligned_hs_dp_numpy(*hist_series["historical bulk"])
        if h_b.size == 0 or h_h.size == 0:
            print(f"SKIP buoy {match.buoy_id}: no valid Hs/Dp pairs")
            continue

        buoy_frac = _count_fraction_by_direction(d_b, edges)
        hist_frac = _count_fraction_by_direction(d_h, edges)
        b_counts, b_mean, b_median, b_p90, b_max = _hs_stats_in_direction_bins(h_b, d_b, edges)
        h_counts, _, _, _, _ = _hs_stats_in_direction_bins(h_h, d_h, edges)

        _print_buoy_only_direction_gaps(
            buoy_id=match.buoy_id,
            edges=edges,
            buoy_frac=buoy_frac,
            model_frac=hist_frac,
            buoy_counts=b_counts,
            model_counts=h_counts,
            buoy_mean_hs=b_mean,
            buoy_median_hs=b_median,
            buoy_p90_hs=b_p90,
            buoy_max_hs=b_max,
            buoy_min_frac=buoy_min_frac,
            model_max_frac=model_max_frac,
            model_max_count=model_max_count,
        )

        period_note = (
            f"{period_start.date()}–"
            f"{(period_end.date() if period_end is not None else 'latest')}"
        )
        title = (
            f"Hs/Dp distributions ({aggregation}) — buoy {match.buoy_id} vs hindcast "
            f"| {period_note} — {site_note}"
        )
        _plot_buoy_hindcast_hs_dp_distribution_figure(
            h_b,
            d_b,
            h_h,
            d_h,
            title=title,
            buoy_label=f"buoy {match.buoy_id}",
            hist_label="hindcast bulk",
            n_dir_bins=n_dir_bins,
            n_hs_bins=n_hs_bins,
            hs_range=hs_range,
            buoy_min_frac=buoy_min_frac,
            model_max_frac=model_max_frac,
            model_max_count=model_max_count,
        )

    return results


def _hs_dp_rectangular_bin_edges(
    *,
    dir_bin_deg: float = 10.0,
    hs_bin_m: float = 1.0,
    hs_range: tuple[float, float] = (0.0, 6.0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Direction edges ``0, 10, …, 360`` (first sector **0–10°**, then 10–20°, …).

    Hs edges from ``hs_range[0]`` to ``hs_range[1]`` in steps of ``hs_bin_m`` (m).
    """
    dp_edges = np.arange(0.0, 360.0 + dir_bin_deg, dir_bin_deg)
    hs_edges = np.arange(hs_range[0], hs_range[1] + hs_bin_m, hs_bin_m)
    return dp_edges, hs_edges


def _direction_sector_labels(dp_edges: np.ndarray) -> list[str]:
    """Human-readable direction ranges, e.g. ``0–10``, ``10–20``, …, ``350–360``."""
    return [f"{dp_edges[i]:.0f}–{dp_edges[i + 1]:.0f}" for i in range(len(dp_edges) - 1)]


def _hist2d_hs_dp_fraction(
    h_vals: np.ndarray,
    d_vals: np.ndarray,
    dp_edges: np.ndarray,
    hs_edges: np.ndarray,
    *,
    normalize: Literal["global", "by_direction"] = "by_direction",
) -> np.ndarray:
    """
    2D counts of (Dp, Hs) on rectangular bins.

    Direction bin *i* is ``[dp_edges[i], dp_edges[i+1])`` (last sector includes upper edge).
    Returns array shaped ``(n_hs_bins, n_dir_bins)``.
    """
    d_vals = np.mod(d_vals, 360.0)
    h2d, _, _ = np.histogram2d(d_vals, h_vals, bins=[dp_edges, hs_edges])
    if normalize == "global":
        total = h2d.sum()
        if total <= 0:
            return np.zeros((len(hs_edges) - 1, len(dp_edges) - 1), dtype=float)
        # % of all samples; column j sums to % of waves in that direction sector
        return (h2d / total * 100.0).T
    col_sums = h2d.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(col_sums > 0, h2d / col_sums, 0.0)
    return (frac * 100.0).T


def _hs_bin_labels(hs_edges: np.ndarray) -> list[str]:
    return [f"{hs_edges[i]:g}–{hs_edges[i + 1]:g}" for i in range(len(hs_edges) - 1)]


# Low → high Hs: distinct hues (readable when stacked)
_HS_BIN_CONTRAST_COLORS = (
    "#8dd3c7",  # teal
    "#80b1d3",  # light blue
    "#bebada",  # lavender
    "#fb8072",  # salmon
    "#fdb462",  # orange
    "#b3de69",  # yellow-green
    "#fccde5",  # pink
    "#bc80bd",  # purple
    "#ccebc5",  # mint
    "#ffed6f",  # yellow
)


def _hs_bin_contrast_colors(n_hs: int) -> list[str]:
    """``n_hs`` distinguishable fill colors (cycles if needed)."""
    palette = list(_HS_BIN_CONTRAST_COLORS)
    if n_hs <= len(palette):
        return palette[:n_hs]
    return [palette[i % len(palette)] for i in range(n_hs)]


def _plot_stacked_hs_by_direction_sector(
    ax: plt.Axes,
    h_vals: np.ndarray,
    d_vals: np.ndarray,
    *,
    dp_edges: np.ndarray,
    hs_edges: np.ndarray,
    normalize: Literal["global", "by_direction"],
    hs_colors: Sequence,
    outline_color: str = "#87CEEB",
) -> float:
    """
    Stacked column per direction sector.

    ``normalize=\"global\"``: column height = % of **all** waves from that sector;
    segment heights = % of all waves in each (direction, Hs) cell (nested distribution).
    ``normalize=\"by_direction\"``: every column forced to 100% (Hs mix only).
    """
    z = _hist2d_hs_dp_fraction(
        h_vals, d_vals, dp_edges, hs_edges, normalize=normalize
    )
    n_hs, n_dir = z.shape
    x = np.arange(n_dir, dtype=float)
    col_totals = z.sum(axis=0)

    if normalize == "global" and np.any(col_totals > 0):
        ax.bar(
            x,
            col_totals,
            width=0.92,
            fill=False,
            edgecolor=outline_color,
            linewidth=2.0,
            zorder=1,
        )

    bottom = np.zeros(n_dir, dtype=float)
    for j in range(n_hs):
        ax.bar(
            x,
            z[j, :],
            bottom=bottom,
            width=0.88,
            color=hs_colors[j],
            edgecolor="white",
            linewidth=0.5,
            align="center",
            zorder=2,
        )
        bottom = bottom + z[j, :]

    ax.set_xlim(-0.5, n_dir - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(_direction_sector_labels(dp_edges), fontsize=6, rotation=90)
    ax.grid(True, axis="y", alpha=0.25)
    return float(np.nanmax(col_totals)) if col_totals.size else 0.0


def _plot_buoy_hindcast_hs_dp_binned_figure(
    h_buoy: np.ndarray,
    d_buoy: np.ndarray,
    h_hist: np.ndarray,
    d_hist: np.ndarray,
    *,
    title: str,
    buoy_label: str = "buoy",
    hist_label: str = "hindcast bulk",
    dir_bin_deg: float = 10.0,
    hs_bin_m: float = 1.0,
    hs_range: tuple[float, float] = (0.0, 6.0),
    normalize: Literal["global", "by_direction"] = "global",
) -> None:
    """
    Stacked Hs distributions per direction sector (0–10°, 10–20°, …).

    Default (``normalize=\"global\"``): column height = % of all waves from that
    direction; coloured segments show the Hs split inside that share (nested, like
    the reference sketch). ``by_direction`` forces every column to 100%.
    """
    dp_edges, hs_edges = _hs_dp_rectangular_bin_edges(
        dir_bin_deg=dir_bin_deg,
        hs_bin_m=hs_bin_m,
        hs_range=hs_range,
    )
    n_hs = len(hs_edges) - 1
    hs_labels = _hs_bin_labels(hs_edges)
    hs_colors = _hs_bin_contrast_colors(n_hs)

    n_dir = len(dp_edges) - 1
    fig_w = max(14.0, 0.22 * n_dir)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 7.0), sharex=True)

    y_label = (
        "% within direction sector (each column = 100%)"
        if normalize == "by_direction"
        else "% of all samples (column height = share from that direction)"
    )

    max_b = _plot_stacked_hs_by_direction_sector(
        axes[0],
        h_buoy,
        d_buoy,
        dp_edges=dp_edges,
        hs_edges=hs_edges,
        normalize=normalize,
        hs_colors=hs_colors,
        outline_color="#b3d9f2",
    )
    axes[0].set_ylabel(y_label)
    axes[0].set_title(buoy_label, fontsize=10, loc="left", color="k")

    max_h = _plot_stacked_hs_by_direction_sector(
        axes[1],
        h_hist,
        d_hist,
        dp_edges=dp_edges,
        hs_edges=hs_edges,
        normalize=normalize,
        hs_colors=hs_colors,
        outline_color="#f9b3ff",
    )
    axes[1].set_ylabel(y_label)
    y_top = 100.0 if normalize == "by_direction" else max(max_b, max_h) * 1.08 + 0.5
    axes[0].set_ylim(0, y_top)
    axes[1].set_ylim(0, y_top)
    axes[1].set_xlabel("Direction sector (°)")
    axes[1].set_title(hist_label, fontsize=10, loc="left", color="fuchsia")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=hs_colors[j], edgecolor="0.25", linewidth=0.6)
        for j in range(n_hs)
    ]
    axes[0].legend(
        handles,
        hs_labels,
        title="Hs (m)",
        fontsize=7,
        title_fontsize=8,
        loc="upper right",
        ncol=min(n_hs, 4),
    )

    bin_note = f"{dir_bin_deg:g}° direction × {hs_bin_m:g} m Hs bins"
    norm_note = (
        "each direction column = 100% (Hs mix only)"
        if normalize == "by_direction"
        else "column height = % from that direction; stack = Hs split within it"
    )
    fig.suptitle(f"{title}\n{bin_note} — {norm_note}", fontsize=10, y=1.02)
    fig.tight_layout()
    plt.show()


def plot_buoy_bulk_hs_dp_binned_distribution(
    buoy_ids: str | Sequence[str],
    *,
    coordinates: Sequence[float] | Sequence[Sequence[float]] | None = None,
    use_buoy_time_range: bool = True,
    time_start=None,
    time_end=None,
    aggregation: str = "daily",
    dir_bin_deg: float = 10.0,
    hs_bin_m: float = 1.0,
    hs_range: tuple[float, float] = (0.0, 6.0),
    normalize: Literal["global", "by_direction"] = "global",
    historical_folder: str | Path | None = None,
    historical_dataset: HistoricalDataset = "merged_grids",
    project_root: str | Path | None = None,
    buoy_data_dir: str | Path = DEFAULT_BUOY_DATA_DIR,
    buoys: Mapping[str, Any] | None = None,
    max_match_distance_km: float | None = None,
) -> list[BuoyHistoricalMatch]:
    """
    Nested stacked bars: direction share × Hs mix (buoy top, hindcast bottom).

    **Direction:** first column = waves with Dp in **0–10°**, then 10–20°, …
    **Default (``normalize=\"global\"``):** column **height** = % of all waves from that
    sector (e.g. 20% from 0–10°); **segments** = how that 20% splits across Hs bins.
    ``normalize=\"by_direction\"`` hides direction weights (every column = 100%).

    Same site matching and period as ``plot_buoy_bulk_hs_dp_distribution_compare``.
    """
    _warn_native(aggregation)
    ids = [buoy_ids] if isinstance(buoy_ids, str) else [str(b) for b in buoy_ids]
    if not ids:
        raise ValueError("buoy_ids must contain at least one buoy ID")

    matches = [
        match_buoy_to_historical(
            bid,
            coordinates=coordinates,
            buoys=buoys,
            buoy_data_dir=buoy_data_dir,
            historical_folder=historical_folder,
            historical_dataset=historical_dataset,
            project_root=project_root,
        )
        for bid in ids
    ]

    hist_folder = resolve_historical_folder(
        historical_folder=historical_folder,
        historical_dataset=historical_dataset,
        project_root=project_root,
    )
    print(f"Buoy vs hindcast binned Hs/Dp — historical_folder: {hist_folder}")

    results: list[BuoyHistoricalMatch] = []

    for match in matches:
        if max_match_distance_km is not None and match.distance_km > max_match_distance_km:
            print(
                f"SKIP buoy {match.buoy_id}: nearest "
                f"{'candidate' if match.match_source == 'coordinates' else 'grid'} point "
                f"is {match.distance_km:.1f} km away (max {max_match_distance_km:.1f} km)"
            )
            results.append(match)
            continue

        lat, lon = match.coordinate

        if use_buoy_time_range:
            t0, t1 = get_buoy_observation_period(match.buoy_id, buoy_data_dir)
            if t0 is None or t1 is None:
                if time_start is None:
                    print(
                        f"SKIP buoy {match.buoy_id}: no buoy observations and no time_start"
                    )
                    results.append(match)
                    continue
                period_start = pd.Timestamp(time_start)
                period_end = pd.Timestamp(time_end) if time_end is not None else None
            else:
                period_start, period_end = t0, t1
        else:
            if time_start is None or time_end is None:
                raise ValueError(
                    "time_start and time_end are required when use_buoy_time_range=False"
                )
            period_start = pd.Timestamp(time_start)
            period_end = pd.Timestamp(time_end)

        match = BuoyHistoricalMatch(
            buoy_id=match.buoy_id,
            buoy_lat=match.buoy_lat,
            buoy_lon=match.buoy_lon,
            coordinate=match.coordinate,
            distance_km=match.distance_km,
            match_source=match.match_source,
            site_index=match.site_index,
            time_start=period_start,
            time_end=period_end,
        )
        results.append(match)

        buoy_pair, _ = load_buoy_hs_dp(
            match.buoy_id,
            buoy_data_dir=buoy_data_dir,
            aggregation=aggregation,
            time_start=period_start,
            time_end=period_end,
        )
        if buoy_pair is None:
            continue

        hist_series, site_note = _load_hs_dp_at_coordinate(
            (lat, lon),
            partition_ids=[],
            aggregation=aggregation,
            time_start=period_start,
            time_end=period_end,
            include_bulk=True,
            partitions_folder=hist_folder,
            historical_folder=hist_folder,
            project_root=project_root,
        )
        if "historical bulk" not in hist_series:
            print(f"No historical bulk Hs/Dp for buoy {match.buoy_id}")
            continue

        h_b, d_b = _aligned_hs_dp_numpy(*buoy_pair)
        h_h, d_h = _aligned_hs_dp_numpy(*hist_series["historical bulk"])
        if h_b.size == 0 or h_h.size == 0:
            print(f"SKIP buoy {match.buoy_id}: no valid Hs/Dp pairs")
            continue

        period_note = (
            f"{period_start.date()}–"
            f"{(period_end.date() if period_end is not None else 'latest')}"
        )
        title = (
            f"Binned Hs/Dp ({aggregation}) — buoy {match.buoy_id} vs hindcast "
            f"| {period_note} — {site_note}"
        )
        _plot_buoy_hindcast_hs_dp_binned_figure(
            h_b,
            d_b,
            h_h,
            d_h,
            title=title,
            buoy_label=f"buoy {match.buoy_id}",
            hist_label="hindcast bulk",
            dir_bin_deg=dir_bin_deg,
            hs_bin_m=hs_bin_m,
            hs_range=hs_range,
            normalize=normalize,
        )

    return results


def _is_historical_scenario(
    label: str,
    cfg: dict,
    *,
    historical_folder: str | Path = DEFAULT_HISTORICAL_FOLDER,
    project_root: str | Path | None = None,
) -> bool:
    if label.strip().lower() == "historical":
        return True
    return resolve_path(cfg.get("folder", ""), project_root) == resolve_path(
        historical_folder, project_root
    )


def _to_plotly_color(color) -> str:
    """Convert matplotlib color specs (e.g. grayscale '0.35', RGB tuples) for Plotly."""
    if isinstance(color, tuple):
        if len(color) == 4:
            r, g, b, a = color
            return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{a})"
        if len(color) >= 3:
            r, g, b = color[:3]
            return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
    r, g, b, _ = mcolors.to_rgba(color)
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"


def _matplotlib_line_kwargs(
    label: str,
    cfg: dict,
    color_index: int,
    default_colors,
    *,
    historical_folder: str | Path = DEFAULT_HISTORICAL_FOLDER,
    project_root: str | Path | None = None,
) -> dict:
    if _is_historical_scenario(label, cfg, historical_folder=historical_folder, project_root=project_root):
        return {
            "lw": cfg.get("lw", 0.8),
            "ls": cfg.get("ls", "-"),
            "color": cfg.get("color", "0.35"),
            "alpha": cfg.get("alpha", 0.85),
            "zorder": cfg.get("zorder", 1),
        }
    return {
        "lw": cfg.get("lw", DEFAULT_MODEL_LW),
        "ls": cfg.get("ls", "-"),
        "color": cfg.get("color", default_colors[color_index % len(default_colors)]),
        "alpha": cfg.get("alpha", DEFAULT_MODEL_ALPHA),
        "zorder": cfg.get("zorder", DEFAULT_MODEL_ZORDER + color_index),
    }


def normalize_variables(variable: str | Sequence[str]) -> list[str]:
    """Accept a single variable name or a list (e.g. ``\"hs\"`` or ``[\"hs\", \"tp\", \"dp\"]``)."""
    if isinstance(variable, str):
        return [variable]
    return list(variable)


def _coord_key(lat: float, lon: float) -> tuple[float, float]:
    return (round(float(lat), 6), round(float(lon), 6))


def normalize_shoreline_orientations(
    coordinates: Sequence[tuple[float, float]],
    shoreline_orientation_deg: float | Sequence[float] | Mapping[tuple[float, float], float],
) -> list[tuple[tuple[float, float], float]]:
    """
    Pair each coordinate with a shoreline orientation angle (degrees).

    ``shoreline_orientation_deg`` may be:

    - a **scalar** — same angle for every coordinate;
    - a **sequence** of angles — one per coordinate, same order as ``coordinates``;
    - a **mapping** ``{(lat, lon): angle, ...}`` — angles keyed by site.
    """
    coords = [(float(lat), float(lon)) for lat, lon in coordinates]
    n = len(coords)
    if n == 0:
        return []

    if isinstance(shoreline_orientation_deg, (int, float)):
        angle = float(shoreline_orientation_deg)
        return [(c, angle) for c in coords]

    if isinstance(shoreline_orientation_deg, Mapping):
        lookup = {_coord_key(*k): float(v) for k, v in shoreline_orientation_deg.items()}
        out: list[tuple[tuple[float, float], float]] = []
        for c in coords:
            key = _coord_key(*c)
            if key not in lookup:
                raise KeyError(
                    f"No shoreline_orientation_deg for coordinate {c}. "
                    f"Keys (rounded): {list(lookup.keys())}"
                )
            out.append((c, lookup[key]))
        return out

    angles = [float(a) for a in shoreline_orientation_deg]
    if len(angles) != n:
        raise ValueError(
            f"shoreline_orientation_deg must have one angle per coordinate "
            f"({n} coordinates, got {len(angles)} angles)."
        )
    return list(zip(coords, angles))


@dataclass(frozen=True)
class ShorelineOrientationResult:
    coordinate: tuple[float, float]
    angle_deg: float
    alongshore_deg: float
    seaward_deg: float
    distance_to_coast_km: float


@dataclass(frozen=True)
class BuoyHistoricalMatch:
    """Nearest historical/model coordinate for a chosen NDBC buoy."""

    buoy_id: str
    buoy_lat: float
    buoy_lon: float
    coordinate: tuple[float, float]
    distance_km: float
    match_source: Literal["coordinates", "historical_grid", "catalog_waves"]
    site_index: int | None = None
    time_start: pd.Timestamp | None = None
    time_end: pd.Timestamp | None = None
    shore_normal_deg: float | None = None


def _geod_azimuth_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from pyproj import Geod

    az, _, _ = Geod(ellps="WGS84").inv(lon1, lat1, lon2, lat2)
    return float((az + 360.0) % 360.0)


def _angle_distance_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


@lru_cache(maxsize=4)
def _load_coastline_lines(
    bbox: tuple[float, float, float, float] = NC_COAST_BBOX,
    resolution: str = "10m",
) -> tuple:
    """Load Natural Earth coastline segments within ``bbox`` (cached)."""
    from shapely.geometry import LineString, MultiLineString

    import cartopy.io.shapereader as shpreader

    path = shpreader.natural_earth(resolution=resolution, category="physical", name="coastline")
    lines = []
    for geom in shpreader.Reader(path).geometries():
        minx, miny, maxx, maxy = geom.bounds
        if maxx < bbox[0] or minx > bbox[2] or maxy < bbox[1] or miny > bbox[3]:
            continue
        if isinstance(geom, LineString):
            lines.append(geom)
        elif isinstance(geom, MultiLineString):
            lines.extend(geom.geoms)
    return tuple(lines)


def _nearest_coastline_line(lat: float, lon: float, lines: Sequence) -> tuple:
    from shapely.geometry import Point

    pt = Point(lon, lat)
    best, best_d = None, float("inf")
    for line in lines:
        d = line.distance(pt)
        if d < best_d:
            best_d, best = d, line
    if best is None:
        raise RuntimeError("No coastline geometry found in the requested bbox.")
    return best, best_d


def _geodesic_offset_on_line(line, lat0: float, lon0: float, offset_m: float) -> tuple[float, float]:
    """Move ``offset_m`` along the polyline from the vertex nearest (lat0, lon0)."""
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    coords = list(line.coords)
    dists = [
        np.hypot(
            (lon - lon0) * 111_320.0 * np.cos(np.radians(lat0)),
            (lat - lat0) * 111_320.0,
        )
        for lon, lat in coords
    ]
    i0 = int(np.argmin(dists))

    def march(start_i: int, direction: int, target_m: float) -> tuple[float, float]:
        i, acc = start_i, 0.0
        la, lo = coords[i][1], coords[i][0]
        while 0 <= i + direction < len(coords) and acc < target_m:
            j = i + direction
            la2, lo2 = coords[j][1], coords[j][0]
            _, _, seg = geod.inv(lo, la, lo2, la2)
            if seg <= 0:
                break
            if acc + seg >= target_m:
                frac = (target_m - acc) / seg
                return la + frac * (la2 - la), lo + frac * (lo2 - lo)
            acc += seg
            i, la, lo = j, la2, lo2
        j = min(max(i + direction, 0), len(coords) - 1)
        return coords[j][1], coords[j][0]

    return march(i0, -1, abs(offset_m)), march(i0, 1, abs(offset_m))


def compute_shoreline_orientations_from_coastline(
    coordinates: Sequence[float] | Sequence[Sequence[float]],
    *,
    convention: Literal["seaward", "alongshore"] = "seaward",
    window_m: float = 4000.0,
    ocean_reference: tuple[float, float] = (34.0, -74.0),
    bbox: tuple[float, float, float, float] = NC_COAST_BBOX,
    coastline_resolution: str = "10m",
) -> list[ShorelineOrientationResult]:
    """
    Estimate shoreline orientation at each coordinate from Natural Earth coastline.

    Uses a local tangent to the 10 m coast polyline (geodesic step ``window_m`` alongshore).

    Conventions (for ``alpha = wave_direction - shoreline_orientation`` in CERC Qs):

    - ``seaward`` (default): direction the coast faces toward the open ocean (~E/SE for NC).
      Matches ALR notebooks (e.g. Duck ~69–74°).
    - ``alongshore``: direction along the coast (SW–NE on the Outer Banks).
    """
    coords = normalize_coordinates(coordinates)
    lines = _load_coastline_lines(bbox=bbox, resolution=coastline_resolution)
    results: list[ShorelineOrientationResult] = []

    for lat, lon in coords:
        line, plane_dist = _nearest_coastline_line(lat, lon, lines)
        (lat_b, lon_b), (lat_f, lon_f) = _geodesic_offset_on_line(line, lat, lon, window_m)
        az_fwd = _geod_azimuth_deg(lat_b, lon_b, lat_f, lon_f)
        az_rev = (az_fwd + 180.0) % 360.0

        ocean_az = _geod_azimuth_deg(lat, lon, ocean_reference[0], ocean_reference[1])
        if lat > 34.5:
            along = az_fwd if _angle_distance_deg(az_fwd, 45.0) < _angle_distance_deg(az_rev, 45.0) else az_rev
        else:
            along = az_fwd if _angle_distance_deg(az_fwd, 135.0) < _angle_distance_deg(az_rev, 135.0) else az_rev

        norm_a = (along + 90.0) % 360.0
        norm_b = (along - 90.0) % 360.0
        seaward = norm_a if _angle_distance_deg(norm_a, ocean_az) < _angle_distance_deg(norm_b, ocean_az) else norm_b

        angle = seaward if convention == "seaward" else along
        dist_km = float(plane_dist * 111.32 * np.cos(np.radians(lat)))
        results.append(
            ShorelineOrientationResult(
                coordinate=(lat, lon),
                angle_deg=float(angle),
                alongshore_deg=float(along),
                seaward_deg=float(seaward),
                distance_to_coast_km=dist_km,
            )
        )
    return results


def shoreline_angles_for_coordinates(
    coordinates: Sequence[float] | Sequence[Sequence[float]],
    **kwargs,
) -> list[float]:
    """Return only the orientation angles (see ``compute_shoreline_orientations_from_coastline``)."""
    return [r.angle_deg for r in compute_shoreline_orientations_from_coastline(coordinates, **kwargs)]


def _shore_angles_for_buoy_matches(
    matches: Sequence[BuoyHistoricalMatch],
    *,
    buoys: Mapping[str, Any] | None,
    shoreline_orientation_deg: float | Sequence[float] | Mapping[tuple[float, float], float] | None,
    shoreline_convention: Literal["seaward", "alongshore"],
    use_buoy_catalog_shore_normal: bool = True,
) -> tuple[list[tuple[tuple[float, float], float]], list[ShorelineOrientationResult | None]]:
    """
    Pair each match with the CERC shoreline angle (degrees from N, clockwise).

    Priority: explicit ``shoreline_orientation_deg`` > buoy catalog ``sn`` >
    Natural Earth coastline estimate.
    """
    coordinates = [m.coordinate for m in matches]
    if shoreline_orientation_deg is not None:
        return (
            normalize_shoreline_orientations(coordinates, shoreline_orientation_deg),
            [None] * len(matches),
        )

    sn_lookup = (
        {k.lower(): v for k, v in buoy_shore_normals_from_catalog(buoys).items()}
        if use_buoy_catalog_shore_normal
        else {}
    )

    pending: list[tuple[tuple[float, float], float] | None] = []
    orient_results: list[ShorelineOrientationResult | None] = [None] * len(matches)
    coast_coords: list[tuple[float, float]] = []

    for m in matches:
        sn = sn_lookup.get(m.buoy_id.lower())
        if sn is not None:
            pending.append((m.coordinate, float(sn)))
        else:
            pending.append(None)
            coast_coords.append(m.coordinate)

    if coast_coords:
        coast = compute_shoreline_orientations_from_coastline(
            coast_coords,
            convention=shoreline_convention,
        )
        ci = 0
        for i, item in enumerate(pending):
            if item is None:
                pending[i] = (matches[i].coordinate, coast[ci].angle_deg)
                orient_results[i] = coast[ci]
                ci += 1

    return pending, orient_results  # type: ignore[return-value]


def _print_qs_shoreline_orientation(
    coordinate: tuple[float, float],
    shore_deg_used: float,
    *,
    convention: Literal["seaward", "alongshore"],
    orient: ShorelineOrientationResult | None = None,
    shore_normal_source: str | None = None,
    buoy_id: str | None = None,
) -> None:
    """Report beach alongshore azimuth and the angle passed to CERC Qs for one site."""
    lat, lon = coordinate
    if shore_normal_source == "buoy_catalog":
        buoy_bit = f" buoy {buoy_id}" if buoy_id else ""
        print(
            f"  Shoreline @ ({lat:.3f}, {lon:.3f}){buoy_bit} — "
            f"shore-normal from catalog (sn): {shore_deg_used:.1f}° "
            f"(alpha = Dp − sn; degrees from N, clockwise)"
        )
        return
    if orient is not None:
        conv_note = (
            "shore-normal seaward"
            if convention == "seaward"
            else "alongshore (beach)"
        )
        print(
            f"  Shoreline @ ({lat:.3f}, {lon:.3f}) [Natural Earth coast, "
            f"{orient.distance_to_coast_km:.2f} km to coast]:\n"
            f"    Beach alongshore direction: {orient.alongshore_deg:.1f}° (from N, clockwise)\n"
            f"    Shore-normal seaward (toward ocean): {orient.seaward_deg:.1f}°\n"
            f"    Angle used in Qs — {conv_note}: {shore_deg_used:.1f}° "
            f"(alpha = Dp − angle; convention={convention!r})"
        )
        return
    print(
        f"  Shoreline @ ({lat:.3f}, {lon:.3f}) — user-supplied angle for Qs: "
        f"{shore_deg_used:.1f}° (alpha = Dp − angle)"
    )


def _warn_native(aggregation: str) -> None:
    if str(aggregation).lower() in ("native", "hourly"):
        print(
            "Note: hourly/native resolution — large plots (~10⁵ points per series). "
            "Use aggregation='daily' or 'monthly' for faster summaries."
        )


def _plot_series_on_ax(
    ax,
    series: dict[str, tuple[xr.DataArray, dict]],
    *,
    variable: str,
    scenarios_cfg: Sequence[tuple[str, dict]],
    historical_folder: str | Path = DEFAULT_HISTORICAL_FOLDER,
    project_root: str | Path | None = None,
    show_legend: bool = True,
) -> None:
    cfg_by_label = {label: cfg for label, cfg in scenarios_cfg}
    default_colors = plt.cm.tab10.colors

    items = list(series.items())
    items.sort(
        key=lambda item: _is_historical_scenario(
            item[0],
            cfg_by_label.get(item[0], {}),
            historical_folder=historical_folder,
            project_root=project_root,
        )
    )

    model_color_i = 0
    for label, (da, _info) in items:
        cfg = cfg_by_label.get(label, {})
        is_hist = _is_historical_scenario(
            label, cfg, historical_folder=historical_folder, project_root=project_root
        )
        style = _matplotlib_line_kwargs(
            label,
            cfg,
            model_color_i,
            default_colors,
            historical_folder=historical_folder,
            project_root=project_root,
        )
        if not is_hist:
            model_color_i += 1
        _plot_timeseries_on_ax(
            ax,
            da["time"].values,
            da.values,
            directional=_is_directional_name(variable),
            label=label,
            **style,
        )

    ax.set_ylabel(variable.upper())
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(loc="upper left", fontsize=7, ncol=2)


def plot_timeseries_matplotlib(
    series: dict[str, tuple[xr.DataArray, dict]],
    *,
    variable: str,
    title: str,
    scenarios_cfg: Sequence[tuple[str, dict]],
    historical_folder: str | Path = DEFAULT_HISTORICAL_FOLDER,
    project_root: str | Path | None = None,
    time_highlights: Sequence[Any] | None = None,
) -> None:
    highlights = _normalize_time_highlights(time_highlights)
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    _plot_series_on_ax(
        ax,
        series,
        variable=variable,
        scenarios_cfg=scenarios_cfg,
        historical_folder=historical_folder,
        project_root=project_root,
    )
    _apply_time_highlights_matplotlib(ax, highlights)
    ax.set_xlabel("Time")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_timeseries_matplotlib_multi(
    bundles: dict[str, dict[str, tuple[xr.DataArray, dict]]],
    *,
    title: str,
    scenarios_cfg: Sequence[tuple[str, dict]],
    historical_folder: str | Path = DEFAULT_HISTORICAL_FOLDER,
    project_root: str | Path | None = None,
) -> None:
    """Stacked subplots (n variables × 1 column), shared x-axis."""
    variables = list(bundles.keys())
    n = len(variables)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True, squeeze=False)
    axes = np.atleast_1d(axes).ravel()

    for i, (var, series) in enumerate(bundles.items()):
        _plot_series_on_ax(
            axes[i],
            series,
            variable=var,
            scenarios_cfg=scenarios_cfg,
            historical_folder=historical_folder,
            project_root=project_root,
            show_legend=(i == n - 1),
        )

    axes[-1].set_xlabel("Time")
    fig.suptitle(title, y=1.01, fontsize=11)
    plt.tight_layout()
    plt.show()


def _add_series_traces_plotly(
    fig,
    series: dict[str, tuple[xr.DataArray, dict]],
    *,
    scenarios_cfg: Sequence[tuple[str, dict]],
    row: int | None = None,
    col: int | None = None,
    show_legend: bool = True,
    historical_folder: str | Path = DEFAULT_HISTORICAL_FOLDER,
    project_root: str | Path | None = None,
) -> None:
    cfg_by_label = {label: cfg for label, cfg in scenarios_cfg}
    default_colors = plt.cm.tab10.colors

    items = list(series.items())
    items.sort(
        key=lambda item: _is_historical_scenario(
            item[0],
            cfg_by_label.get(item[0], {}),
            historical_folder=historical_folder,
            project_root=project_root,
        )
    )

    model_color_i = 0
    for label, (da, _info) in items:
        cfg = cfg_by_label.get(label, {})
        is_hist = _is_historical_scenario(
            label, cfg, historical_folder=historical_folder, project_root=project_root
        )
        style = _matplotlib_line_kwargs(
            label,
            cfg,
            model_color_i,
            default_colors,
            historical_folder=historical_folder,
            project_root=project_root,
        )
        if not is_hist:
            model_color_i += 1
        line_color = _to_plotly_color(style["color"])
        trace_kw = dict(
            x=pd.to_datetime(da["time"].values),
            y=da.values,
            mode="lines",
            name=label,
            line=dict(
                color=line_color,
                width=style["lw"],
                dash="solid" if style["ls"] == "-" else "dash",
            ),
            opacity=style["alpha"],
            legendgroup=label,
            showlegend=show_legend,
        )
        if row is not None:
            fig.add_trace(go.Scatter(**trace_kw), row=row, col=col or 1)
        else:
            fig.add_trace(go.Scatter(**trace_kw))


def plot_timeseries_plotly(
    series: dict[str, tuple[xr.DataArray, dict]],
    *,
    variable: str,
    title: str,
    scenarios_cfg: Sequence[tuple[str, dict]],
    historical_folder: str | Path = DEFAULT_HISTORICAL_FOLDER,
    project_root: str | Path | None = None,
    time_highlights: Sequence[Any] | None = None,
) -> None:
    if not PLOTLY_AVAILABLE:
        print("Plotly not installed — pip/conda install plotly for interactive plots.")
        return

    highlights = _normalize_time_highlights(time_highlights)
    fig = go.Figure()
    _add_series_traces_plotly(
        fig,
        series,
        scenarios_cfg=scenarios_cfg,
        historical_folder=historical_folder,
        project_root=project_root,
    )
    _apply_time_highlights_plotly(fig, highlights, n_rows=1, n_cols=1)
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=variable.upper(),
        hovermode="x unified",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(rangeslider_visible=True)
    fig.show(config={"scrollZoom": True, "displayModeBar": True})


def plot_timeseries_plotly_multi(
    bundles: dict[str, dict[str, tuple[xr.DataArray, dict]]],
    *,
    title: str,
    scenarios_cfg: Sequence[tuple[str, dict]],
    historical_folder: str | Path = DEFAULT_HISTORICAL_FOLDER,
    project_root: str | Path | None = None,
) -> None:
    if not PLOTLY_AVAILABLE:
        print("Plotly not installed — pip/conda install plotly for interactive plots.")
        return

    variables = list(bundles.keys())
    n = len(variables)
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[v.upper() for v in variables],
        vertical_spacing=0.06,
    )
    for i, (var, series) in enumerate(bundles.items(), start=1):
        _add_series_traces_plotly(
            fig,
            series,
            scenarios_cfg=scenarios_cfg,
            row=i,
            col=1,
            show_legend=(i == n),
            historical_folder=historical_folder,
            project_root=project_root,
        )
        fig.update_yaxes(title_text=var.upper(), row=i, col=1)

    fig.update_layout(
        title=title,
        hovermode="x unified",
        height=max(480, 320 * n),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(rangeslider_visible=True, row=n, col=1)
    fig.show(config={"scrollZoom": True, "displayModeBar": True})


def plot_historical(
    variable: str,
    coordinates: Sequence[float] | Sequence[Sequence[float]],
    *,
    time_start=None,
    time_end=None,
    aggregation: str = "daily",
    historical_folder: str | Path = DEFAULT_HISTORICAL_FOLDER,
    project_root: str | Path | None = None,
) -> None:
    """Matplotlib time series for historical data only."""
    coords = normalize_coordinates(coordinates)
    nc_path = resolve_path(historical_folder, project_root) / f"{variable}_500m.nc"

    fig, axes = plt.subplots(len(coords), 1, figsize=(14, 3.5 * len(coords)), sharex=True, squeeze=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, (lat, lon) in zip(axes, coords):
        da, info = load_timeseries_at_coordinate(
            nc_path,
            (lat, lon),
            variable=variable,
            aggregation=aggregation,
            time_start=time_start,
            time_end=time_end,
            project_root=project_root,
        )
        ax.plot(da["time"].values, da.values, color="k", lw=1.2, label="historical")
        ax.set_ylabel(variable.upper())
        ax.set_title(
            f"Historical {variable} @ ({lat:.3f}, {lon:.3f}) → "
            f"site {info['site_index']} ({info['lat']:.3f}, {info['lon']:.3f}), "
            f"{info['distance_km']:.2f} km"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Historical — {historical_folder} / {variable}_500m.nc ({aggregation})", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_comparison(
    variable: str | Sequence[str],
    coordinates: Sequence[float] | Sequence[Sequence[float]],
    *,
    models: Sequence[str] | None = None,
    scenarios: Mapping[str, dict] | None = None,
    time_start=None,
    time_end=None,
    aggregation: str = "daily",
    interactive: bool = False,
    static: bool = True,
    historical_folder: str | Path = DEFAULT_HISTORICAL_FOLDER,
    project_root: str | Path | None = None,
) -> None:
    """
    Overlay time series for selected models (and historical if in ``models``).

    ``variable``: one name (``\"hs\"``) or several (``[\"hs\", \"tp\", \"dp\"]``) for
    stacked subplots (n rows × 1 column, shared time axis).

    ``interactive``: Plotly (legend toggles traces, zoom/pan).
    ``static``: matplotlib figure.
    """
    _warn_native(aggregation)
    variables = normalize_variables(variable)
    coords = normalize_coordinates(coordinates)
    scenarios_list = select_scenarios(
        scenarios,
        models=models,
        include_historical=True,
        historical_folder=historical_folder,
        project_root=project_root,
    )
    if not scenarios_list:
        raise ValueError("No scenarios selected.")

    multi_var = len(variables) > 1
    var_label = ", ".join(v.upper() for v in variables)

    for lat, lon in coords:
        if multi_var:
            bundles: dict[str, dict[str, tuple[xr.DataArray, dict]]] = {}
            site_note = ""
            for var in variables:
                bundles[var], note = load_scenario_bundle(
                    (lat, lon),
                    var,
                    scenarios_list,
                    aggregation=aggregation,
                    time_start=time_start,
                    time_end=time_end,
                    project_root=project_root,
                )
                if not site_note:
                    site_note = note
            title = f"{var_label} ({aggregation}) @ ({lat:.3f}, {lon:.3f}) — {site_note}"
            if static:
                plot_timeseries_matplotlib_multi(
                    bundles,
                    title=title,
                    scenarios_cfg=scenarios_list,
                    historical_folder=historical_folder,
                    project_root=project_root,
                )
            if interactive:
                n_pts = sum(
                    info["n_points"]
                    for b in bundles.values()
                    for _da, info in b.values()
                )
                print(f"Interactive: {len(variables)} panels, ~{n_pts:,} points ({aggregation})")
                plot_timeseries_plotly_multi(
                    bundles,
                    title=title,
                    scenarios_cfg=scenarios_list,
                    historical_folder=historical_folder,
                    project_root=project_root,
                )
        else:
            var = variables[0]
            series, site_note = load_scenario_bundle(
                (lat, lon),
                var,
                scenarios_list,
                aggregation=aggregation,
                time_start=time_start,
                time_end=time_end,
                project_root=project_root,
            )
            title = f"{var} ({aggregation}) @ ({lat:.3f}, {lon:.3f}) — {site_note}"
            if static:
                plot_timeseries_matplotlib(
                    series,
                    variable=var,
                    title=title,
                    scenarios_cfg=scenarios_list,
                    historical_folder=historical_folder,
                    project_root=project_root,
                )
            if interactive:
                n_pts = sum(info["n_points"] for _da, info in series.values())
                print(f"Interactive: {len(series)} series, ~{n_pts:,} points ({aggregation})")
                plot_timeseries_plotly(
                    series,
                    variable=var,
                    title=title,
                    scenarios_cfg=scenarios_list,
                    historical_folder=historical_folder,
                    project_root=project_root,
                )


def plot_longshore_transport(
    coordinates: Sequence[float] | Sequence[Sequence[float]],
    shoreline_orientation_deg: float | Sequence[float] | Mapping[tuple[float, float], float],
    *,
    models: Sequence[str] | None = None,
    scenarios: Mapping[str, dict] | None = None,
    time_start=None,
    time_end=None,
    aggregation: str = "daily",
    K: float = DEFAULT_LONGSHORE_K,
    interactive: bool = False,
    static: bool = True,
    plot_cumulative: bool = False,
    historical_folder: str | Path = DEFAULT_HISTORICAL_FOLDER,
    project_root: str | Path | None = None,
    include_buoy: bool = False,
    buoy_data_dir: str | Path | None = DEFAULT_BUOY_DATA_DIR,
    buoys: Mapping[str, Sequence[float]] | None = None,
    buoy_max_distance_km: float | None = 75.0,
) -> None:
    """
    Longshore transport index (Qs) from Hs and Dp for each model vs historical.

    ``shoreline_orientation_deg``: scalar, list of angles (one per coordinate, same order),
    or dict ``{(lat, lon): angle}`` — direction (deg) the shoreline faces;
    ``alpha = wave_direction - shoreline_orientation``.

    ``aggregation``: ``\"daily\"`` or ``\"monthly\"`` (applied to Hs/Dp before Qs).

    ``plot_cumulative``: if True, also plot cumulative sum of Qs over time.

    For buoy-driven analysis (pick buoy IDs, match to historical sites), use
    ``plot_longshore_transport_for_buoys`` instead.
    """
    _warn_native(aggregation)
    coords = normalize_coordinates(coordinates)
    coord_angles = normalize_shoreline_orientations(coords, shoreline_orientation_deg)
    scenarios_list = select_scenarios(
        scenarios,
        models=models,
        include_historical=True,
        historical_folder=historical_folder,
        project_root=project_root,
    )
    if not scenarios_list:
        raise ValueError("No scenarios selected.")

    for (lat, lon), shore_deg in coord_angles:
        series, site_note = load_qs_scenario_bundle(
            (lat, lon),
            scenarios_list,
            shore_deg,
            aggregation=aggregation,
            time_start=time_start,
            time_end=time_end,
            K=K,
            project_root=project_root,
        )
        if not series:
            print(f"No Qs series loaded for ({lat:.3f}, {lon:.3f})")
            continue

        scenarios_cfg = list(scenarios_list)
        buoy_note = ""
        if include_buoy and buoy_data_dir is not None:
            series, buoy_cfg, buoy_note = _append_nearest_buoy_qs(
                series,
                (lat, lon),
                shore_deg,
                buoy_data_dir=buoy_data_dir,
                buoys=buoys,
                buoy_max_distance_km=buoy_max_distance_km,
                aggregation=aggregation,
                time_start=time_start,
                time_end=time_end,
                K=K,
            )
            scenarios_cfg.extend(buoy_cfg)

        title = (
            f"Qs CERC ({aggregation}) @ ({lat:.3f}, {lon:.3f}), "
            f"shore={shore_deg:.1f}° — {site_note}"
        )
        if buoy_note:
            title = f"{title}; {buoy_note}"
        if static:
            plot_timeseries_matplotlib(
                series,
                variable="Qs",
                title=title,
                scenarios_cfg=scenarios_cfg,
                historical_folder=historical_folder,
                project_root=project_root,
            )
            if plot_cumulative:
                cum_series = {
                    label: (da.cumsum("time"), info) for label, (da, info) in series.items()
                }
                plot_timeseries_matplotlib(
                    cum_series,
                    variable="Cumulative Qs",
                    title=f"Cumulative {title}",
                    scenarios_cfg=scenarios_cfg,
                    historical_folder=historical_folder,
                    project_root=project_root,
                )
        if interactive:
            n_pts = sum(info["n_points"] for _da, info in series.values())
            print(f"Interactive Qs: {len(series)} series, ~{n_pts:,} points ({aggregation})")
            plot_timeseries_plotly(
                series,
                variable="Qs",
                title=title,
                scenarios_cfg=scenarios_cfg,
                historical_folder=historical_folder,
                project_root=project_root,
            )


def plot_longshore_transport_for_coordinates(
    coordinates: Sequence[float] | Sequence[Sequence[float]],
    *,
    shoreline_orientation_deg: float | Sequence[float] | Mapping[tuple[float, float], float] | None = None,
    shoreline_convention: Literal["seaward", "alongshore"] = "seaward",
    time_start=None,
    time_end=None,
    aggregation: str = "daily",
    K: float = DEFAULT_LONGSHORE_K,
    interactive: bool = False,
    static: bool = True,
    plot_cumulative: bool = True,
    historical_folder: str | Path | None = None,
    historical_dataset: HistoricalDataset = "merged_grids",
    project_root: str | Path | None = None,
    include_partitions: bool = False,
    partitions_folder: str | Path | None = None,
    partitions_dataset: HistoricalDataset = "merged_grids",
    partition_ids: Sequence[int] | None = None,
    include_historical_bulk: bool = True,
    include_partitions_sum: bool = True,
    include_gcm_models: bool = False,
    models: Sequence[str] | None = None,
    scenarios: Mapping[str, dict] | None = None,
) -> None:
    """
    Plot Qs at raw coordinates (no buoy required), matching buoy-style outputs.

    This is the coordinate equivalent of ``plot_longshore_transport_for_buoys``:
    historical bulk ± partitions ± partition sum, optional GCM overlays, and
    optional cumulative Qs plot.
    """
    _warn_native(aggregation)
    coords = normalize_coordinates(coordinates)
    if not coords:
        raise ValueError("coordinates must contain at least one (lat, lon) pair")

    hist_folder = resolve_historical_folder(
        historical_folder=historical_folder,
        historical_dataset=historical_dataset,
        project_root=project_root,
    )
    part_folder = resolve_partitions_folder(
        partitions_folder=partitions_folder,
        partitions_dataset=partitions_dataset,
        project_root=project_root,
    )

    if shoreline_orientation_deg is None:
        orient_results = compute_shoreline_orientations_from_coastline(
            coords, convention=shoreline_convention
        )
        coord_angles = [(r.coordinate, r.angle_deg) for r in orient_results]
    else:
        coord_angles = normalize_shoreline_orientations(coords, shoreline_orientation_deg)
        orient_results = [None] * len(coord_angles)

    gcm_scenarios: list[tuple[str, dict]] = []
    if include_gcm_models:
        scenarios_list = select_scenarios(
            scenarios,
            models=models,
            include_historical=False,
            historical_folder=hist_folder,
            project_root=project_root,
        )
        gcm_scenarios = list(scenarios_list)

    for (coord, shore_deg), orient in zip(coord_angles, orient_results):
        lat, lon = coord
        print(f"Coordinate ({lat:.3f}, {lon:.3f}) | period {time_start} – {time_end}")
        _print_qs_shoreline_orientation(
            coord,
            shore_deg,
            convention=shoreline_convention,
            orient=orient,
        )

        pids: Sequence[int] | None = partition_ids if include_partitions else []
        series, site_note = load_qs_historical_with_partitions(
            coord,
            shore_deg,
            aggregation=aggregation,
            time_start=time_start,
            time_end=time_end,
            K=K,
            historical_folder=hist_folder,
            partitions_folder=part_folder,
            partition_ids=pids,
            include_bulk=include_historical_bulk,
            include_partitions_sum=include_partitions_sum if include_partitions else False,
            project_root=project_root,
        )

        if gcm_scenarios:
            scenario_series, scenario_note = load_qs_scenario_bundle(
                coord,
                gcm_scenarios,
                shore_deg,
                aggregation=aggregation,
                time_start=time_start,
                time_end=time_end,
                K=K,
                project_root=project_root,
            )
            series.update(scenario_series)
            if not site_note:
                site_note = scenario_note

        if not series:
            print(f"No Qs series loaded for ({lat:.3f}, {lon:.3f})")
            continue

        scenarios_cfg = _scenarios_cfg_for_qs_plot(
            series,
            gcm_scenarios=gcm_scenarios,
            historical_folder=hist_folder,
            project_root=project_root,
        )
        title = (
            f"Qs CERC ({aggregation}) @ ({lat:.3f}, {lon:.3f})"
            + (f" — {site_note}" if site_note else "")
        )
        if static:
            plot_timeseries_matplotlib(
                series,
                variable="Qs",
                title=title,
                scenarios_cfg=scenarios_cfg,
                historical_folder=hist_folder,
                project_root=project_root,
            )
            if plot_cumulative:
                cum_series = {
                    label: (da.cumsum("time"), info) for label, (da, info) in series.items()
                }
                plot_timeseries_matplotlib(
                    cum_series,
                    variable="Cumulative Qs",
                    title=f"Cumulative {title}",
                    scenarios_cfg=scenarios_cfg,
                    historical_folder=hist_folder,
                    project_root=project_root,
                )
        if interactive:
            n_pts = sum(info["n_points"] for _da, info in series.values())
            print(f"Interactive coordinate Qs: {len(series)} series, ~{n_pts:,} points ({aggregation})")
            plot_timeseries_plotly(
                series,
                variable="Qs",
                title=title,
                scenarios_cfg=scenarios_cfg,
                historical_folder=hist_folder,
                project_root=project_root,
            )


def plot_longshore_transport_for_buoys(
    buoy_ids: str | Sequence[str],
    *,
    coordinates: Sequence[float] | Sequence[Sequence[float]] | None = None,
    shoreline_orientation_deg: float | Sequence[float] | Mapping[tuple[float, float], float] | None = None,
    shoreline_convention: Literal["seaward", "alongshore"] = "seaward",
    use_buoy_time_range: bool = True,
    time_start=None,
    time_end=None,
    aggregation: str = "daily",
    K: float = DEFAULT_LONGSHORE_K,
    interactive: bool = False,
    static: bool = True,
    plot_cumulative: bool = True,
    historical_folder: str | Path | None = None,
    historical_dataset: HistoricalDataset = "merged_grids",
    project_root: str | Path | None = None,
    buoy_data_dir: str | Path = DEFAULT_BUOY_DATA_DIR,
    buoys: Mapping[str, Any] | None = None,
    use_buoy_catalog_shore_normal: bool = True,
    max_match_distance_km: float | None = None,
    include_partitions: bool = False,
    partitions_folder: str | Path | None = None,
    partitions_dataset: HistoricalDataset = "merged_grids",
    partition_ids: Sequence[int] | None = None,
    include_historical_bulk: bool = True,
    include_partitions_sum: bool = True,
    include_gcm_models: bool = False,
    models: Sequence[str] | None = None,
    scenarios: Mapping[str, dict] | None = None,
    cumulative_reference_start=None,
    time_highlights: Sequence[Any] | None = None,
) -> list[BuoyHistoricalMatch]:
    """
    Compare buoy Qs with historical waves from ``merged_grids`` (no GCMs by default).

    ``time_highlights``: optional shaded vertical bands on Qs (and cumulative Qs) plots.
    Each entry is ``(start, end)``, ``(start, end, color)``, ``(start, end, color, label)``,
    or a dict with ``start``/``end``, optional ``color``, ``alpha``, ``label``.

    Per buoy:

    1. Look up buoy location; find nearest historical grid site (``hs_merged_all.nc``).
    2. Set the time window from that buoy's valid observations (each buoy differs).
    3. Load historical Qs (bulk ± partitions) and buoy Qs on the **same period**.
    4. Plot on one axes (buoy vs ``historical bulk`` / partitions).

    **Time range:** with ``use_buoy_time_range=True`` (default), ``time_start`` /
    ``time_end`` are inferred from each buoy pickle. Pass ``use_buoy_time_range=False``
    to use a fixed ``time_start`` / ``time_end`` for all buoys.
    For IDs that only exist in ``buoys`` (no pickle data), the function falls back to
    ``time_start``/``time_end`` (or ``cumulative_reference_start`` as start) and plots
    historical-only curves.

    **Historical data:** ``historical_dataset=\"merged_grids\"`` (default). Use
    ``historical_dataset=\"merged_500m\"`` only if you need the 500 m merged files.

    ``include_partitions=True`` adds partition lines from the same ``merged_grids`` folder.

    **GCM models:** off by default. Set ``include_gcm_models=True`` and ``models=[...]``
    to overlay Century etc.

    **Plots:** instantaneous Qs, then cumulative ``Qs.cumsum(time)`` (``plot_cumulative``,
    default ``True``). Set ``plot_cumulative=False`` for Qs only.
    If ``cumulative_reference_start`` is provided, buoy cumulative curves are shifted so
    they start at the hindcast cumulative value at the buoy first timestamp, using
    hindcast integrated from ``cumulative_reference_start`` to the buoy start date.
    Historical (and optional GCM) Qs are also loaded from ``cumulative_reference_start``
    to ``time_end`` while buoy Qs remains constrained to the buoy observation period.

    **Time resolution:** use ``aggregation=\"hourly\"`` or ``aggregation=\"native\"`` for
    original hourly buoy + merged_grids data (no daily/monthly averaging). Slower and
    heavier than ``aggregation=\"daily\"``.

    **Shore-normal angle:** pass per-buoy ``sn`` in ``buoys`` (see ``parse_buoy_entry``).
    When ``shoreline_orientation_deg`` is ``None`` and ``use_buoy_catalog_shore_normal=True``
    (default), each buoy's ``sn`` is used for ``alpha = Dp - sn``. Buoys without ``sn`` fall
    back to coastline geometry (``shoreline_convention``).

    Returns ``BuoyHistoricalMatch`` records (coordinate, distance, time window).
    """
    _warn_native(aggregation)
    ids = [buoy_ids] if isinstance(buoy_ids, str) else [str(b) for b in buoy_ids]
    if not ids:
        raise ValueError("buoy_ids must contain at least one buoy ID")

    matches = [
        match_buoy_to_historical(
            bid,
            coordinates=coordinates,
            buoys=buoys,
            buoy_data_dir=buoy_data_dir,
            historical_folder=historical_folder,
            historical_dataset=historical_dataset,
            project_root=project_root,
        )
        for bid in ids
    ]

    hist_folder = resolve_historical_folder(
        historical_folder=historical_folder,
        historical_dataset=historical_dataset,
        project_root=project_root,
    )
    part_folder = resolve_partitions_folder(
        partitions_folder=partitions_folder,
        partitions_dataset=partitions_dataset,
        project_root=project_root,
    )

    coord_angles, orient_per_match = _shore_angles_for_buoy_matches(
        matches,
        buoys=buoys,
        shoreline_orientation_deg=shoreline_orientation_deg,
        shoreline_convention=shoreline_convention,
        use_buoy_catalog_shore_normal=use_buoy_catalog_shore_normal,
    )
    sn_lookup = {k.lower(): v for k, v in buoy_shore_normals_from_catalog(buoys).items()}

    gcm_scenarios: list[tuple[str, dict]] = []
    if include_gcm_models:
        scenarios_list = select_scenarios(
            scenarios,
            models=models,
            include_historical=False,
            historical_folder=hist_folder,
            project_root=project_root,
        )
        gcm_scenarios = list(scenarios_list)

    results: list[BuoyHistoricalMatch] = []

    for match, (coord, shore_deg), orient_result in zip(matches, coord_angles, orient_per_match):
        if max_match_distance_km is not None and match.distance_km > max_match_distance_km:
            print(
                f"SKIP buoy {match.buoy_id}: nearest "
                f"{'candidate' if match.match_source == 'coordinates' else 'grid'} point "
                f"is {match.distance_km:.1f} km away (max {max_match_distance_km:.1f} km)"
            )
            results.append(match)
            continue

        lat, lon = coord

        if use_buoy_time_range:
            t0, t1 = get_buoy_observation_period(match.buoy_id, buoy_data_dir)
            if t0 is None or t1 is None:
                fallback_start = (
                    pd.Timestamp(time_start)
                    if time_start is not None
                    else (
                        pd.Timestamp(cumulative_reference_start)
                        if cumulative_reference_start is not None
                        else None
                    )
                )
                fallback_end = pd.Timestamp(time_end) if time_end is not None else None
                if fallback_start is None:
                    print(
                        f"SKIP buoy {match.buoy_id}: no buoy observations and no fallback "
                        "start date (set time_start or cumulative_reference_start)"
                    )
                    results.append(match)
                    continue
                buoy_time_start, buoy_time_end = fallback_start, fallback_end
                print(
                    f"INFO buoy {match.buoy_id}: no buoy observations; using fallback period "
                    f"{buoy_time_start.date()} – "
                    f"{(buoy_time_end.date() if buoy_time_end is not None else 'latest')}"
                )
            else:
                buoy_time_start, buoy_time_end = t0, t1
        else:
            if time_start is None or time_end is None:
                raise ValueError(
                    "time_start and time_end are required when use_buoy_time_range=False"
                )
            buoy_time_start = pd.Timestamp(time_start)
            buoy_time_end = pd.Timestamp(time_end)

        history_time_start = buoy_time_start
        if cumulative_reference_start is not None:
            ref_start = pd.Timestamp(cumulative_reference_start)
            if buoy_time_end is None or ref_start <= buoy_time_end:
                history_time_start = ref_start

        sn_deg = sn_lookup.get(match.buoy_id.lower())
        match = BuoyHistoricalMatch(
            buoy_id=match.buoy_id,
            buoy_lat=match.buoy_lat,
            buoy_lon=match.buoy_lon,
            coordinate=match.coordinate,
            distance_km=match.distance_km,
            match_source=match.match_source,
            site_index=match.site_index,
            time_start=buoy_time_start,
            time_end=buoy_time_end,
            shore_normal_deg=sn_deg if sn_deg is not None else shore_deg,
        )
        results.append(match)

        print(
            f"Buoy {match.buoy_id} @ ({match.buoy_lat:.3f}, {match.buoy_lon:.3f}) → "
            f"historical ({lat:.3f}, {lon:.3f}), {match.distance_km:.1f} km "
            f"[{match.match_source}"
            + (f", site {match.site_index}" if match.site_index is not None else "")
            + f"] | period {buoy_time_start.date()} – "
            + (f"{buoy_time_end.date()}" if buoy_time_end is not None else "latest")
        )
        _print_qs_shoreline_orientation(
            coord,
            shore_deg,
            convention=shoreline_convention,
            orient=orient_result,
            shore_normal_source="buoy_catalog" if sn_deg is not None else None,
            buoy_id=match.buoy_id,
        )

        series: dict[str, tuple[xr.DataArray, dict]] = {}
        pids: Sequence[int] | None = partition_ids
        if not include_partitions:
            pids = []

        hist_series, site_note = load_qs_historical_with_partitions(
            coord,
            shore_deg,
            aggregation=aggregation,
            time_start=history_time_start,
            time_end=buoy_time_end,
            K=K,
            historical_folder=hist_folder,
            partitions_folder=part_folder,
            partition_ids=pids,
            include_bulk=include_historical_bulk,
            include_partitions_sum=include_partitions_sum if include_partitions else False,
            project_root=project_root,
        )
        series.update(hist_series)

        if gcm_scenarios:
            scenario_series, scenario_note = load_qs_scenario_bundle(
                coord,
                gcm_scenarios,
                shore_deg,
                aggregation=aggregation,
                time_start=history_time_start,
                time_end=buoy_time_end,
                K=K,
                project_root=project_root,
            )
            series.update(scenario_series)
            if not site_note:
                site_note = scenario_note

        qs_buoy, _buoy_info = load_buoy_qs(
            match.buoy_id,
            shore_deg,
            buoy_data_dir=buoy_data_dir,
            aggregation=aggregation,
            time_start=buoy_time_start,
            time_end=buoy_time_end,
            K=K,
        )
        buoy_label = f"buoy {match.buoy_id}"
        if qs_buoy is None:
            print(f"No buoy Qs for {match.buoy_id} in selected period")
        else:
            series[buoy_label] = (qs_buoy, _buoy_info)

        if not hist_series:
            print(f"No historical Qs for buoy {match.buoy_id} at ({lat:.3f}, {lon:.3f})")
            continue

        scenarios_cfg = _scenarios_cfg_for_qs_plot(
            series,
            gcm_scenarios=gcm_scenarios,
            historical_folder=hist_folder,
            project_root=project_root,
        )

        title = (
            f"Qs CERC ({aggregation}) — buoy {match.buoy_id} vs merged_grids "
            f"@ ({lat:.3f}, {lon:.3f}), hist {history_time_start.date()}–"
            f"{(buoy_time_end.date() if buoy_time_end is not None else 'latest')} "
            f"(buoy {buoy_time_start.date()}–"
            f"{(buoy_time_end.date() if buoy_time_end is not None else 'latest')}) — {site_note}"
        )
        if static:
            plot_timeseries_matplotlib(
                series,
                variable="Qs",
                title=title,
                scenarios_cfg=scenarios_cfg,
                historical_folder=hist_folder,
                project_root=project_root,
                time_highlights=time_highlights,
            )
            if plot_cumulative:
                buoy_cum_offset = None
                if cumulative_reference_start is not None and "historical bulk" in series:
                    try:
                        hist_bulk_da = series["historical bulk"][0]
                        hist_up_to_buoy = hist_bulk_da.sel(time=slice(None, buoy_time_start))
                        if hist_up_to_buoy.sizes.get("time", 0) > 0:
                            buoy_cum_offset = hist_up_to_buoy.cumsum("time").isel(time=-1)
                    except Exception as exc:
                        print(
                            "WARN cumulative_reference_start: could not compute hindcast "
                            f"offset ({exc})"
                        )

                cum_series = {}
                for label, (da, info) in series.items():
                    cum_da = da.cumsum("time")
                    if buoy_cum_offset is not None and label.startswith("buoy "):
                        if cum_da.sizes.get("time", 0) > 0:
                            first_val = cum_da.isel(time=0)
                            cum_da = cum_da - first_val + buoy_cum_offset
                    cum_series[label] = (cum_da, info)
                plot_timeseries_matplotlib(
                    cum_series,
                    variable="Cumulative Qs",
                    title=f"Cumulative {title}",
                    scenarios_cfg=scenarios_cfg,
                    historical_folder=hist_folder,
                    project_root=project_root,
                    time_highlights=time_highlights,
                )
        if interactive:
            n_pts = sum(info["n_points"] for _da, info in series.values())
            print(f"Interactive Qs: {len(series)} series, ~{n_pts:,} points ({aggregation})")
            plot_timeseries_plotly(
                series,
                variable="Qs",
                title=title,
                scenarios_cfg=scenarios_cfg,
                historical_folder=hist_folder,
                project_root=project_root,
                time_highlights=time_highlights,
            )

    return results


def _scatter_var_label(var: str) -> str:
    return {"dp": "DP", "dir": "DP"}.get(var.lower(), var.upper())


def _plot_overlap_scatter_ax(ax, hist_da: xr.DataArray, model_da: xr.DataArray, var: str, *, bins: int = 65) -> None:
    hist_ov, model_ov = xr.align(hist_da, model_da, join="inner")
    sx = np.asarray(hist_ov.values).ravel()
    sy = np.asarray(model_ov.values).ravel()
    valid = np.isfinite(sx) & np.isfinite(sy)
    sx = sx[valid]
    sy = sy[valid]

    if sx.size == 0:
        ax.set_title(f"{_scatter_var_label(var)} overlap scatter (no overlap)")
        ax.set_xlabel("Historical (overlap)")
        ax.set_ylabel("Model (overlap)")
        ax.grid(True, alpha=0.3)
        return

    z = _point_density_from_hist2d(sx, sy, bins=bins)
    order = np.argsort(z)
    ax.scatter(sx[order], sy[order], c=z[order], s=8, cmap="YlGnBu_r", alpha=0.75, edgecolors="none")
    lo = min(sx.min(), sy.min())
    hi = max(sx.max(), sy.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    rms = np.sqrt(np.mean((sy - sx) ** 2))
    bias = np.mean(sy - sx)
    if sx.size > 1:
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
    ax.set_title(f"{_scatter_var_label(var)} overlap scatter")
    ax.set_xlabel("Historical (overlap)")
    ax.set_ylabel("Model (overlap)")
    ax.grid(True, alpha=0.3)


def plot_scatter_vs_historical(
    model_label: str,
    model_folder: str | Path,
    coordinate: tuple[float, float],
    *,
    variables: Sequence[str] = DEFAULT_SCATTER_VARIABLES,
    time_start=None,
    time_end=None,
    aggregation: str = "monthly",
    historical_folder: str | Path = DEFAULT_HISTORICAL_FOLDER,
    project_root: str | Path | None = None,
) -> None:
    """One figure (hs, tp, dp, …) scatter: model vs historical with time overlap."""
    historical_folder = resolve_path(historical_folder, project_root)
    model_folder = resolve_path(model_folder, project_root)

    hist_series: dict[str, xr.DataArray] = {}
    model_series: dict[str, xr.DataArray] = {}
    for var in variables:
        hist_nc = historical_folder / f"{var}_500m.nc"
        model_nc = model_folder / f"{var}_500m.nc"
        if not hist_nc.is_file():
            print(f"SKIP {var}: missing historical {hist_nc}")
            continue
        if not model_nc.is_file():
            print(f"SKIP {var} for {model_label}: missing {model_nc}")
            continue
        hist_series[var], _ = load_timeseries_at_coordinate(
            hist_nc, coordinate, variable=var, aggregation=aggregation,
            time_start=time_start, time_end=time_end, project_root=project_root,
        )
        model_series[var], _ = load_timeseries_at_coordinate(
            model_nc, coordinate, variable=var, aggregation=aggregation,
            time_start=time_start, time_end=time_end, project_root=project_root,
        )

    vars_ok = [v for v in variables if v in hist_series and v in model_series]
    if not vars_ok:
        print(f"No variables to plot for {model_label}")
        return

    lat, lon = coordinate
    fig, axes = plt.subplots(1, len(vars_ok), figsize=(5.2 * len(vars_ok), 4.8), squeeze=False)
    axes = np.atleast_1d(axes).ravel()
    for ax, var in zip(axes, vars_ok):
        _plot_overlap_scatter_ax(ax, hist_series[var], model_series[var], var)

    plt.suptitle(
        f"Scatter ({aggregation}): {model_label} vs historical @ ({lat:.3f}, {lon:.3f})",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


def plot_scatter_all_models(
    coordinates: Sequence[float] | Sequence[Sequence[float]],
    *,
    models: Sequence[str] | None = None,
    variables: Sequence[str] = DEFAULT_SCATTER_VARIABLES,
    time_start=None,
    time_end=None,
    aggregation: str = "monthly",
    scenarios: Mapping[str, dict] | None = None,
    historical_folder: str | Path = DEFAULT_HISTORICAL_FOLDER,
    project_root: str | Path | None = None,
) -> None:
    """Scatter panels for each GCM vs historical (excludes historical from ``models``)."""
    coords = normalize_coordinates(coordinates)
    gcm_runs = select_scenarios(
        scenarios,
        models=models,
        include_historical=False,
        historical_folder=historical_folder,
        project_root=project_root,
    )
    if not gcm_runs:
        raise ValueError("No GCM models selected.")

    for lat, lon in coords:
        for label, cfg in gcm_runs:
            print(f"\n=== {label} ===")
            plot_scatter_vs_historical(
                label,
                cfg["folder"],
                (lat, lon),
                variables=variables,
                time_start=time_start,
                time_end=time_end,
                aggregation=aggregation,
                historical_folder=historical_folder,
                project_root=project_root,
            )


def _validation_metrics_text(obs: np.ndarray, model: np.ndarray) -> str:
    """MAE, RMSE, R², SI, and Bias for buoy vs model validation scatter panels."""
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


def _plot_buoy_validation_scatter_panel(
    ax,
    obs: np.ndarray,
    model: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    lim: tuple[float, float],
    scatter_cmap: str = "plasma",
    scatter_alpha: float = 0.6,
    density_bins: int = 65,
) -> None:
    valid = np.isfinite(obs) & np.isfinite(model)
    ox = obs[valid]
    oy = model[valid]
    if ox.size == 0:
        ax.text(0.5, 0.5, "no overlap", transform=ax.transAxes, ha="center", color="white")
        ax.set_xlabel(xlabel, color="white")
        ax.set_ylabel(ylabel, color="white")
        return

    density = _point_density_from_hist2d(ox, oy, bins=density_bins)
    order = np.argsort(density)
    sc = ax.scatter(
        ox[order],
        oy[order],
        c=density[order],
        cmap=scatter_cmap,
        s=1,
        alpha=scatter_alpha,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors="white")

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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("white")
    ax.spines["bottom"].set_color("white")
    ax.set_facecolor((0, 0, 0, 0))
    ax.patch.set_alpha(0.0)


def _align_buoy_hist_bulk_series(
    buoy_bulk: Mapping[str, xr.DataArray | None],
    hist_bulk: Mapping[str, xr.DataArray],
) -> dict[str, xr.DataArray | None]:
    """Inner-join buoy and hindcast Hs/Tp/Dp on a common hourly (or aggregated) time index."""
    arrays: list[xr.DataArray] = [
        buoy_bulk["hs"],  # type: ignore[index]
        hist_bulk["hs"],
        buoy_bulk["dp"],  # type: ignore[index]
        hist_bulk["dp"],
    ]
    keys = ["hs_b", "hs_m", "dp_b", "dp_m"]
    if buoy_bulk.get("tp") is not None and "tp" in hist_bulk:
        arrays.extend([buoy_bulk["tp"], hist_bulk["tp"]])  # type: ignore[list-item]
        keys.extend(["tp_b", "tp_m"])
    aligned = xr.align(*arrays, join="inner")
    return dict(zip(keys, aligned))


def _buoy_validation_dataframe(aligned: Mapping[str, xr.DataArray | None]) -> pd.DataFrame:
    """ShoreShop-style CSV: Hs/Tp/Dir for buoy and BinWaves hindcast."""
    hs_b = aligned["hs_b"]
    assert hs_b is not None
    n = int(hs_b.sizes.get("time", 0))
    nan_col = np.full(n, np.nan, dtype=float)
    tp_b = aligned.get("tp_b")
    tp_m = aligned.get("tp_m")
    df = pd.DataFrame(
        {
            "Hs_Buoy": np.asarray(hs_b.values, dtype=float),
            "Hs_BinWaves": np.asarray(aligned["hs_m"].values, dtype=float),
            "Tp_Buoy": np.asarray(tp_b.values, dtype=float) if tp_b is not None else nan_col,
            "Tp_BinWaves": np.asarray(tp_m.values, dtype=float) if tp_m is not None else nan_col,
            "Dir_Buoy": np.asarray(aligned["dp_b"].values, dtype=float),
            "Dir_BinWaves": np.asarray(aligned["dp_m"].values, dtype=float),
        },
        index=pd.to_datetime(hs_b["time"].values),
    )
    return df[
        ["Hs_Buoy", "Hs_BinWaves", "Tp_Buoy", "Tp_BinWaves", "Dir_Buoy", "Dir_BinWaves"]
    ]


def _plot_buoy_bulk_validation_timeseries(
    aligned: Mapping[str, xr.DataArray | None],
    *,
    bid: str,
    model_label: str,
    lat: float,
    lon: float,
    period_note: str,
    aggregation: str,
) -> plt.Figure:
    """Three-row Hs/Tp/Dp time series: buoy vs hindcast (Dp as dots only)."""
    panel_cfg = (
        ("hs", "hs_b", "hs_m", "Hs [m]"),
        ("tp", "tp_b", "tp_m", "Tp [s]"),
        ("dp", "dp_b", "dp_m", "Dp [°]"),
    )
    n = int(aligned["hs_b"].sizes.get("time", 0))  # type: ignore[union-attr]

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
            color=BUOY_OVERLAY_COLOR,
            lw=1.0,
            alpha=0.9,
            label="Buoy",
        )
        _plot_timeseries_on_ax(
            ax,
            times,
            mod.values,
            directional=directional,
            color=BINWAVES_OVERLAY_COLOR,
            lw=1.0,
            alpha=0.85,
            label=model_label,
        )
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("time")
    fig.suptitle(
        f"Buoy {bid} vs {model_label} @ ({lat:.3f}, {lon:.3f}) | "
        f"{period_note} | n={n:,} ({aggregation})",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_buoy_bulk_validation_scatter(
    buoy_ids: str | Sequence[str],
    *,
    buoys: Mapping[str, Any] | None = None,
    buoy_data_dir: str | Path = DEFAULT_BUOY_DATA_DIR,
    historical_folder: str | Path | None = None,
    historical_dataset: HistoricalDataset = "merged_grids",
    partitions_dataset: HistoricalDataset = "merged_grids",
    project_root: str | Path | None = None,
    use_buoy_time_range: bool = True,
    time_start=None,
    time_end=None,
    aggregation: str = "hourly",
    model_label: str | None = None,
    output_folder: str | Path | None = None,
    csv_output_folder: str | Path | None = None,
    save: bool = True,
    save_timeseries: bool = False,
    show: bool = False,
    dpi: int = 150,
    scatter_cmap: str = "plasma",
    scatter_alpha: float = 0.6,
    max_match_distance_km: float | None = None,
) -> list[BuoyHistoricalMatch]:
    """
    Buoy vs hindcast bulk validation scatter plots (Hs, Tp, Dir).

    Replicates the ShoreShop validation style: KDE-style density coloring on a
  transparent/black background, 1:1 dashed line, and MAE / RMSE / R² / SI / Bias.

    One 1×3 scatter figure is written per buoy when ``save=True`` and ``output_folder`` is set.
    Optional CSV export (``csv_output_folder``) and 3-row Hs/Tp/Dp time series PNG
    (``save_timeseries=True``) use the same aligned buoy/hindcast samples.

    ``partitions_dataset`` is accepted for ``**_HIST_KW`` compatibility but unused
    (this function compares buoy observations to hindcast bulk only).
    """
    ids = [buoy_ids] if isinstance(buoy_ids, str) else [str(b) for b in buoy_ids]
    if not ids:
        raise ValueError("buoy_ids must contain at least one buoy ID")

    hist_folder = resolve_historical_folder(
        historical_folder=historical_folder,
        historical_dataset=historical_dataset,
        project_root=project_root,
    )
    if model_label is None:
        model_label = (
            "BinWaves + KMA clusters"
            if _uses_kma_merged_grids_folder(hist_folder, project_root)
            else "BinWaves"
        )

    out_dir: Path | None = None
    if save or save_timeseries:
        if output_folder is None:
            raise ValueError(
                "output_folder is required when save=True or save_timeseries=True"
            )
        out_dir = resolve_path(output_folder, project_root)
        out_dir.mkdir(parents=True, exist_ok=True)

    csv_dir: Path | None = None
    if csv_output_folder is not None:
        csv_dir = resolve_path(csv_output_folder, project_root)
        csv_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Buoy bulk validation scatter — historical_folder: {hist_folder} | "
        f"model label: {model_label}"
    )

    results: list[BuoyHistoricalMatch] = []
    panel_cfg = (
        ("hs", "Hs", f"Hs - Buoy [m]", f"Hs - {model_label} [m]", (0.0, 6.0)),
        ("tp", "Tp", f"Tp - Buoy [s]", f"Tp - {model_label} [s]", (0.0, 25.0)),
        ("dp", "Dir", f"Dir - Buoy [°]", f"Dir - {model_label} [°]", (0.0, 350.0)),
    )

    for bid in ids:
        match = match_buoy_to_historical(
            bid,
            buoys=buoys,
            buoy_data_dir=buoy_data_dir,
            historical_folder=historical_folder,
            historical_dataset=historical_dataset,
            project_root=project_root,
        )
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
                raise ValueError(
                    "time_start and time_end are required when use_buoy_time_range=False"
                )
            period_start = pd.Timestamp(time_start)
            period_end = pd.Timestamp(time_end)

        buoy_bulk, buoy_info = load_buoy_hs_tp_dp(
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
        hist_bulk: dict[str, xr.DataArray] = {}
        for key in ("hs", "tp", "dp"):
            nc_path = _resolve_variable_nc_path(hist_folder, key, project_root)
            da, _info = load_timeseries_at_coordinate(
                nc_path,
                (lat, lon),
                variable=key,
                aggregation=aggregation,
                time_start=period_start,
                time_end=period_end,
                project_root=project_root,
            )
            hist_bulk[key] = da

        aligned = _align_buoy_hist_bulk_series(buoy_bulk, hist_bulk)
        hs_b = aligned["hs_b"]
        if hs_b is None or hs_b.sizes.get("time", 0) == 0:
            print(f"SKIP buoy {bid}: no overlapping Hs samples")
            continue

        period_note = (
            f"{period_start.date()} – {period_end.date()}"
            if period_end is not None
            else f"{period_start.date()} – latest"
        )
        n_samples = int(hs_b.sizes.get("time", 0))

        if csv_dir is not None:
            csv_path = csv_dir / f"buoy_validation_{bid}.csv"
            _buoy_validation_dataframe(aligned).to_csv(csv_path)
            print(f"Saved: {csv_path}")

        if save_timeseries and out_dir is not None:
            ts_fig = _plot_buoy_bulk_validation_timeseries(
                aligned,
                bid=bid,
                model_label=model_label,
                lat=lat,
                lon=lon,
                period_note=period_note,
                aggregation=aggregation,
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
            "tp": None,
            "dp": np.asarray(aligned["dp_b"].values),
        }
        mod_map = {
            "hs": np.asarray(aligned["hs_m"].values),
            "tp": None,
            "dp": np.asarray(aligned["dp_m"].values),
        }
        if aligned.get("tp_b") is not None and aligned.get("tp_m") is not None:
            obs_map["tp"] = np.asarray(aligned["tp_b"].values)
            mod_map["tp"] = np.asarray(aligned["tp_m"].values)

        for ax, (key, _title, xlab, ylab, lim) in zip(axes, panel_cfg):
            obs = obs_map.get(key)
            mod = mod_map.get(key)
            if obs is None or mod is None:
                ax.text(
                    0.5,
                    0.5,
                    f"no {key.upper()} data",
                    transform=ax.transAxes,
                    ha="center",
                    color="white",
                )
                ax.set_facecolor((0, 0, 0, 0))
                continue
            _plot_buoy_validation_scatter_panel(
                ax,
                obs,
                mod,
                xlabel=xlab,
                ylabel=ylab,
                lim=lim,
                scatter_cmap=scatter_cmap,
                scatter_alpha=scatter_alpha,
            )

        fig.suptitle(
            f"Buoy {bid} vs {model_label} @ ({lat:.3f}, {lon:.3f}) | "
            f"{period_note} | n={n_samples:,} ({aggregation})",
            color="white",
            fontsize=11,
            y=1.02,
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
