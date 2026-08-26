"""
North Carolina regional cumulative longshore sediment transport.

Polygon site selection, batch hindcast loading from ``merged_grids``, and Cartopy maps.
Supports CERC-style and deep-water Qs formulas, and Hs/Dp wave roses (buoy or hindcast).
Used by ``Sediment_transport_check.ipynb`` and ``Sediment_transport_check_Duck.ipynb``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

QsFormula = Literal["cerc", "deep"]

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd
import xarray as xr

from utils.bmu_bootstrap_timeseries import (
    _guess_lat_lon,
    _guess_site_dim,
    haversine_m,
)

# Notebook reload(st) must pick up gcm_comparison changes (plot helpers, etc.).
import importlib
from utils import gcm_comparison as _gcm_comparison

importlib.reload(_gcm_comparison)

from utils.gcm_comparison import (
    DEFAULT_BUOY_DATA_DIR,
    DEFAULT_LONGSHORE_K,
    BuoyHistoricalMatch,
    HistoricalDataset,
    _aligned_hs_dp_numpy,
    _buoy_pkl_path,
    _hs_bin_contrast_colors,
    _hs_bin_labels,
    _partition_ids_from_folder,
    _print_qs_shoreline_orientation,
    _resolve_nc_var,
    _resolve_variable_nc_path,
    _scenarios_cfg_for_qs_plot,
    _series_to_dataarray,
    _shore_angles_for_buoy_matches,
    _variable_from_nc_path,
    _warn_native,
    apply_aggregation,
    buoy_shore_normals_from_catalog,
    normalize_buoy_locations,
    get_buoy_observation_period,
    load_buoy_bulk_dataframe,
    load_buoy_hs_dp,
    load_qs_scenario_bundle,
    load_timeseries_at_coordinate,
    match_buoy_to_historical,
    plot_timeseries_matplotlib,
    plot_timeseries_plotly,
    resolve_historical_folder,
    resolve_partitions_folder,
    resolve_path,
    select_scenarios,
    BINWAVES_BULK_LABEL,
    DEFAULT_MERGED_GRIDS_FOLDER,
)

DEFAULT_DEEP_K2 = 0.023
DEFAULT_WHACS_FOLDER = Path(
    "/nfs/home/geocean/montanoj/ShoreShop2026/inputs/WHACS"
)
DEFAULT_KMA_MERGED_GRIDS_FOLDER = "outputs/BinWaves_BMUS"
DEFAULT_BINWAVES_MERGED_GRIDS_FOLDER = DEFAULT_MERGED_GRIDS_FOLDER


def _resolve_hindcast_rose_folder(
    *,
    hindcast_source: str = "binwaves_kma",
    historical_folder: str | Path | None = None,
    project_root: str | Path | None = None,
) -> tuple[Path, str]:
    """Return (folder, legend label) for hindcast bulk wave roses."""
    if historical_folder is not None:
        folder = resolve_path(historical_folder, project_root).resolve()
        if _uses_kma_merged_grids_folder(folder, project_root):
            return folder, "BinWaves + KMA clusters"
        return folder, "BinWaves"

    source = str(hindcast_source).strip().lower().replace("-", "_")
    if source in ("binwaves", "binwaves_only", "merged_grids"):
        return Path(DEFAULT_BINWAVES_MERGED_GRIDS_FOLDER).resolve(), "BinWaves"
    if source in ("binwaves_kma", "kma", "binwaves_plus_kma"):
        if project_root is None:
            raise ValueError(
                "project_root is required for hindcast_source='binwaves_kma' "
                "when historical_folder is not set"
            )
        folder = resolve_path(DEFAULT_KMA_MERGED_GRIDS_FOLDER, project_root).resolve()
        return folder, "BinWaves + KMA clusters"
    raise ValueError(
        f"hindcast_source must be 'binwaves' or 'binwaves_kma'; got {hindcast_source!r}"
    )


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
    """Legend key for hindcast bulk Qs (KMA merged grids vs plain merged_grids)."""
    if _uses_kma_merged_grids_folder(historical_folder, project_root):
        from utils import kma_cluster_swan as kcs

        return kcs.BINWAVES_PLUS_KMA_LABEL
    return "historical bulk"


def _is_historical_bulk_series_key(label: str) -> bool:
    from utils import kma_cluster_swan as kcs

    return label in ("historical bulk", kcs.BINWAVES_PLUS_KMA_LABEL, "historical")


def _resolve_binwaves_reference_folder(
    *,
    historical_folder: str | Path | None,
    historical_dataset: HistoricalDataset = "merged_grids",
    partitions_folder: str | Path | None = None,
    partitions_dataset: HistoricalDataset = "merged_grids",
    binwaves_reference_folder: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path | None:
    """Original BinWaves ``merged_grids`` folder when primary hindcast is KMA merged grids."""
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


_WHACS_REGION_SUBGRID = re.compile(r"_0[234]_")

__all__ = [
    "QsFormula",
    "RegionCumulativeQsResult",
    "DEFAULT_DEEP_K2",
    "normalize_polygon_vertices_lonlat",
    "select_historical_sites_in_polygon",
    "cerc_longshore_transport_index",
    "deep_water_longshore_transport_index",
    "compute_region_cumulative_qs",
    "plot_region_sites_map",
    "plot_region_cumulative_qs_map",
    "load_timeseries_at_site_index",
    "load_historical_hs_dp",
    "load_historical_partition_hs_dp",
    "plot_buoy_wave_rose",
    "plot_historical_wave_rose",
    "plot_historical_partition_wave_rose",
    "DEFAULT_WHACS_FOLDER",
    "DEFAULT_GEBCO_BATHYMETRY",
    "DEFAULT_KMA_MERGED_GRIDS_FOLDER",
    "DEFAULT_BINWAVES_MERGED_GRIDS_FOLDER",
    "find_nearest_whacs_site",
    "load_whacs_bulk_hs_dp",
    "load_whacs_partition_hs_dp",
    "wave_roses_from_spectra",
    "wave_roses_whacs",
    "load_qs_historical_with_partitions",
    "load_kma_combined_qs",
    "load_buoy_qs",
    "plot_longshore_transport_for_buoys",
    "plot_partition_hs_tp_dp_timeseries",
    "plot_buoy_bulk_validation_scatter",
]


@dataclass(frozen=True)
class RegionCumulativeQsResult:
    """Cumulative longshore transport (CERC Qs) at hindcast sites inside a polygon."""

    site_index: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    cumulative_qs: np.ndarray
    shoreline_orientation_deg: float
    polygon_lonlat: tuple[tuple[float, float], ...]
    coord_x: np.ndarray | None = None
    coord_y: np.ndarray | None = None
    time_start: str | None = None
    time_end: str | None = None
    aggregation: str = "daily"
    K: float = DEFAULT_LONGSHORE_K
    qs_formula: str = "cerc"
    wave_direction: str = "dp"
    include_partitions: bool = True
    historical_dataset: str = "merged_grids"


def normalize_polygon_vertices_lonlat(
    vertices: Sequence[float] | Sequence[Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    """Polygon vertices as ``(lon, lat)`` pairs (4+ corners, closed or open)."""
    if not vertices:
        raise ValueError("polygon vertices must not be empty")
    first = vertices[0]
    if isinstance(first, (int, float)):
        if len(vertices) != 4:
            raise ValueError("flat vertex list must have four numbers: lon, lat, lon, lat")
        lon1, lat1, lon2, lat2 = (float(v) for v in vertices)
        return ((lon1, lat1), (lon2, lat2))
    out: list[tuple[float, float]] = []
    for v in vertices:
        if len(v) != 2:
            raise ValueError(f"each vertex must be (lon, lat); got {v!r}")
        lon, lat = float(v[0]), float(v[1])
        out.append((lon, lat))
    return tuple(out)


def _points_in_polygon_lonlat(
    lons: np.ndarray,
    lats: np.ndarray,
    polygon_lonlat: Sequence[tuple[float, float]],
) -> np.ndarray:
    from matplotlib.path import Path as MplPath

    verts = [(float(lon), float(lat)) for lon, lat in polygon_lonlat]
    return MplPath(verts).contains_points(np.column_stack([lons.astype(float), lats.astype(float)]))


def select_historical_sites_in_polygon(
    polygon_vertices_lonlat: Sequence[float] | Sequence[Sequence[float]],
    *,
    historical_folder: str | Path | None = None,
    historical_dataset: HistoricalDataset = "merged_grids",
    project_root: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """
    Hindcast sites from ``merged_grids`` (or other historical folder) inside a lon/lat polygon.

    Returns arrays: ``site_index``, ``lat``, ``lon``, and optional ``coord_x`` / ``coord_y``.
    """
    polygon = normalize_polygon_vertices_lonlat(polygon_vertices_lonlat)
    hist_folder = resolve_historical_folder(
        historical_folder=historical_folder,
        historical_dataset=historical_dataset,
        project_root=project_root,
    )
    hs_path = _resolve_variable_nc_path(hist_folder, "hs", project_root)
    with xr.open_dataset(hs_path) as ds:
        lat_name, lon_name = _guess_lat_lon(ds)
        lats = np.asarray(ds[lat_name].values, dtype=float)
        lons = np.asarray(ds[lon_name].values, dtype=float)
        inside = _points_in_polygon_lonlat(lons, lats, polygon)
        idx = np.where(inside)[0].astype(int)
        coord_x = (
            np.asarray(ds["coord_x"].values, dtype=float)[idx]
            if "coord_x" in ds
            else None
        )
        coord_y = (
            np.asarray(ds["coord_y"].values, dtype=float)[idx]
            if "coord_y" in ds
            else None
        )
    if idx.size == 0:
        print(f"WARNING: no hindcast sites inside polygon ({len(polygon)} vertices)")
    else:
        print(f"Selected {idx.size} hindcast sites inside polygon")
    return {
        "site_index": idx,
        "lat": lats[idx],
        "lon": lons[idx],
        "coord_x": coord_x,
        "coord_y": coord_y,
        "polygon_lonlat": np.array(polygon, dtype=float),
    }


def _nearest_site_indices_for_latlon(
    nc_path: str | Path,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    *,
    project_root: str | Path | None = None,
) -> np.ndarray:
    """Map target (lat, lon) points to nearest site indices in a merged NetCDF."""
    nc_path = resolve_path(nc_path, project_root)
    target_lats = np.asarray(target_lats, dtype=float)
    target_lons = np.asarray(target_lons, dtype=float)
    with xr.open_dataset(nc_path) as ds:
        lat_name, lon_name = _guess_lat_lon(ds)
        src_lats = np.asarray(ds[lat_name].values, dtype=float)
        src_lons = np.asarray(ds[lon_name].values, dtype=float)
    out = np.empty(target_lats.shape[0], dtype=int)
    for i, (tlat, tlon) in enumerate(zip(target_lats, target_lons)):
        out[i] = int(np.argmin(haversine_m(tlat, tlon, src_lats, src_lons)))
    return out


def _harmonize_batch_site_coords(
    da: xr.DataArray,
    *,
    site_labels: np.ndarray,
    lat: xr.DataArray | None = None,
    lon: xr.DataArray | None = None,
) -> xr.DataArray:
    """Use shared site labels (and optional lat/lon) so xarray.align keeps all sites."""
    if "site" not in da.dims:
        return da
    da = da.assign_coords(site=np.asarray(site_labels, dtype=int))
    if lat is not None and lon is not None:
        da = da.assign_coords(lat=lat, lon=lon)
    return da


def _load_merged_batch_at_sites(
    nc_path: str | Path,
    site_indices: np.ndarray,
    *,
    variable: str,
    aggregation: str,
    time_start=None,
    time_end=None,
) -> xr.DataArray:
    """Load one merged NetCDF variable for many sites, then aggregate in time."""
    nc_path = Path(nc_path)
    site_indices = np.asarray(site_indices, dtype=int)
    with xr.open_dataset(nc_path) as ds:
        lat_name, lon_name = _guess_lat_lon(ds)
        site_dim = _guess_site_dim(ds, lat_name, lon_name)
        nc_var = _resolve_nc_var(ds, variable)
        da = ds[nc_var].isel({site_dim: site_indices})
        # Some merged grids (e.g. merged_grids_binwaves_bmus) store lat/lon as data_vars;
        # BinWaves_BMUS files already carry them as coords on the variable.
        if lat_name not in da.coords:
            lat = ds[lat_name].isel({site_dim: site_indices})
            lon = ds[lon_name].isel({site_dim: site_indices})
            da = da.assign_coords({lat_name: lat, lon_name: lon})
        if time_start is not None or time_end is not None:
            da = da.sel(time=slice(time_start, time_end))
        da = da.load()
    return apply_aggregation(da, variable, aggregation)


def cerc_longshore_transport_index(
    H: xr.DataArray,
    phi_deg: xr.DataArray,
    shoreline_orientation_deg: float,
    *,
    K: float = DEFAULT_LONGSHORE_K,
) -> xr.DataArray:
    """
    CERC-like Qs: ``Q = K * H^(5/2) * sin(2*alpha)``, ``alpha = phi - theta`` (degrees).

    ``phi_deg`` is peak wave direction (from N, clockwise); ``shoreline_orientation_deg`` is
    shore-normal ``theta`` (same convention).
    """
    hs_a, phi_a = xr.align(H, phi_deg, join="inner")
    alpha_rad = np.deg2rad(phi_a - float(shoreline_orientation_deg))
    return (hs_a ** (5 / 2)) * np.sin(2.0 * alpha_rad) * K


def deep_water_longshore_transport_index(
    H: xr.DataArray,
    Tp: xr.DataArray,
    phi_deg: xr.DataArray,
    shoreline_orientation_deg: float,
    *,
    K2: float = DEFAULT_DEEP_K2,
) -> xr.DataArray:
    """
    Deep-water Qs (relative units):

    ``Q_s = K_2 * T_p^(1/5) * H_0^(12/5) * cos^(6/5)(phi - theta) * sin(phi - theta)``

    ``phi_deg`` = peak wave angle (Dp), ``shoreline_orientation_deg`` = shore-normal ``theta``,
    ``Tp`` = peak period (s), ``H`` = deep-water height (Hs).
    """
    hs_a, tp_a, phi_a = xr.align(H, Tp, phi_deg, join="inner")
    alpha_rad = np.deg2rad(phi_a - float(shoreline_orientation_deg))
    cos_a = np.cos(alpha_rad)
    sin_a = np.sin(alpha_rad)
    h_pos = xr.where(hs_a > 0, hs_a, 0.0)
    t_pos = xr.where(tp_a > 0, tp_a, 0.0)
    return (
        float(K2)
        * (t_pos ** (1 / 5))
        * (h_pos ** (12 / 5))
        * (cos_a ** (6 / 5))
        * sin_a
    )


def _region_qs_batch(
    hs: xr.DataArray,
    dp: xr.DataArray,
    shoreline_orientation_deg: float,
    *,
    K: float,
    tp: xr.DataArray | None = None,
    qs_formula: QsFormula = "cerc",
) -> xr.DataArray:
    """Instantaneous Qs for many sites; scalar ``theta`` = ``shoreline_orientation_deg``."""
    if qs_formula == "deep":
        if tp is None:
            raise ValueError("qs_formula='deep' requires Tp (peak period).")
        return deep_water_longshore_transport_index(
            hs, tp, dp, shoreline_orientation_deg, K2=K
        )
    return cerc_longshore_transport_index(hs, dp, shoreline_orientation_deg, K=K)


def _partition_file_triplet(
    pid: int,
    qs_formula: QsFormula,
    *,
    wave_direction: str = "dp",
) -> tuple[str, str, str]:
    """NetCDF variable basenames (hs, tp, direction) for one partition."""
    dir_name = _normalize_wave_direction(wave_direction)
    if qs_formula == "deep":
        return f"phs{pid}", f"ptp{pid}", f"{dir_name}{pid}"
    return f"phs{pid}", "", f"{dir_name}{pid}"


def _bulk_file_triplet(
    qs_formula: QsFormula,
    *,
    wave_direction: str = "dp",
) -> tuple[str, str, str]:
    dir_name = _normalize_wave_direction(wave_direction)
    if qs_formula == "deep":
        return "hs", "tp", dir_name
    return "hs", "", dir_name


def _normalize_wave_direction(wave_direction: str) -> str:
    """Map peak/mean (or dp/dm) to NetCDF variable names ``dp`` / ``dm``."""
    name = str(wave_direction).strip().lower()
    aliases = {
        "peak": "dp",
        "dp": "dp",
        "mean": "dm",
        "dm": "dm",
    }
    if name not in aliases:
        raise ValueError(
            f"wave_direction must be 'peak' or 'mean' (or 'dp' / 'dm'); got {wave_direction!r}"
        )
    return aliases[name]


def _wave_direction_display_name(wave_direction: str) -> str:
    return "mean" if _normalize_wave_direction(wave_direction) == "dm" else "peak"


def _load_qs_inputs_at_sites(
    folder: str | Path,
    idx: np.ndarray,
    *,
    hs_name: str,
    dp_name: str,
    tp_name: str,
    aggregation: str,
    time_start,
    time_end,
    qs_formula: QsFormula,
    project_root: str | Path | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray | None]:
    """Load hs/dp (and tp for deep) for a list of site indices from one folder."""
    folder = Path(folder)
    idx = np.asarray(idx, dtype=int)
    hs_path = _resolve_variable_nc_path(folder, hs_name, project_root)
    dp_path = _resolve_variable_nc_path(folder, dp_name, project_root)
    hs = _load_merged_batch_at_sites(
        hs_path, idx, variable=hs_name, aggregation=aggregation,
        time_start=time_start, time_end=time_end,
    )
    ref_lats = np.asarray(hs["lat"].values, dtype=float)
    ref_lons = np.asarray(hs["lon"].values, dtype=float)
    site_labels = np.asarray(hs["site"].values, dtype=int)
    dp_idx = _nearest_site_indices_for_latlon(
        dp_path, ref_lats, ref_lons, project_root=project_root
    )
    dp = _load_merged_batch_at_sites(
        dp_path, dp_idx, variable=dp_name, aggregation=aggregation,
        time_start=time_start, time_end=time_end,
    )
    hs = _harmonize_batch_site_coords(hs, site_labels=site_labels)
    dp = _harmonize_batch_site_coords(
        dp, site_labels=site_labels, lat=hs["lat"], lon=hs["lon"]
    )
    tp: xr.DataArray | None = None
    if qs_formula == "deep":
        if not tp_name:
            raise ValueError("deep formula requires a Tp variable name")
        tp_path = _resolve_variable_nc_path(folder, tp_name, project_root)
        tp = _load_merged_batch_at_sites(
            tp_path, idx, variable=tp_name, aggregation=aggregation,
            time_start=time_start, time_end=time_end,
        )
        tp = _harmonize_batch_site_coords(tp, site_labels=site_labels)
    return hs, dp, tp


def compute_region_cumulative_qs(
    polygon_vertices_lonlat: Sequence[float] | Sequence[Sequence[float]],
    *,
    qs_formula: QsFormula = "cerc",
    shoreline_orientation_deg: float = 70.0,
    K: float = 0.023,
    aggregation: str = "daily",
    time_start: str | None = "1980-01-01",
    time_end: str | None = None,
    cumulative_reference_start: str | None = "1980-01-01",
    include_partitions: bool = True,
    include_bulk: bool = False,
    wave_direction: str = "dp",
    partition_ids: Sequence[int] | None = None,
    historical_folder: str | Path | None = None,
    historical_dataset: HistoricalDataset = "merged_grids",
    partitions_folder: str | Path | None = None,
    partitions_dataset: HistoricalDataset = "merged_grids",
    project_root: str | Path | None = None,
) -> RegionCumulativeQsResult:
    """
    Cumulative longshore transport at all hindcast sites inside a polygon.

    ``qs_formula``:

    - ``\"cerc\"`` (default): ``Q = K * H^(5/2) * sin(2*(phi - theta))`` with ``phi`` = wave direction.
    - ``\"deep\"``: ``Q = K_2 * T_p^(1/5) * H^(12/5) * cos^(6/5)(phi - theta) * sin(phi - theta)``
      with ``phi`` = wave direction, ``T_p`` = Tp, ``theta`` = shore-normal (``shoreline_orientation_deg``).

    ``wave_direction``: ``\"dp\"`` (peak / higher-energy direction) or ``\"dm\"`` (mean direction,
    energy-weighted in BinWaves+KMA bulk grids). Bulk uses ``dp_merged_all.nc`` or ``dm_merged_all.nc``.

    ``K`` is the CERC coefficient or ``K_2`` for the deep formula (default 0.023).
    ``shoreline_orientation_deg`` is applied at every site (degrees from N, clockwise).

    Set ``include_partitions=True`` (default) to sum partition Qs; ``include_bulk=True`` for bulk.
    ``cumulative_reference_start`` sets the integration start when ``time_start`` is None.
    """
    qs_formula = str(qs_formula).lower()  # type: ignore[assignment]
    if qs_formula not in ("cerc", "deep"):
        raise ValueError(f"qs_formula must be 'cerc' or 'deep'; got {qs_formula!r}")
    wave_direction = _normalize_wave_direction(wave_direction)

    _warn_native(aggregation)
    polygon = normalize_polygon_vertices_lonlat(polygon_vertices_lonlat)
    sites = select_historical_sites_in_polygon(
        polygon,
        historical_folder=historical_folder,
        historical_dataset=historical_dataset,
        project_root=project_root,
    )
    idx = sites["site_index"]
    if idx.size == 0:
        raise ValueError("No hindcast sites inside the polygon.")

    t_start = time_start if time_start is not None else cumulative_reference_start
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

    qs_total: xr.DataArray | None = None

    part_root = resolve_path(part_folder, project_root)

    if include_partitions:
        pids = (
            [int(i) for i in partition_ids]
            if partition_ids is not None
            else _partition_ids_from_folder(part_folder, project_root)
        )
        for pid in pids:
            hs_name, tp_name, dp_name = _partition_file_triplet(
                pid, qs_formula, wave_direction=wave_direction
            )
            try:
                hs, dp, tp = _load_qs_inputs_at_sites(
                    part_root,
                    idx,
                    hs_name=hs_name,
                    dp_name=dp_name,
                    tp_name=tp_name,
                    aggregation=aggregation,
                    time_start=t_start,
                    time_end=time_end,
                    qs_formula=qs_formula,
                    project_root=project_root,
                )
            except FileNotFoundError as exc:
                print(f"SKIP partition {pid}: {exc}")
                continue
            qs_part = _region_qs_batch(
                hs, dp, shoreline_orientation_deg, K=K, tp=tp, qs_formula=qs_formula
            )
            qs_total = qs_part if qs_total is None else qs_total + qs_part

    if include_bulk:
        hs_name, tp_name, dp_name = _bulk_file_triplet(
            qs_formula, wave_direction=wave_direction
        )
        try:
            hs, dp, tp = _load_qs_inputs_at_sites(
                hist_folder,
                idx,
                hs_name=hs_name,
                dp_name=dp_name,
                tp_name=tp_name,
                aggregation=aggregation,
                time_start=t_start,
                time_end=time_end,
                qs_formula=qs_formula,
                project_root=project_root,
            )
        except FileNotFoundError as exc:
            raise ValueError(f"Bulk Qs failed: {exc}") from exc
        qs_bulk = _region_qs_batch(
            hs, dp, shoreline_orientation_deg, K=K, tp=tp, qs_formula=qs_formula
        )
        qs_total = qs_bulk if qs_total is None else qs_total + qs_bulk

    if qs_total is None:
        raise ValueError(
            "No Qs computed (enable include_partitions and/or include_bulk; check NetCDF paths)."
        )

    cumulative = qs_total.cumsum("time").isel(time=-1)
    if "site" in cumulative.dims:
        cumulative_vals = np.asarray(cumulative.transpose("site").values, dtype=float)
    else:
        cumulative_vals = np.asarray(cumulative.values, dtype=float).ravel()
    if cumulative_vals.size != idx.size:
        raise ValueError(
            f"Computed cumulative Qs at {cumulative_vals.size} sites but {idx.size} "
            "sites were selected inside the polygon. Wave direction (dp/dm) may use a "
            "different site grid than hs/tp — check NetCDF paths."
        )

    t_end_str = (
        str(pd.Timestamp(qs_total.time.values[-1]).date())
        if qs_total.sizes.get("time", 0) > 0
        else None
    )
    t_start_str = (
        str(pd.Timestamp(qs_total.time.values[0]).date())
        if qs_total.sizes.get("time", 0) > 0
        else str(t_start)
    )
    coeff_label = "K_2" if qs_formula == "deep" else "K"
    print(
        f"Region cumulative Qs ({qs_formula}, {wave_direction.upper()}): {idx.size} sites, "
        f"theta={shoreline_orientation_deg:.1f}°, {coeff_label}={K}, "
        f"{aggregation}, {t_start_str} – {t_end_str}"
    )

    return RegionCumulativeQsResult(
        site_index=idx,
        lat=sites["lat"],
        lon=sites["lon"],
        coord_x=sites.get("coord_x"),
        coord_y=sites.get("coord_y"),
        cumulative_qs=cumulative_vals,
        shoreline_orientation_deg=float(shoreline_orientation_deg),
        polygon_lonlat=polygon,
        time_start=t_start_str,
        time_end=t_end_str,
        aggregation=str(aggregation),
        K=float(K),
        qs_formula=str(qs_formula),
        wave_direction=str(wave_direction),
        include_partitions=bool(include_partitions),
        historical_dataset=str(historical_dataset),
    )


_REGION_MAP_LAND_COLOR = "#e6e2d8"
_REGION_MAP_OCEAN_COLOR = "#5ba3d0"


def _region_map_extent(
    result: RegionCumulativeQsResult,
    *,
    margin_deg: float = 0.18,
    margin_lon_west: float | None = None,
    margin_lon_east: float | None = None,
    margin_lat: float | None = None,
) -> tuple[float, float, float, float]:
    """Map bounds with extra padding on the west/south for inland context (NC coast)."""
    lons = np.asarray(result.lon, dtype=float)
    lats = np.asarray(result.lat, dtype=float)
    poly = np.asarray(result.polygon_lonlat, dtype=float)
    m_lat = float(margin_lat if margin_lat is not None else margin_deg)
    m_west = float(margin_lon_west if margin_lon_west is not None else margin_deg * 1.6)
    m_east = float(margin_lon_east if margin_lon_east is not None else margin_deg * 0.75)
    lon_min = min(float(lons.min()), float(poly[:, 0].min())) - m_west
    lon_max = max(float(lons.max()), float(poly[:, 0].max())) + m_east
    lat_min = min(float(lats.min()), float(poly[:, 1].min())) - m_lat
    lat_max = max(float(lats.max()), float(poly[:, 1].max())) + m_lat
    return lon_min, lon_max, lat_min, lat_max


def _add_region_cartopy_basemap(ax, *, skip_ocean: bool = False) -> None:
    """Land/ocean fill for region maps (shared by site and Qs cartopy figures)."""
    import cartopy.feature as cfeature

    if not skip_ocean:
        ax.add_feature(cfeature.OCEAN, facecolor=_REGION_MAP_OCEAN_COLOR, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=_REGION_MAP_LAND_COLOR, zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="#3d5c3d", zorder=2)


def _guess_gebco_elevation_var(ds: xr.Dataset) -> str:
    for var in ds.data_vars:
        vl = var.lower()
        if "elevation" in vl or "depth" in vl or "bathymetry" in vl or vl == "z":
            return var
    return next(iter(ds.data_vars))


def _subset_gebco_for_map_extent(
    ds: xr.Dataset,
    elev_var: str,
    lon_name: str,
    lat_name: str,
    extent: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lon/lat/elevation cropped strictly to the map extent (not full GEBCO)."""
    lon_min, lon_max, lat_min, lat_max = extent
    lon_vals = np.asarray(ds[lon_name].values, dtype=float)
    lat_vals = np.asarray(ds[lat_name].values, dtype=float)

    lon_idx = np.flatnonzero((lon_vals >= lon_min) & (lon_vals <= lon_max))
    lat_idx = np.flatnonzero((lat_vals >= lat_min) & (lat_vals <= lat_max))
    if lon_idx.size == 0 or lat_idx.size == 0:
        return lon_vals[:0], lat_vals[:0], np.empty((0, 0), dtype=float)

    # Include one neighbouring cell on each side for pcolormesh edges.
    i0 = max(int(lon_idx[0]) - 1, 0)
    i1 = min(int(lon_idx[-1]) + 2, lon_vals.size)
    j0 = max(int(lat_idx[0]) - 1, 0)
    j1 = min(int(lat_idx[-1]) + 2, lat_vals.size)

    sub = ds.isel({lon_name: slice(i0, i1), lat_name: slice(j0, j1)})
    lons = np.asarray(sub[lon_name].values, dtype=float)
    lats = np.asarray(sub[lat_name].values, dtype=float)
    elev = np.asarray(sub[elev_var].values, dtype=float)

    lon2d, lat2d = np.meshgrid(lons, lats)
    outside = (
        (lon2d < lon_min)
        | (lon2d > lon_max)
        | (lat2d < lat_min)
        | (lat2d > lat_max)
    )
    elev = np.where(outside, np.nan, elev)
    return lons, lats, elev


_DEFAULT_SHALLOW_BATHY_DEPTH_BOUNDS = (0.0, 5.0, 10.0, 15.0, 25.0, 35.0, 50.0, 120.0, 400.0)
_DEFAULT_SHALLOW_BATHY_CONTOUR_LEVELS = (-100.0, -50.0, -35.0, -25.0, -15.0, -10.0, -5.0)


def _gebco_contour_levels(levels: Sequence[float] | None) -> list[float]:
    """Return sorted GEBCO elevation levels (negative, increasing/deeper → shallower)."""
    raw = list(levels or _DEFAULT_SHALLOW_BATHY_CONTOUR_LEVELS)
    elev = sorted({-abs(float(v)) for v in raw})
    return elev


def _shallow_emphasis_bathymetry_cmap(
    depth_bounds: Sequence[float] = _DEFAULT_SHALLOW_BATHY_DEPTH_BOUNDS,
) -> tuple[mcolors.Colormap, mcolors.BoundaryNorm]:
    """Colormap with extra colour steps in the nearshore depth bands."""
    bounds = [float(b) for b in depth_bounds]
    if bounds[0] != 0.0:
        bounds = [0.0, *bounds]
    # Light-to-dark blues: most colour change between 0–50 m, fewer steps deeper.
    palette = [
        "#eef7fc",
        "#d4ebf7",
        "#b3daf0",
        "#8fc4e8",
        "#67a9db",
        "#458fcc",
        "#2d74b8",
        "#1a5494",
        "#0d335f",
    ]
    n_bins = len(bounds) - 1
    colors = palette[:n_bins]
    if len(colors) < n_bins:
        base = plt.get_cmap("Blues_r", n_bins)
        colors = [base(i / max(n_bins - 1, 1)) for i in range(n_bins)]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, ncolors=cmap.N, clip=True)
    return cmap, norm


def _overlay_gebco_bathymetry_on_map_ax(
    ax,
    bathymetry_nc: str | Path,
    extent: tuple[float, float, float, float],
    *,
    project_root: str | Path | None = None,
    cmap: str | mcolors.Colormap | None = None,
    depth_color_bounds: Sequence[float] | None = None,
    zorder: float = 0.5,
    add_contours: bool = True,
    contour_levels: Sequence[float] | None = None,
) -> None:
    """Plot GEBCO ocean depth (positive m) beneath land/coastline features."""
    import cartopy.crs as ccrs

    path = resolve_path(bathymetry_nc, project_root)
    if not path.is_file():
        print(f"WARN: bathymetry file not found: {path}")
        return

    with xr.open_dataset(path) as ds:
        elev_var = _guess_gebco_elevation_var(ds)
        lon_name = next(
            (c for c in ds.coords if "lon" in c.lower() or c.lower() == "x"),
            None,
        )
        lat_name = next(
            (c for c in ds.coords if "lat" in c.lower() or c.lower() == "y"),
            None,
        )
        if lon_name is None or lat_name is None:
            raise ValueError(f"Could not find lon/lat coordinates in {path}")

        lons, lats, elev = _subset_gebco_for_map_extent(
            ds, elev_var, lon_name, lat_name, extent
        )

    if lons.size == 0 or lats.size == 0:
        print(f"WARN: no GEBCO cells inside map extent for {path}")
        return

    depth = np.where(elev < 0.0, -elev, np.nan)
    if not np.any(np.isfinite(depth)):
        print(f"WARN: no ocean cells in bathymetry extent for {path}")
        return

    bounds = depth_color_bounds or _DEFAULT_SHALLOW_BATHY_DEPTH_BOUNDS
    if isinstance(cmap, mcolors.Colormap):
        bathy_cmap = cmap
        bathy_norm = mcolors.BoundaryNorm(
            [float(b) for b in bounds], ncolors=bathy_cmap.N, clip=True
        )
    elif cmap is not None:
        bathy_cmap = plt.get_cmap(cmap)
        bathy_norm = mcolors.BoundaryNorm(
            [float(b) for b in bounds], ncolors=256, clip=True
        )
    else:
        bathy_cmap, bathy_norm = _shallow_emphasis_bathymetry_cmap(bounds)

    ax.pcolormesh(
        lons,
        lats,
        depth,
        cmap=bathy_cmap,
        norm=bathy_norm,
        transform=ccrs.PlateCarree(),
        zorder=zorder,
        shading="auto",
    )

    if add_contours:
        elev_levels = _gebco_contour_levels(contour_levels)
        cs = ax.contour(
            lons,
            lats,
            elev,
            levels=elev_levels,
            colors="#1a3a5c",
            linewidths=0.45,
            transform=ccrs.PlateCarree(),
            zorder=zorder + 0.05,
        )
        ax.clabel(
            cs,
            inline=True,
            fontsize=7,
            fmt=lambda level: f"{int(abs(level))} m",
        )


def _overlay_buoys_on_map_ax(
    ax,
    buoys: Mapping[str, Any],
    *,
    use_cartopy: bool,
    label_buoys: bool = True,
    buoy_color: str = "#c0392b",
    buoy_marker: str = "o",
    buoy_size: float = 72.0,
    buoy_edgecolor: str = "white",
    buoy_linewidth: float = 0.9,
    buoy_zorder: int = 6,
    label_fontsize: float = 8.0,
) -> list[tuple[float, float]]:
    """Plot NDBC buoy positions from a ``buoys`` catalog on a map axes."""
    locations = normalize_buoy_locations(buoys)
    if not locations:
        return []

    coords = list(locations.items())
    lats = [lat for _bid, (lat, lon) in coords]
    lons = [lon for _bid, (lat, lon) in coords]
    scatter_kw: dict[str, Any] = dict(
        c=buoy_color,
        s=buoy_size,
        marker=buoy_marker,
        edgecolors=buoy_edgecolor,
        linewidths=buoy_linewidth,
        zorder=buoy_zorder,
    )
    if use_cartopy:
        import cartopy.crs as ccrs

        scatter_kw["transform"] = ccrs.PlateCarree()
    ax.scatter(lons, lats, **scatter_kw)

    if label_buoys:
        text_kw: dict[str, Any] = dict(fontsize=label_fontsize, fontweight="bold", color=buoy_color)
        if use_cartopy:
            import cartopy.crs as ccrs

            text_kw["transform"] = ccrs.PlateCarree()
        for bid, (lat, lon) in coords:
            ax.text(lon + 0.04, lat + 0.04, str(bid), ha="left", va="bottom", zorder=buoy_zorder + 1, **text_kw)

    return [(lat, lon) for lat, lon in zip(lats, lons)]


def _sort_sites_along_coast(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Order sites alongshore (sort by x, then y) for line overlays."""
    order = np.lexsort((y, x))
    return x[order], y[order], values[order]


def plot_region_sites_map(
    result: RegionCumulativeQsResult,
    *,
    margin_deg: float = 0.18,
    figsize: tuple[float, float] = (10, 8),
    point_size: float = 28,
) -> plt.Figure:
    """Map of hindcast sites selected inside the polygon (lon/lat)."""
    try:
        import cartopy.crs as ccrs
    except ImportError as exc:
        raise ImportError("plot_region_sites_map requires cartopy") from exc

    lon_min, lon_max, lat_min, lat_max = _region_map_extent(result, margin_deg=margin_deg)

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    _add_region_cartopy_basemap(ax)

    ax.scatter(
        result.lon,
        result.lat,
        c="crimson",
        s=point_size,
        edgecolors="none",
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    ax.set_title(
        f"Selected hindcast sites (n={result.site_index.size}), "
        f"shore-normal = {result.shoreline_orientation_deg:.0f}°"
    )
    plt.tight_layout()
    return fig


def plot_region_cumulative_qs_map(
    result: RegionCumulativeQsResult,
    *,
    margin_deg: float = 0.18,
    figsize: tuple[float, float] = (11, 8),
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    point_size: float = 36,
    connect_alongshore: bool = False,
    use_cartopy: bool = True,
    use_projected_xy: bool = False,
    background_image: str | Path | None = None,
    bathymetry_nc: str | Path | None = None,
    project_root: str | Path | None = None,
    bathymetry_cmap: str | mcolors.Colormap | None = None,
    bathymetry_depth_bounds: Sequence[float] | None = None,
    bathymetry_contour_levels: Sequence[float] | None = None,
    bathymetry_contours: bool = True,
    buoys: Mapping[str, Any] | None = None,
    label_buoys: bool = True,
    buoy_color: str = "#c0392b",
    buoy_size: float = 72.0,
) -> plt.Figure:
    """
    Map of total (cumulative) Qs at hindcast sites.

    Default ``use_cartopy=True``: lon/lat map with land/ocean fill (same style as
    ``plot_region_sites_map``), colored points only — no alongshore connector.

    Pass ``bathymetry_nc`` (e.g. ``inputs/gebco_bathymetry.nc``) to replace the flat
    ocean fill with GEBCO depth shading and isobaths. By default, colour levels
    emphasise shallow water at 5, 10, 15, 25, 35, and 50 m, with coarser bins deeper.

    Pass ``buoys`` (same catalog as ``plot_longshore_transport_for_buoys``) to overlay
    NDBC buoy locations on top of the Qs field.

    Set ``use_cartopy=False`` and ``use_projected_xy=True`` for a plain x/y axes plot
    using ``coord_x`` / ``coord_y`` from the merged NetCDF.
    """
    qs = np.asarray(result.cumulative_qs, dtype=float)
    lons = np.asarray(result.lon, dtype=float)
    lats = np.asarray(result.lat, dtype=float)

    lim = float(np.nanmax(np.abs(qs))) if np.any(np.isfinite(qs)) else 1.0
    if vmin is None:
        vmin = -lim
    if vmax is None:
        vmax = lim
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    year_lo = (result.time_start or "1980")[:4]
    year_hi = (result.time_end or "2023")[:4]
    formula_note = "deep" if getattr(result, "qs_formula", "cerc") == "deep" else "CERC"
    dir_note = getattr(result, "wave_direction", "dp").upper()
    title = f"Total Sediment Flux ({year_lo}-{year_hi}) — {formula_note}, {dir_note}"

    if use_cartopy:
        try:
            import cartopy.crs as ccrs
        except ImportError as exc:
            raise ImportError("plot_region_cumulative_qs_map requires cartopy") from exc

        lon_min, lon_max, lat_min, lat_max = _region_map_extent(
            result, margin_deg=margin_deg
        )
        if buoys:
            buoy_pts = normalize_buoy_locations(buoys)
            blons = [lon for lat, lon in buoy_pts.values()]
            blats = [lat for lat, lon in buoy_pts.values()]
            if blons:
                lon_min = min(lon_min, min(blons) - margin_deg * 0.25)
                lon_max = max(lon_max, max(blons) + margin_deg * 0.25)
                lat_min = min(lat_min, min(blats) - margin_deg * 0.25)
                lat_max = max(lat_max, max(blats) + margin_deg * 0.25)

        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        map_extent = [lon_min, lon_max, lat_min, lat_max]
        ax.set_extent(map_extent, crs=ccrs.PlateCarree())
        if bathymetry_nc is not None:
            _overlay_gebco_bathymetry_on_map_ax(
                ax,
                bathymetry_nc,
                (lon_min, lon_max, lat_min, lat_max),
                project_root=project_root,
                cmap=bathymetry_cmap,
                depth_color_bounds=bathymetry_depth_bounds,
                add_contours=bathymetry_contours,
                contour_levels=bathymetry_contour_levels,
            )
            _add_region_cartopy_basemap(ax, skip_ocean=True)
        else:
            _add_region_cartopy_basemap(ax)

        sc = ax.scatter(
            lons,
            lats,
            c=qs,
            s=point_size,
            cmap=cmap,
            norm=norm,
            edgecolors="none",
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        if buoys:
            _overlay_buoys_on_map_ax(
                ax,
                buoys,
                use_cartopy=True,
                label_buoys=label_buoys,
                buoy_color=buoy_color,
                buoy_size=buoy_size,
            )
        cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label(r"Total $Q_s$")
        ax.set_title(title)
        plt.tight_layout()
        # Keep the same lon/lat window as without bathymetry (cartopy/tight_layout
        # can otherwise expand the view to fit the full GEBCO grid aspect).
        ax.set_extent(map_extent, crs=ccrs.PlateCarree())
        return fig

    if use_projected_xy and result.coord_x is not None and result.coord_y is not None:
        x = np.asarray(result.coord_x, dtype=float)
        y = np.asarray(result.coord_y, dtype=float)
        xlabel, ylabel = "x", "y"
    else:
        x, y = lons, lats
        xlabel, ylabel = "lon", "lat"

    x_s, y_s, qs_s = _sort_sites_along_coast(x, y, qs)

    fig, ax = plt.subplots(figsize=figsize)
    if background_image is not None:
        bg_path = Path(background_image)
        if bg_path.is_file():
            img = plt.imread(str(bg_path))
            ax.imshow(img, extent=(x.min(), x.max(), y.min(), y.max()), aspect="auto", zorder=0)
        else:
            print(f"WARN: background_image not found: {bg_path}")

    if connect_alongshore:
        ax.plot(x_s, y_s, color="0.35", lw=0.8, zorder=1, alpha=0.7)
    sc = ax.scatter(x_s, y_s, c=qs_s, s=point_size, cmap=cmap, norm=norm, edgecolors="none", zorder=2)
    if buoys and not use_cartopy:
        _overlay_buoys_on_map_ax(
            ax,
            buoys,
            use_cartopy=False,
            label_buoys=label_buoys,
            buoy_color=buoy_color,
            buoy_size=buoy_size,
        )
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(r"Total $Q_s$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title + (" over Landsat Mosaic." if background_image else ""))
    plt.tight_layout()
    return fig


# --- Wave roses (buoy or hindcast Hs + Dp) ------------------------------------


def load_timeseries_at_site_index(
    nc_path: str | Path,
    site_index: int,
    *,
    variable: str | None = None,
    aggregation: str = "daily",
    time_start=None,
    time_end=None,
    project_root: str | Path | None = None,
) -> tuple[xr.DataArray, dict]:
    """Load aggregated time series at integer ``site`` index in a merged NetCDF."""
    nc_path = resolve_path(nc_path, project_root)
    if not nc_path.is_file():
        raise FileNotFoundError(nc_path)

    variable = variable or _variable_from_nc_path(Path(nc_path))
    site_index = int(site_index)

    with xr.open_dataset(nc_path) as ds:
        lat_name, lon_name = _guess_lat_lon(ds)
        site_dim = _guess_site_dim(ds, lat_name, lon_name)
        n_site = int(ds.sizes[site_dim])
        if site_index < 0 or site_index >= n_site:
            raise ValueError(f"site_index {site_index} out of range [0, {n_site - 1}]")
        nc_var = _resolve_nc_var(ds, variable)
        da = ds[nc_var].isel({site_dim: site_index})
        lat = float(np.asarray(ds[lat_name].isel({site_dim: site_index}).values))
        lon = float(np.asarray(ds[lon_name].isel({site_dim: site_index}).values))
        da = apply_aggregation(da, variable, aggregation)
        if time_start is not None or time_end is not None:
            da = da.sel(time=slice(time_start, time_end))

    info = {
        "site_index": site_index,
        "lat": lat,
        "lon": lon,
        "distance_km": 0.0,
        "nc_var": nc_var,
        "path": str(nc_path),
        "aggregation": str(aggregation).lower(),
        "n_points": int(da.sizes.get("time", da.size)),
    }
    return da, info


def load_historical_hs_dp(
    *,
    coordinate: tuple[float, float] | None = None,
    site_index: int | None = None,
    aggregation: str = "hourly",
    time_start=None,
    time_end=None,
    historical_folder: str | Path | None = None,
    historical_dataset: HistoricalDataset = "merged_grids",
    wave_direction: str = "dp",
    project_root: str | Path | None = None,
) -> tuple[tuple[xr.DataArray, xr.DataArray] | None, dict]:
    """
    Load hindcast bulk ``hs`` and direction (``dp`` or ``dm``) for a wave rose.

    Provide **either** ``coordinate`` ``(lat, lon)`` (nearest grid site) **or**
    ``site_index`` (integer index into the ``site`` dimension of ``hs_merged_all.nc``).
    """
    if (coordinate is None) == (site_index is None):
        raise ValueError("Provide exactly one of coordinate=(lat, lon) or site_index.")

    wave_direction = _normalize_wave_direction(wave_direction)
    hist_folder = resolve_historical_folder(
        historical_folder=historical_folder,
        historical_dataset=historical_dataset,
        project_root=project_root,
    )
    hs_nc = _resolve_variable_nc_path(hist_folder, "hs", project_root)
    dp_nc = _resolve_variable_nc_path(hist_folder, wave_direction, project_root)

    if site_index is not None:
        hs, info_hs = load_timeseries_at_site_index(
            hs_nc,
            site_index,
            variable="hs",
            aggregation=aggregation,
            time_start=time_start,
            time_end=time_end,
            project_root=project_root,
        )
        dp, _ = load_timeseries_at_site_index(
            dp_nc,
            site_index,
            variable=wave_direction,
            aggregation=aggregation,
            time_start=time_start,
            time_end=time_end,
            project_root=project_root,
        )
        info = dict(info_hs)
        info["wave_direction"] = wave_direction
    else:
        lat, lon = coordinate  # type: ignore[misc]
        hs, info_hs = load_timeseries_at_coordinate(
            hs_nc,
            (lat, lon),
            variable="hs",
            aggregation=aggregation,
            time_start=time_start,
            time_end=time_end,
            project_root=project_root,
        )
        dp, _ = load_timeseries_at_coordinate(
            dp_nc,
            (lat, lon),
            variable=wave_direction,
            aggregation=aggregation,
            time_start=time_start,
            time_end=time_end,
            project_root=project_root,
        )
        info = dict(info_hs)
        info["wave_direction"] = wave_direction

    return (hs, dp), info


def _load_partition_hs_dp_one(
    partitions_folder: str | Path,
    pid: int,
    *,
    coordinate: tuple[float, float] | None,
    site_index: int | None,
    aggregation: str,
    time_start,
    time_end,
    project_root: str | Path | None,
) -> tuple[xr.DataArray, xr.DataArray, dict] | None:
    """Load ``phs{pid}`` and ``dp{pid}`` at one site; return None if files or overlap missing."""
    part_folder = Path(resolve_path(partitions_folder, project_root))
    hs_name = f"phs{pid}"
    dp_name = f"dm{pid}"
    try:
        hs_path = _resolve_variable_nc_path(part_folder, hs_name, project_root)
        dp_path = _resolve_variable_nc_path(part_folder, dp_name, project_root)
    except FileNotFoundError as exc:
        print(f"SKIP partition {pid}: {exc}")
        return None

    load_kw = dict(
        aggregation=aggregation,
        time_start=time_start,
        time_end=time_end,
        project_root=project_root,
    )
    if site_index is not None:
        hs, info = load_timeseries_at_site_index(
            hs_path, site_index, variable=hs_name, **load_kw
        )
        dp, _ = load_timeseries_at_site_index(
            dp_path, site_index, variable=dp_name, **load_kw
        )
    else:
        hs, info = load_timeseries_at_coordinate(
            hs_path, coordinate, variable=hs_name, **load_kw  # type: ignore[arg-type]
        )
        dp, _ = load_timeseries_at_coordinate(
            dp_path, coordinate, variable=dp_name, **load_kw  # type: ignore[arg-type]
        )
    hs_a, dp_a = xr.align(hs, dp, join="inner")
    if hs_a.sizes.get("time", 0) == 0:
        print(f"SKIP partition {pid}: no overlapping phs/dp times")
        return None
    info = dict(info)
    info["partition_id"] = int(pid)
    info["hs_var"] = hs_name
    info["dp_var"] = dp_name
    return hs_a, dp_a, info


def load_historical_partition_hs_dp(
    *,
    coordinate: tuple[float, float] | None = None,
    site_index: int | None = None,
    partition_ids: Sequence[int] = (0, 1),
    aggregation: str = "hourly",
    time_start=None,
    time_end=None,
    partitions_folder: str | Path | None = None,
    partitions_dataset: HistoricalDataset = "merged_grids",
    project_root: str | Path | None = None,
) -> tuple[dict[int, tuple[xr.DataArray, xr.DataArray]], dict]:
    """
    Load hindcast partition ``phs*`` / ``dp*`` for wave roses.

    Returns ``({pid: (hs, dp), ...}, site_info)``. Missing partitions are omitted.
    """
    if (coordinate is None) == (site_index is None):
        raise ValueError("Provide exactly one of coordinate=(lat, lon) or site_index.")

    part_folder = resolve_partitions_folder(
        partitions_folder=partitions_folder,
        partitions_dataset=partitions_dataset,
        project_root=project_root,
    )
    out: dict[int, tuple[xr.DataArray, xr.DataArray]] = {}
    info: dict[str, Any] = {}
    for pid in [int(p) for p in partition_ids]:
        loaded = _load_partition_hs_dp_one(
            part_folder,
            pid,
            coordinate=coordinate,
            site_index=site_index,
            aggregation=aggregation,
            time_start=time_start,
            time_end=time_end,
            project_root=project_root,
        )
        if loaded is None:
            continue
        hs_a, dp_a, pinfo = loaded
        out[pid] = (hs_a, dp_a)
        if not info:
            info = dict(pinfo)

    info["partition_ids_loaded"] = sorted(out.keys())
    info["partition_ids_requested"] = [int(p) for p in partition_ids]
    return out, info


def _parse_radial_range(
    radial_range: float | str | Sequence[float] | None,
) -> tuple[float, np.ndarray | None] | None:
    """
    Parse radial grid spec for wave roses (% on polar radius).

    Returns ``(r_max, yticks)`` or ``None`` for auto scaling.

    Accepted forms:

    - ``None`` — auto from data
    - ``25`` — fixed maximum, matplotlib chooses tick spacing
    - ``"0:5:25"`` — string ``start:step:stop`` (grid at 0, 5, …, 25)
    - ``(0, 25, 5)`` — tuple ``(start, stop, step)``
    - ``(0, 25)`` — tuple ``(start, stop)``, fixed max only
    - ``[0, 5, 10, 15, 20, 25]`` — explicit circle positions
    """
    if radial_range is None:
        return None

    if isinstance(radial_range, (int, float, np.floating)):
        return float(radial_range), None

    if isinstance(radial_range, str):
        parts = [float(x.strip()) for x in radial_range.split(":")]
        if len(parts) == 3:
            start, step, stop = parts
            ticks = np.arange(start, stop + step * 0.5, step)
            return float(stop), ticks
        if len(parts) == 2:
            _start, stop = parts
            return float(stop), None
        raise ValueError(
            "radial_range string must be 'start:step:stop' or 'start:stop', "
            f"got {radial_range!r}"
        )

    seq = [float(x) for x in radial_range]
    if len(seq) == 1:
        return seq[0], None
    if len(seq) == 2:
        return seq[1], None
    if len(seq) == 3:
        start, stop, step = seq
        ticks = np.arange(start, stop + step * 0.5, step)
        return float(stop), ticks

    ticks = np.asarray(seq, dtype=float)
    return float(ticks.max()), ticks


def _default_radial_ticks(r_max: float) -> np.ndarray:
    """Even % grid circles up to ``r_max`` (5% step when max ≤ 50, else 10%)."""
    step = 5.0 if r_max <= 50.0 else 10.0
    return np.arange(step, r_max + step * 0.5, step)


def _resolved_wave_rose_radial(
    radial_range: float | str | Sequence[float] | None,
    *,
    auto_r_max: float,
) -> tuple[float, np.ndarray | None, bool]:
    """
    Return ``(r_max, display_ticks, is_fixed)`` for a wave-rose polar axis.

    ``is_fixed`` is True when the caller supplied ``radial_range`` (comparison mode).
    """
    parsed = _parse_radial_range(radial_range)
    if parsed is None:
        r_max = float(auto_r_max)
        return r_max, None, False

    r_max, yticks = parsed
    if yticks is None:
        yticks = _default_radial_ticks(r_max)
    display_ticks = np.asarray(yticks, dtype=float)
    display_ticks = display_ticks[display_ticks > 0]
    return float(r_max), display_ticks, True


def _apply_radial_range_on_polar(
    ax: plt.Axes,
    radial_range: float | str | Sequence[float] | None,
    *,
    auto_r_max: float,
) -> float:
    """Set polar radial limits/ticks; returns the radial maximum in use."""
    r_max, display_ticks, is_fixed = _resolved_wave_rose_radial(
        radial_range, auto_r_max=auto_r_max
    )
    ax.set_autoscale_on(False)
    ax.set_ylim(0, r_max, auto=False)
    if is_fixed and display_ticks is not None and display_ticks.size:
        ax.yaxis.set_major_locator(FixedLocator(display_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}%"))
    ax._wave_rose_r_max = r_max  # type: ignore[attr-defined]
    ax._wave_rose_yticks = display_ticks  # type: ignore[attr-defined]
    ax._wave_rose_radial_fixed = is_fixed  # type: ignore[attr-defined]
    return r_max


def _reapply_wave_rose_radial_after_layout(fig: plt.Figure) -> None:
    """Re-lock radial limits/ticks after ``tight_layout`` (polar autoscale workaround)."""
    pct_fmt = FuncFormatter(lambda y, _: f"{y:g}%")
    for ax in fig.get_axes():
        r_max = getattr(ax, "_wave_rose_r_max", None)
        if r_max is None:
            continue
        ax.set_autoscale_on(False)
        ax.set_ylim(0, float(r_max), auto=False)
        display_ticks = getattr(ax, "_wave_rose_yticks", None)
        if getattr(ax, "_wave_rose_radial_fixed", False) and display_ticks is not None:
            ticks = np.asarray(display_ticks, dtype=float)
            if ticks.size:
                ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(pct_fmt)


def _resolve_rose_radial_ranges(
    *,
    radial_range: float | str | Sequence[float] | None,
    energy_radial_range: float | str | Sequence[float] | None,
    occurrence_radial_range: float | str | Sequence[float] | None,
) -> tuple[
    float | str | Sequence[float] | None,
    float | str | Sequence[float] | None,
]:
    """
    Map radial-range args to (occurrence panel, energy panel).

    ``radial_range`` fixes the **% energy** rose (right) for cross-plot comparison.
    ``energy_radial_range`` overrides that; ``occurrence_radial_range`` is optional
    for the left panel (auto-scales when omitted).
    """
    energy = energy_radial_range if energy_radial_range is not None else radial_range
    return occurrence_radial_range, energy


def _add_shore_orientation_arrows_on_polar(
    ax: plt.Axes,
    shoreline_orientation_deg: float,
    *,
    longshore_offset_deg: float = 90.0,
    arrow_length_frac: float = 0.88,
    shore_normal_color: str = "#1a1a1a",
    longshore_color: str = "#c0392b",
    lw: float = 2.2,
    zorder: int = 10,
) -> None:
    """
    Overlay shore-normal and longshore on a wave rose.

    Shore-normal (θ): arrow from the centre. Longshore (θ + offset): diameter line
    across the rose in that direction. Angles are ° from North, clockwise (CERC Qs).
    """
    r_max = float(ax.get_ylim()[1])
    r_arrow = max(r_max * float(arrow_length_frac), 0.5)
    sn_deg = float(shoreline_orientation_deg) % 360.0
    along_deg = (sn_deg + float(longshore_offset_deg)) % 360.0

    sn_theta = np.deg2rad(sn_deg)
    ax.annotate(
        "",
        xy=(sn_theta, r_arrow),
        xytext=(0.0, 0.0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=shore_normal_color,
            lw=lw,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=zorder,
    )
    ax.text(
        sn_theta,
        r_arrow * 1.07,
        r"shore-normal ($\theta$)",
        ha="center",
        va="center",
        fontsize=7.5,
        color=shore_normal_color,
        fontweight="bold",
        zorder=zorder + 1,
    )

    along_theta = np.deg2rad(along_deg)
    ax.plot(
        [along_theta, along_theta + np.pi],
        [r_max, r_max],
        color=longshore_color,
        lw=lw,
        linestyle="-",
        solid_capstyle="round",
        zorder=zorder,
    )
    ax.text(
        along_theta,
        r_max * 1.07,
        rf"longshore ($\theta$+{longshore_offset_deg:g}°)",
        ha="center",
        va="center",
        fontsize=7.5,
        color=longshore_color,
        fontweight="bold",
        zorder=zorder + 1,
    )


def _stacked_hs_dp_wave_rose_on_ax(
    ax: plt.Axes,
    h_vals: np.ndarray,
    d_vals: np.ndarray,
    *,
    mode: Literal["frequency", "energy"],
    title: str,
    n_dir_bins: int = 16,
    hs_range: tuple[float, float] = (0.0, 6.0),
    hs_bin_m: float = 1.0,
    energy_power: float = 2.0,
    shoreline_orientation_deg: float | None = None,
    show_orientation_arrows: bool | None = None,
    longshore_offset_deg: float = 90.0,
    arrow_length_frac: float = 0.88,
    radial_range: float | str | Sequence[float] | None = None,
) -> tuple[float, np.ndarray, list[str], list[str]]:
    """Stacked polar wave rose on ``ax`` (frequency or energy mode)."""
    dir_edges = np.linspace(0.0, 360.0, int(n_dir_bins) + 1)
    hs_edges = np.arange(hs_range[0], hs_range[1] + hs_bin_m, hs_bin_m)
    if len(hs_edges) < 2:
        hs_edges = np.array([hs_range[0], hs_range[1] + hs_bin_m])

    weights = np.power(h_vals, float(energy_power)) if mode == "energy" else None
    hist, _, _ = np.histogram2d(
        d_vals, h_vals, bins=[dir_edges, hs_edges], weights=weights
    )
    n_hs = hist.shape[1]
    hs_labels = _hs_bin_labels(hs_edges)
    hs_colors = _hs_bin_contrast_colors(n_hs)

    total = float(hist.sum())
    dir_pct = (
        hist.sum(axis=1) / total * 100.0 if total > 0 else np.zeros(hist.shape[0])
    )

    theta = np.deg2rad(0.5 * (dir_edges[:-1] + dir_edges[1:]))
    width = np.deg2rad(360.0 / int(n_dir_bins))

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    bottom = np.zeros(hist.shape[0], dtype=float)
    for j in range(n_hs):
        layer = hist[:, j] / total * 100.0 if total > 0 else np.zeros(hist.shape[0])
        ax.bar(
            theta,
            layer,
            width=width,
            bottom=bottom,
            color=hs_colors[j],
            edgecolor="white",
            linewidth=0.45,
            align="center",
        )
        bottom += layer

    auto_r_max = max(float(dir_pct.max()) * 1.12, 1.0) if total > 0 else 1.0
    r_max = _apply_radial_range_on_polar(
        ax, radial_range, auto_r_max=auto_r_max
    )
    draw_arrows = (
        show_orientation_arrows
        if show_orientation_arrows is not None
        else shoreline_orientation_deg is not None
    )
    if draw_arrows and shoreline_orientation_deg is not None:
        _add_shore_orientation_arrows_on_polar(
            ax,
            shoreline_orientation_deg,
            longshore_offset_deg=longshore_offset_deg,
            arrow_length_frac=arrow_length_frac,
        )
    ax.set_title(title, pad=16)
    return total, hs_edges, hs_labels, hs_colors


def _plot_hs_dp_dual_wave_rose(
    h_vals: np.ndarray,
    d_vals: np.ndarray,
    *,
    site_label: str,
    period_note: str,
    aggregation: str,
    footnote: str,
    n_dir_bins: int = 16,
    hs_range: tuple[float, float] = (0.0, 6.0),
    hs_bin_m: float = 1.0,
    energy_power: float = 2.0,
    figsize: tuple[float, float] = (14.0, 7.0),
    shoreline_orientation_deg: float | None = None,
    show_orientation_arrows: bool | None = None,
    longshore_offset_deg: float = 90.0,
    arrow_length_frac: float = 0.88,
    radial_range: float | str | Sequence[float] | None = None,
    energy_radial_range: float | str | Sequence[float] | None = None,
    occurrence_radial_range: float | str | Sequence[float] | None = None,
) -> tuple[int, float]:
    """Side-by-side % occurrence rose (left) and % energy rose (right), colored by Hs bin."""
    occ_radial, eng_radial = _resolve_rose_radial_ranges(
        radial_range=radial_range,
        energy_radial_range=energy_radial_range,
        occurrence_radial_range=occurrence_radial_range,
    )
    rose_kw = dict(
        n_dir_bins=n_dir_bins,
        hs_range=hs_range,
        hs_bin_m=hs_bin_m,
        energy_power=energy_power,
        shoreline_orientation_deg=shoreline_orientation_deg,
        show_orientation_arrows=show_orientation_arrows,
        longshore_offset_deg=longshore_offset_deg,
        arrow_length_frac=arrow_length_frac,
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        subplot_kw={"projection": "polar"},
    )

    _stacked_hs_dp_wave_rose_on_ax(
        axes[0],
        h_vals,
        d_vals,
        mode="frequency",
        title=f"Wave rose (% occurrence) — {site_label} ({aggregation})\n{period_note}",
        radial_range=occ_radial,
        **rose_kw,
    )
    e_total, hs_edges, hs_labels, hs_colors = _stacked_hs_dp_wave_rose_on_ax(
        axes[1],
        h_vals,
        d_vals,
        mode="energy",
        title=(
            f"Wave rose (% energy) — {site_label} ({aggregation})\n"
            f"{period_note} | % of total E, E ∝ Hs^{energy_power:g}"
        ),
        radial_range=eng_radial,
        **rose_kw,
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=hs_colors[j], edgecolor="white", linewidth=0.6)
        for j in range(len(hs_labels))
    ]
    axes[1].legend(
        handles,
        hs_labels,
        title="Hs (m)",
        loc="upper left",
        bbox_to_anchor=(1.14, 1.02),
        fontsize=8,
        title_fontsize=9,
    )
    fig.text(0.5, 0.02, footnote, ha="center", fontsize=8, color="0.35")
    fig.tight_layout()
    _reapply_wave_rose_radial_after_layout(fig)
    plt.show()
    return int(h_vals.size), float(e_total)


def _plot_hs_dp_partition_grid_wave_rose(
    partition_arrays: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    partition_ids: Sequence[int],
    site_label: str,
    period_note: str,
    aggregation: str,
    footnote: str,
    n_dir_bins: int = 16,
    hs_range: tuple[float, float] = (0.0, 6.0),
    hs_bin_m: float = 1.0,
    energy_power: float = 2.0,
    figsize: tuple[float, float] | None = None,
    shoreline_orientation_deg: float | None = None,
    show_orientation_arrows: bool | None = None,
    longshore_offset_deg: float = 90.0,
    arrow_length_frac: float = 0.88,
    radial_range: float | str | Sequence[float] | None = None,
    energy_radial_range: float | str | Sequence[float] | None = None,
    occurrence_radial_range: float | str | Sequence[float] | None = None,
) -> dict[int, dict[str, float | int]]:
    """
    2×N grid: rows = partitions, columns = % occurrence rose (left) and % energy rose (right).
    """
    occ_radial, eng_radial = _resolve_rose_radial_ranges(
        radial_range=radial_range,
        energy_radial_range=energy_radial_range,
        occurrence_radial_range=occurrence_radial_range,
    )
    pids = [int(p) for p in partition_ids if int(p) in partition_arrays]
    if not pids:
        return {}

    n_rows = len(pids)
    if figsize is None:
        figsize = (14.0, max(6.5 * n_rows, 7.0))

    rose_kw = dict(
        n_dir_bins=n_dir_bins,
        hs_range=hs_range,
        hs_bin_m=hs_bin_m,
        energy_power=energy_power,
        shoreline_orientation_deg=shoreline_orientation_deg,
        show_orientation_arrows=show_orientation_arrows,
        longshore_offset_deg=longshore_offset_deg,
        arrow_length_frac=arrow_length_frac,
    )

    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=figsize,
        subplot_kw={"projection": "polar"},
        squeeze=False,
    )

    stats: dict[int, dict[str, float | int]] = {}
    hs_edges: np.ndarray | None = None
    hs_labels: list[str] | None = None
    hs_colors: list[str] | None = None

    for row, pid in enumerate(pids):
        h_vals, d_vals = partition_arrays[pid]
        freq_title = (
            f"Wave rose (% occurrence) — partition {pid}\n"
            f"{site_label} ({aggregation}) | {period_note}"
        )
        energy_title = (
            f"Wave rose (% energy) — partition {pid}\n"
            f"{site_label} ({aggregation}) | {period_note} | % of total E, E ∝ Hs^{energy_power:g}"
        )

        _stacked_hs_dp_wave_rose_on_ax(
            axes[row, 0],
            h_vals,
            d_vals,
            mode="frequency",
            title=freq_title,
            radial_range=occ_radial,
            **rose_kw,
        )
        e_total, hs_edges, hs_labels, hs_colors = _stacked_hs_dp_wave_rose_on_ax(
            axes[row, 1],
            h_vals,
            d_vals,
            mode="energy",
            title=energy_title,
            radial_range=eng_radial,
            **rose_kw,
        )
        stats[pid] = {"n_samples": int(h_vals.size), "energy_total_hs_power": float(e_total)}

    if hs_labels and hs_colors:
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=hs_colors[j], edgecolor="white", linewidth=0.6)
            for j in range(len(hs_labels))
        ]
        axes[-1, 1].legend(
            handles,
            hs_labels,
            title="Hs (m)",
            loc="upper left",
            bbox_to_anchor=(1.12, 1.02),
            fontsize=8,
            title_fontsize=9,
        )

    fig.text(0.5, 0.02, footnote, ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _reapply_wave_rose_radial_after_layout(fig)
    plt.show()
    return stats


def plot_buoy_wave_rose(
    buoy_id: str,
    *,
    time_start: str = "1993-01-01",
    time_end: str = "2023-12-31",
    aggregation: str = "hourly",
    n_dir_bins: int = 16,
    hs_range: tuple[float, float] = (0.0, 6.0),
    hs_bin_m: float = 1.0,
    energy_power: float = 2.0,
    buoy_data_dir: str | Path = DEFAULT_BUOY_DATA_DIR,
    figsize: tuple[float, float] = (14.0, 7.0),
    shoreline_orientation_deg: float | None = None,
    show_orientation_arrows: bool | None = None,
    longshore_offset_deg: float = 90.0,
    arrow_length_frac: float = 0.88,
    radial_range: float | str | Sequence[float] | None = None,
    energy_radial_range: float | str | Sequence[float] | None = None,
    occurrence_radial_range: float | str | Sequence[float] | None = None,
) -> dict[str, Any]:
    """
    Dual wave roses from buoy ``Hs_Buoy`` and ``Dir_Buoy`` (NDBC MWD).

    Pass ``shoreline_orientation_deg`` (shore-normal θ, ° from N) to overlay arrows on
    both roses: shore-normal from centre, and longshore at θ+90° (default).

    ``radial_range`` fixes the **% energy** rose (right) radial grid for cross-plot
    comparison, e.g. ``"0:5:25"`` (circles at 5%, 10%, …, 25%). Use the same value on
    buoy, hindcast, and partition plots. ``energy_radial_range`` overrides this;
    ``occurrence_radial_range`` optionally fixes the left panel (auto when omitted).
    """
    pair, info = load_buoy_hs_dp(
        buoy_id,
        buoy_data_dir=buoy_data_dir,
        aggregation=aggregation,
        time_start=time_start,
        time_end=time_end,
    )
    if pair is None:
        return {"buoy_id": str(buoy_id), "n_samples": 0}

    h_vals, d_vals = _aligned_hs_dp_numpy(*pair)
    if h_vals.size == 0:
        print(f"SKIP buoy {buoy_id}: no valid Hs/Dir for wave rose")
        return {"buoy_id": str(buoy_id), "n_samples": 0}

    period_note = (
        f"{pd.Timestamp(time_start).date()} – {pd.Timestamp(time_end).date()}"
    )
    sector_deg = 360.0 / int(n_dir_bins)
    n_samples, e_total = _plot_hs_dp_dual_wave_rose(
        h_vals,
        d_vals,
        site_label=f"buoy {buoy_id}",
        period_note=period_note,
        aggregation=str(aggregation),
        footnote=(
            f"Dir_Buoy (MWD) | {int(h_vals.size):,} samples | "
            f"left: % occurrence per {sector_deg:g}° sector; "
            f"right: % of total ΣHs^{energy_power:g} energy per sector; "
            f"stack = Hs bin (colored) | radial circles = %"
        ),
        n_dir_bins=n_dir_bins,
        hs_range=hs_range,
        hs_bin_m=hs_bin_m,
        energy_power=energy_power,
        figsize=figsize,
        shoreline_orientation_deg=shoreline_orientation_deg,
        show_orientation_arrows=show_orientation_arrows,
        longshore_offset_deg=longshore_offset_deg,
        arrow_length_frac=arrow_length_frac,
        radial_range=radial_range,
        energy_radial_range=energy_radial_range,
        occurrence_radial_range=occurrence_radial_range,
    )

    return {
        "buoy_id": str(buoy_id),
        "n_samples": n_samples,
        "energy_total_hs_power": e_total,
        "energy_power": float(energy_power),
        "aggregation": str(aggregation),
        "time_start": str(time_start),
        "time_end": str(time_end),
        "shoreline_orientation_deg": shoreline_orientation_deg,
        **info,
    }


def _plot_historical_wave_rose_one(
    wave_direction: str,
    *,
    coordinate: tuple[float, float] | None = None,
    site_index: int | None = None,
    time_start: str = "1993-01-01",
    time_end: str = "2023-12-31",
    aggregation: str = "hourly",
    n_dir_bins: int = 16,
    hs_range: tuple[float, float] = (0.0, 6.0),
    hs_bin_m: float = 1.0,
    energy_power: float = 2.0,
    historical_folder: str | Path | None = None,
    historical_dataset: HistoricalDataset = "merged_grids",
    project_root: str | Path | None = None,
    figsize: tuple[float, float] = (14.0, 7.0),
    shoreline_orientation_deg: float | None = None,
    show_orientation_arrows: bool | None = None,
    longshore_offset_deg: float = 90.0,
    arrow_length_frac: float = 0.88,
    radial_range: float | str | Sequence[float] | None = None,
    energy_radial_range: float | str | Sequence[float] | None = None,
    occurrence_radial_range: float | str | Sequence[float] | None = None,
    hindcast_source_label: str = "BinWaves",
) -> dict[str, Any]:
    wave_direction = _normalize_wave_direction(wave_direction)
    pair, info = load_historical_hs_dp(
        coordinate=coordinate,
        site_index=site_index,
        aggregation=aggregation,
        time_start=time_start,
        time_end=time_end,
        historical_folder=historical_folder,
        historical_dataset=historical_dataset,
        wave_direction=wave_direction,
        project_root=project_root,
    )
    if pair is None:
        return {"n_samples": 0, "wave_direction": wave_direction}

    h_vals, d_vals = _aligned_hs_dp_numpy(*pair)
    if h_vals.size == 0:
        label = (
            f"site {info.get('site_index')}"
            if site_index is not None
            else f"({info.get('lat', 0):.3f}, {info.get('lon', 0):.3f})"
        )
        print(
            f"SKIP hindcast {label}: no valid Hs/"
            f"{_wave_direction_display_name(wave_direction)} direction for wave rose"
        )
        return {"n_samples": 0, "wave_direction": wave_direction, **info}

    dir_label = _wave_direction_display_name(wave_direction)
    period_note = (
        f"{pd.Timestamp(time_start).date()} – {pd.Timestamp(time_end).date()}"
    )
    if site_index is not None:
        site_label = (
            f"{hindcast_source_label} site {info['site_index']} "
            f"({info['lat']:.3f}, {info['lon']:.3f}) | {dir_label}"
        )
    else:
        dist = info.get("distance_km", 0.0)
        site_label = (
            f"{hindcast_source_label} site {info['site_index']} "
            f"({info['lat']:.3f}, {info['lon']:.3f}), {dist:.2f} km from target | {dir_label}"
        )

    sector_deg = 360.0 / int(n_dir_bins)
    n_samples, e_total = _plot_hs_dp_dual_wave_rose(
        h_vals,
        d_vals,
        site_label=site_label,
        period_note=period_note,
        aggregation=str(aggregation),
        footnote=(
            f"{hindcast_source_label} {dir_label} direction | {int(h_vals.size):,} samples | "
            f"left: % occurrence per {sector_deg:g}° sector; "
            f"right: % of total ΣHs^{energy_power:g} energy per sector; "
            f"stack = Hs bin (colored) | radial circles = %"
        ),
        n_dir_bins=n_dir_bins,
        hs_range=hs_range,
        hs_bin_m=hs_bin_m,
        energy_power=energy_power,
        figsize=figsize,
        shoreline_orientation_deg=shoreline_orientation_deg,
        show_orientation_arrows=show_orientation_arrows,
        longshore_offset_deg=longshore_offset_deg,
        arrow_length_frac=arrow_length_frac,
        radial_range=radial_range,
        energy_radial_range=energy_radial_range,
        occurrence_radial_range=occurrence_radial_range,
    )

    return {
        "n_samples": n_samples,
        "energy_total_hs_power": e_total,
        "energy_power": float(energy_power),
        "aggregation": str(aggregation),
        "time_start": str(time_start),
        "time_end": str(time_end),
        "historical_dataset": str(historical_dataset),
        "shoreline_orientation_deg": shoreline_orientation_deg,
        "wave_direction": wave_direction,
        "hindcast_source_label": hindcast_source_label,
        **info,
    }


def plot_historical_wave_rose(
    *,
    coordinate: tuple[float, float] | None = None,
    site_index: int | None = None,
    time_start: str = "1993-01-01",
    time_end: str = "2023-12-31",
    aggregation: str = "hourly",
    n_dir_bins: int = 16,
    hs_range: tuple[float, float] = (0.0, 6.0),
    hs_bin_m: float = 1.0,
    energy_power: float = 2.0,
    hindcast_source: str = "binwaves_kma",
    historical_folder: str | Path | None = None,
    historical_dataset: HistoricalDataset = "merged_grids",
    wave_direction: str = "peak",
    project_root: str | Path | None = None,
    figsize: tuple[float, float] = (14.0, 7.0),
    shoreline_orientation_deg: float | None = None,
    show_orientation_arrows: bool | None = None,
    longshore_offset_deg: float = 90.0,
    arrow_length_frac: float = 0.88,
    radial_range: float | str | Sequence[float] | None = None,
    energy_radial_range: float | str | Sequence[float] | None = None,
    occurrence_radial_range: float | str | Sequence[float] | None = None,
) -> dict[str, Any]:
    """
    Dual wave roses from merged hindcast bulk ``hs`` and direction.

    Use ``coordinate=(lat, lon)`` (nearest site) or ``site_index`` (grid site id).

    ``hindcast_source``: ``\"binwaves\"`` (original ``merged_grids`` bulk) or
    ``\"binwaves_kma\"`` (pre-built ``merged_grids_binwaves_kma``). ``historical_folder``
    overrides the default folder for either source.

    ``wave_direction``: ``\"peak\"`` (``dp_merged_all.nc``) or ``\"mean\"``
    (energy-weighted mean direction, ``dm_merged_all.nc``). Aliases ``\"dp\"`` / ``\"dm\"``
    are also accepted.

    ``shoreline_orientation_deg``: shore-normal θ (° from N, clockwise). When set,
    arrows are drawn on both roses unless ``show_orientation_arrows=False``.
    Longshore arrow is at θ + ``longshore_offset_deg`` (default 90°).

    ``radial_range`` fixes the % energy rose radial grid (see ``plot_buoy_wave_rose``).
    """
    hist_folder, source_label = _resolve_hindcast_rose_folder(
        hindcast_source=hindcast_source,
        historical_folder=historical_folder,
        project_root=project_root,
    )
    return _plot_historical_wave_rose_one(
        wave_direction,
        coordinate=coordinate,
        site_index=site_index,
        time_start=time_start,
        time_end=time_end,
        aggregation=aggregation,
        n_dir_bins=n_dir_bins,
        hs_range=hs_range,
        hs_bin_m=hs_bin_m,
        energy_power=energy_power,
        historical_folder=hist_folder,
        historical_dataset=historical_dataset,
        project_root=project_root,
        figsize=figsize,
        shoreline_orientation_deg=shoreline_orientation_deg,
        show_orientation_arrows=show_orientation_arrows,
        longshore_offset_deg=longshore_offset_deg,
        arrow_length_frac=arrow_length_frac,
        radial_range=radial_range,
        energy_radial_range=energy_radial_range,
        occurrence_radial_range=occurrence_radial_range,
        hindcast_source_label=source_label,
    )


def plot_historical_partition_wave_rose(
    *,
    coordinate: tuple[float, float] | None = None,
    site_index: int | None = None,
    partition_ids: Sequence[int] = (0, 1),
    time_start: str = "1993-01-01",
    time_end: str = "2023-12-31",
    aggregation: str = "hourly",
    n_dir_bins: int = 16,
    hs_range: tuple[float, float] = (0.0, 6.0),
    hs_bin_m: float = 1.0,
    energy_power: float = 2.0,
    partitions_folder: str | Path | None = None,
    partitions_dataset: HistoricalDataset = "merged_grids",
    project_root: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    shoreline_orientation_deg: float | None = None,
    show_orientation_arrows: bool | None = None,
    longshore_offset_deg: float = 90.0,
    arrow_length_frac: float = 0.88,
    radial_range: float | str | Sequence[float] | None = None,
    energy_radial_range: float | str | Sequence[float] | None = None,
    occurrence_radial_range: float | str | Sequence[float] | None = None,
) -> dict[str, Any]:
    """
    Partition wave roses in a 2×N grid (default 2×2 for partitions 0 and 1).

    Rows = partitions; columns = % occurrence rose (left) and % energy rose (right).
    Uses ``phs{pid}`` and ``dp{pid}`` from the partitions folder (``merged_grids``).

    ``radial_range`` fixes the % energy rose radial grid (see ``plot_buoy_wave_rose``).
    """
    series, info = load_historical_partition_hs_dp(
        coordinate=coordinate,
        site_index=site_index,
        partition_ids=partition_ids,
        aggregation=aggregation,
        time_start=time_start,
        time_end=time_end,
        partitions_folder=partitions_folder,
        partitions_dataset=partitions_dataset,
        project_root=project_root,
    )
    if not series:
        return {"n_samples": 0, **info}

    partition_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for pid, (hs_da, dp_da) in series.items():
        h_vals, d_vals = _aligned_hs_dp_numpy(hs_da, dp_da)
        if h_vals.size == 0:
            print(f"SKIP partition {pid}: no valid phs/dp for wave rose")
            continue
        partition_arrays[pid] = (h_vals, d_vals)

    if not partition_arrays:
        return {"n_samples": 0, **info}

    period_note = (
        f"{pd.Timestamp(time_start).date()} – {pd.Timestamp(time_end).date()}"
    )
    if site_index is not None:
        site_label = (
            f"hindcast site {info['site_index']} "
            f"({info['lat']:.3f}, {info['lon']:.3f})"
        )
    else:
        dist = info.get("distance_km", 0.0)
        site_label = (
            f"hindcast site {info['site_index']} "
            f"({info['lat']:.3f}, {info['lon']:.3f}), {dist:.2f} km from target"
        )

    pids_loaded = [int(p) for p in partition_ids if int(p) in partition_arrays]
    sector_deg = 360.0 / int(n_dir_bins)
    n_total = sum(int(partition_arrays[p][0].size) for p in pids_loaded)
    footnote = (
        f"hindcast phs/dp partitions {pids_loaded} | {n_total:,} samples (all parts) | "
        f"left: % occurrence per {sector_deg:g}° sector; "
        f"right: % of total ΣHs^{energy_power:g} energy per sector; "
        f"stack = Hs bin (colored) | radial circles = %"
    )

    stats = _plot_hs_dp_partition_grid_wave_rose(
        partition_arrays,
        partition_ids=pids_loaded,
        site_label=site_label,
        period_note=period_note,
        aggregation=str(aggregation),
        footnote=footnote,
        n_dir_bins=n_dir_bins,
        hs_range=hs_range,
        hs_bin_m=hs_bin_m,
        energy_power=energy_power,
        figsize=figsize,
        shoreline_orientation_deg=shoreline_orientation_deg,
        show_orientation_arrows=show_orientation_arrows,
        longshore_offset_deg=longshore_offset_deg,
        arrow_length_frac=arrow_length_frac,
        radial_range=radial_range,
        energy_radial_range=energy_radial_range,
        occurrence_radial_range=occurrence_radial_range,
    )

    return {
        "n_samples": n_total,
        "partition_stats": stats,
        "partition_ids": pids_loaded,
        "energy_power": float(energy_power),
        "aggregation": str(aggregation),
        "time_start": str(time_start),
        "time_end": str(time_end),
        "partitions_dataset": str(partitions_dataset),
        "shoreline_orientation_deg": shoreline_orientation_deg,
        **info,
    }


# --- WHACS wave roses (north_carolina_*_WHACS.nc, no 02/03/04 sub-grids) -----


def _is_whacs_main_grid_nc(path: str | Path) -> bool:
    """True for ``north_carolina_hs_WHACS.nc`` style files (exclude ``_02_``, ``_03_``, ``_04_``)."""
    name = Path(path).name
    return (
        name.startswith("north_carolina_")
        and name.endswith("_WHACS.nc")
        and _WHACS_REGION_SUBGRID.search(name) is None
    )


def _whacs_folder_path(whacs_folder: str | Path | None) -> Path:
    folder = Path(whacs_folder or DEFAULT_WHACS_FOLDER)
    if not folder.is_dir():
        raise FileNotFoundError(f"WHACS folder not found: {folder}")
    return folder


def _whacs_nc_path(folder: Path, stem: str) -> Path:
    path = folder / f"{stem}_WHACS.nc"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _whacs_var_from_stem(stem: str) -> str:
    if stem == "north_carolina_hs":
        return "hs"
    if stem == "north_carolina_dp":
        return "dp"
    if stem.startswith("north_carolina_phs"):
        return "phs" + stem.removeprefix("north_carolina_phs")
    if stem.startswith("north_carolina_pdp"):
        return "pdp" + stem.removeprefix("north_carolina_pdp")
    raise ValueError(f"Unknown WHACS stem: {stem}")


def _whacs_seapoint_lat_lon_arrays(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, str]:
    """Per-seapoint lat/lon (uses first time step when coordinates vary in time)."""
    lat_name, lon_name = _guess_lat_lon(ds)
    site_dim = _guess_site_dim(ds, lat_name, lon_name)
    lat = ds[lat_name] if lat_name in ds else ds.coords[lat_name]
    lon = ds[lon_name] if lon_name in ds else ds.coords[lon_name]
    if lat.ndim == 2 and "time" in lat.dims:
        lat = lat.isel(time=0)
        lon = lon.isel(time=0)
    return (
        np.asarray(lat.values, dtype=float).reshape(-1),
        np.asarray(lon.values, dtype=float).reshape(-1),
        site_dim,
    )


def find_nearest_whacs_site(
    coordinate: tuple[float, float],
    whacs_folder: str | Path | None = None,
    *,
    reference_stem: str = "north_carolina_hs",
    max_distance_km: float | None = 5.0,
) -> dict[str, Any]:
    """
    Nearest WHACS seapoint to ``coordinate`` ``(lat, lon)`` in the main (non-02/03/04) grid.

    Raises ``ValueError`` if distance exceeds ``max_distance_km`` (pass ``None`` to skip).
    """
    folder = _whacs_folder_path(whacs_folder)
    ref_nc = _whacs_nc_path(folder, reference_stem)
    if not _is_whacs_main_grid_nc(ref_nc):
        raise ValueError(f"Reference file is not a main-grid WHACS NetCDF: {ref_nc.name}")

    lat_t, lon_t = float(coordinate[0]), float(coordinate[1])
    with xr.open_dataset(ref_nc) as ds:
        src_lat, src_lon, site_dim = _whacs_seapoint_lat_lon_arrays(ds)
        n_site = int(ds.sizes[site_dim])

    dist_m = haversine_m(lat_t, lon_t, src_lat, src_lon)
    i_pick = int(np.argmin(dist_m))
    dist_km = float(dist_m[i_pick] / 1000.0)
    if max_distance_km is not None and dist_km > float(max_distance_km):
        raise ValueError(
            f"No WHACS seapoint within {max_distance_km:g} km of ({lat_t:.3f}, {lon_t:.3f}); "
            f"nearest is {dist_km:.2f} km (seapoint {i_pick} at "
            f"{src_lat[i_pick]:.3f}, {src_lon[i_pick]:.3f})"
        )

    info = {
        "site_index": i_pick,
        "seapoint_dim": site_dim,
        "n_seapoints": n_site,
        "lat": float(src_lat[i_pick]),
        "lon": float(src_lon[i_pick]),
        "distance_km": dist_km,
        "target_lat": lat_t,
        "target_lon": lon_t,
        "reference_nc": str(ref_nc),
        "whacs_folder": str(folder),
    }
    print(
        f"WHACS seapoint {i_pick} ({info['lat']:.3f}, {info['lon']:.3f}), "
        f"{dist_km:.2f} km from target ({lat_t:.3f}, {lon_t:.3f})"
    )
    return info


def _whacs_seapoint_info_at_index(
    folder: Path,
    site_index: int,
    *,
    reference_stem: str = "north_carolina_hs",
) -> dict[str, Any]:
    """Metadata for one WHACS seapoint (no distance-to-target check)."""
    ref_nc = _whacs_nc_path(folder, reference_stem)
    site_index = int(site_index)
    with xr.open_dataset(ref_nc) as ds:
        src_lat, src_lon, site_dim = _whacs_seapoint_lat_lon_arrays(ds)
        n_site = int(ds.sizes[site_dim])
    if site_index < 0 or site_index >= n_site:
        raise ValueError(f"seapoint index {site_index} out of range [0, {n_site - 1}]")
    return {
        "site_index": site_index,
        "seapoint_dim": site_dim,
        "n_seapoints": n_site,
        "lat": float(src_lat[site_index]),
        "lon": float(src_lon[site_index]),
        "reference_nc": str(ref_nc),
        "whacs_folder": str(folder),
    }


def _load_whacs_var_at_site(
    folder: Path,
    stem: str,
    site_index: int,
    *,
    aggregation: str,
    time_start,
    time_end,
) -> xr.DataArray:
    nc_path = _whacs_nc_path(folder, stem)
    var_name = _whacs_var_from_stem(stem)
    site_index = int(site_index)

    with xr.open_dataset(nc_path) as ds:
        lat_name, lon_name = _guess_lat_lon(ds)
        site_dim = _guess_site_dim(ds, lat_name, lon_name)
        n_site = int(ds.sizes[site_dim])
        if site_index < 0 or site_index >= n_site:
            raise ValueError(
                f"seapoint index {site_index} out of range [0, {n_site - 1}] in {nc_path.name}"
            )
        nc_var = _resolve_nc_var(ds, var_name)
        da = ds[nc_var].isel({site_dim: site_index})
        da = apply_aggregation(da, var_name, aggregation)
        if time_start is not None or time_end is not None:
            da = da.sel(time=slice(time_start, time_end))
    return da


def load_whacs_bulk_hs_dp(
    coordinate: tuple[float, float],
    whacs_folder: str | Path | None = None,
    *,
    site_index: int | None = None,
    aggregation: str = "hourly",
    time_start=None,
    time_end=None,
    max_distance_km: float = 5.0,
) -> tuple[tuple[xr.DataArray, xr.DataArray] | None, dict[str, Any]]:
    """Load bulk ``hs`` and ``dp`` from main-grid WHACS files at one seapoint."""
    folder = _whacs_folder_path(whacs_folder)
    lat_t, lon_t = float(coordinate[0]), float(coordinate[1])
    if site_index is None:
        info = find_nearest_whacs_site(
            coordinate, folder, max_distance_km=max_distance_km
        )
        site_index = int(info["site_index"])
    else:
        site_index = int(site_index)
        info = _whacs_seapoint_info_at_index(folder, site_index)
        info["target_lat"] = lat_t
        info["target_lon"] = lon_t
        info["distance_km"] = float(
            haversine_m(lat_t, lon_t, info["lat"], info["lon"]) / 1000.0
        )

    hs = _load_whacs_var_at_site(
        folder, "north_carolina_hs", site_index,
        aggregation=aggregation, time_start=time_start, time_end=time_end,
    )
    dp = _load_whacs_var_at_site(
        folder, "north_carolina_dp", site_index,
        aggregation=aggregation, time_start=time_start, time_end=time_end,
    )
    info = dict(info)
    info["site_index"] = site_index
    return (hs, dp), info


def load_whacs_partition_hs_dp(
    coordinate: tuple[float, float],
    whacs_folder: str | Path | None = None,
    *,
    partition_ids: Sequence[int] = (0, 1),
    site_index: int | None = None,
    aggregation: str = "hourly",
    time_start=None,
    time_end=None,
    max_distance_km: float = 5.0,
) -> tuple[dict[int, tuple[xr.DataArray, xr.DataArray]], dict[str, Any]]:
    """
    Load partition ``phs*`` / ``pdp*`` from main-grid WHACS (e.g. ``north_carolina_phs1``).
    """
    folder = _whacs_folder_path(whacs_folder)
    lat_t, lon_t = float(coordinate[0]), float(coordinate[1])
    if site_index is None:
        info = find_nearest_whacs_site(
            coordinate, folder, max_distance_km=max_distance_km
        )
        site_index = int(info["site_index"])
    else:
        site_index = int(site_index)
        info = _whacs_seapoint_info_at_index(folder, site_index)
        info["target_lat"] = lat_t
        info["target_lon"] = lon_t
        info["distance_km"] = float(
            haversine_m(lat_t, lon_t, info["lat"], info["lon"]) / 1000.0
        )

    out: dict[int, tuple[xr.DataArray, xr.DataArray]] = {}
    for pid in [int(p) for p in partition_ids]:
        hs_stem = f"north_carolina_phs{pid}"
        dp_stem = f"north_carolina_pdp{pid}"
        try:
            hs = _load_whacs_var_at_site(
                folder, hs_stem, site_index,
                aggregation=aggregation, time_start=time_start, time_end=time_end,
            )
            dp = _load_whacs_var_at_site(
                folder, dp_stem, site_index,
                aggregation=aggregation, time_start=time_start, time_end=time_end,
            )
        except FileNotFoundError:
            print(f"SKIP WHACS partition {pid}: missing {hs_stem} or {dp_stem}")
            continue
        out[pid] = (hs, dp)

    info = dict(info)
    info["site_index"] = site_index
    info["partition_ids_loaded"] = sorted(out.keys())
    return out, info


def _open_whacs_spectra_dataset(spectra: str | Path | xr.Dataset) -> xr.Dataset:
    if isinstance(spectra, xr.Dataset):
        return spectra
    path = resolve_path(spectra)
    ds = xr.open_dataset(path)
    if "__xarray_dataarray_variable__" in ds.data_vars:
        ds = ds.rename({"__xarray_dataarray_variable__": "efth"})
    return ds


def _hs_dp_from_spectra(
    spectra: xr.Dataset,
    *,
    part: int | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    sub = spectra.sel(part=part) if part is not None and "part" in spectra.dims else spectra
    hs = getattr(sub.spec, "hs")()
    dp = getattr(sub.spec, "dp")()
    return hs, dp


def _prepare_hs_dp_for_wave_rose(
    hs: xr.DataArray,
    dp: xr.DataArray,
    *,
    time_start=None,
    time_end=None,
    aggregation: str = "hourly",
) -> tuple[xr.DataArray, xr.DataArray]:
    if time_start is not None or time_end is not None:
        hs = hs.sel(time=slice(time_start, time_end))
        dp = dp.sel(time=slice(time_start, time_end))
    hs = apply_aggregation(hs, "hs", aggregation)
    dp = apply_aggregation(dp, "dp", aggregation)
    return hs, dp


def wave_roses_from_spectra(
    spectra: str | Path | xr.Dataset,
    *,
    hs: xr.DataArray | None = None,
    dp: xr.DataArray | None = None,
    site_label: str = "WHACS spectra",
    partition_ids: Sequence[int] | None = None,
    plot_bulk: bool = True,
    plot_partitions: bool = True,
    time_start=None,
    time_end=None,
    aggregation: str = "hourly",
    n_dir_bins: int = 16,
    hs_range: tuple[float, float] = (0.0, 6.0),
    hs_bin_m: float = 1.0,
    energy_power: float = 2.0,
    figsize_bulk: tuple[float, float] = (14.0, 7.0),
    figsize_partitions: tuple[float, float] | None = None,
    shoreline_orientation_deg: float | None = None,
    show_orientation_arrows: bool | None = None,
    longshore_offset_deg: float = 90.0,
    arrow_length_frac: float = 0.88,
    radial_range: float | str | Sequence[float] | None = None,
    energy_radial_range: float | str | Sequence[float] | None = None,
    occurrence_radial_range: float | str | Sequence[float] | None = None,
) -> dict[str, Any]:
    """
    Wave roses from a WHACS ``efth`` spectra NetCDF (via ``wavespectra`` Hs/Dp).

    Pass a loaded dataset or path (``44014_spec_WHACS_buoy_correted.nc``). Optionally
    pass precomputed ``hs`` / ``dp`` arrays (e.g. from ``ds.spec.hs()`` / ``dp()``).
    If the dataset has a ``part`` dimension, set ``partition_ids`` to plot partition roses.

    ``radial_range`` fixes the % energy rose radial grid (see ``plot_buoy_wave_rose``).
    """
    _warn_native(aggregation)
    ds = _open_whacs_spectra_dataset(spectra)

    period_note = ""
    if time_start is not None or time_end is not None:
        t0 = pd.Timestamp(time_start).date() if time_start is not None else "..."
        t1 = pd.Timestamp(time_end).date() if time_end is not None else "latest"
        period_note = f"{t0} – {t1}"

    result: dict[str, Any] = {"site_label": site_label}

    if plot_bulk:
        hs_da = hs
        dp_da = dp
        if hs_da is None or dp_da is None:
            hs_da, dp_da = _hs_dp_from_spectra(ds)
        hs_da, dp_da = _prepare_hs_dp_for_wave_rose(
            hs_da,
            dp_da,
            time_start=time_start,
            time_end=time_end,
            aggregation=aggregation,
        )
        h_vals, d_vals = _aligned_hs_dp_numpy(hs_da, dp_da)
        if h_vals.size == 0:
            print("SKIP spectra bulk wave rose: no valid hs/dp")
            result["bulk"] = {"n_samples": 0}
        else:
            sector_deg = 360.0 / int(n_dir_bins)
            n_samples, e_total = _plot_hs_dp_dual_wave_rose(
                h_vals,
                d_vals,
                site_label=f"{site_label} bulk",
                period_note=period_note,
                aggregation=str(aggregation),
                footnote=(
                    f"Spectra-derived hs/dp | {int(h_vals.size):,} samples | "
                    f"left: % per {sector_deg:g}° sector; "
                    f"right: % of ΣHs^{energy_power:g} energy per sector"
                ),
                n_dir_bins=n_dir_bins,
                hs_range=hs_range,
                hs_bin_m=hs_bin_m,
                energy_power=energy_power,
                figsize=figsize_bulk,
                shoreline_orientation_deg=shoreline_orientation_deg,
                show_orientation_arrows=show_orientation_arrows,
                longshore_offset_deg=longshore_offset_deg,
                arrow_length_frac=arrow_length_frac,
                radial_range=radial_range,
                energy_radial_range=energy_radial_range,
                occurrence_radial_range=occurrence_radial_range,
            )
            result["bulk"] = {
                "n_samples": n_samples,
                "energy_total_hs_power": e_total,
            }

    has_parts = "part" in ds.dims
    if plot_partitions and has_parts:
        pids = (
            [int(p) for p in partition_ids]
            if partition_ids is not None
            else [int(p) for p in np.asarray(ds.part.values).ravel()]
        )
        partition_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for pid in pids:
            try:
                hs_da, dp_da = _hs_dp_from_spectra(ds, part=pid)
            except Exception as exc:
                print(f"SKIP spectra partition {pid}: {exc}")
                continue
            hs_da, dp_da = _prepare_hs_dp_for_wave_rose(
                hs_da,
                dp_da,
                time_start=time_start,
                time_end=time_end,
                aggregation=aggregation,
            )
            h_vals, d_vals = _aligned_hs_dp_numpy(hs_da, dp_da)
            if h_vals.size == 0:
                print(f"SKIP spectra partition {pid}: no valid hs/dp")
                continue
            partition_arrays[pid] = (h_vals, d_vals)

        if partition_arrays:
            pids_loaded = sorted(partition_arrays.keys())
            sector_deg = 360.0 / int(n_dir_bins)
            n_total = sum(int(partition_arrays[p][0].size) for p in pids_loaded)
            stats = _plot_hs_dp_partition_grid_wave_rose(
                partition_arrays,
                partition_ids=pids_loaded,
                site_label=site_label,
                period_note=period_note,
                aggregation=str(aggregation),
                footnote=(
                    f"Spectra partitions {pids_loaded} | {n_total:,} samples | "
                    f"left: % per {sector_deg:g}° sector; "
                    f"right: % of ΣHs^{energy_power:g} energy per sector"
                ),
                n_dir_bins=n_dir_bins,
                hs_range=hs_range,
                hs_bin_m=hs_bin_m,
                energy_power=energy_power,
                figsize=figsize_partitions,
                shoreline_orientation_deg=shoreline_orientation_deg,
                show_orientation_arrows=show_orientation_arrows,
                longshore_offset_deg=longshore_offset_deg,
                arrow_length_frac=arrow_length_frac,
                radial_range=radial_range,
                energy_radial_range=energy_radial_range,
                occurrence_radial_range=occurrence_radial_range,
            )
            result["partitions"] = stats
            result["partition_ids"] = pids_loaded
        else:
            print("SKIP spectra partition wave roses: no partition data loaded")
            result["partitions"] = {}
    elif plot_partitions and not has_parts:
        print("SKIP spectra partition wave roses: dataset has no `part` dimension")

    return result


def wave_roses_whacs(
    coordinate: tuple[float, float],
    whacs_folder: str | Path | None = None,
    *,
    site_index: int | None = None,
    partition_ids: Sequence[int] = (0, 1),
    plot_bulk: bool = True,
    plot_partitions: bool = True,
    time_start=None,
    time_end=None,
    aggregation: str = "hourly",
    n_dir_bins: int = 16,
    hs_range: tuple[float, float] = (0.0, 6.0),
    hs_bin_m: float = 1.0,
    energy_power: float = 2.0,
    max_distance_km: float = 5.0,
    figsize_bulk: tuple[float, float] = (14.0, 7.0),
    figsize_partitions: tuple[float, float] | None = None,
    shoreline_orientation_deg: float | None = None,
    show_orientation_arrows: bool | None = None,
    longshore_offset_deg: float = 90.0,
    arrow_length_frac: float = 0.88,
    radial_range: float | str | Sequence[float] | None = None,
    energy_radial_range: float | str | Sequence[float] | None = None,
    occurrence_radial_range: float | str | Sequence[float] | None = None,
) -> dict[str, Any]:
    """
    Wave roses from ShoreShop WHACS main grid (not ``_02_`` / ``_03_`` / ``_04_`` files).

    - **Bulk**: ``north_carolina_hs_WHACS.nc`` + ``north_carolina_dp_WHACS.nc``
    - **Partitions**: ``north_carolina_phs{pid}_WHACS.nc`` + ``north_carolina_pdp{pid}_WHACS.nc``

    Picks the nearest seapoint to ``coordinate`` ``(lat, lon)``; fails if farther than
    ``max_distance_km`` (default 5 km) unless ``site_index`` is given explicitly.

    The main WHACS grid has only ~11 regional seapoints; buoy-scale targets (e.g. 44014)
    may require ``site_index`` after inspecting ``find_nearest_whacs_site`` without the
    distance limit.

    ``radial_range`` fixes the % energy rose radial grid (see ``plot_buoy_wave_rose``).
    """
    _warn_native(aggregation)
    lat, lon = float(coordinate[0]), float(coordinate[1])

    result: dict[str, Any] = {
        "coordinate": (lat, lon),
        "whacs_folder": str(_whacs_folder_path(whacs_folder)),
        "site_index": site_index,
    }

    period_note = ""
    if time_start is not None or time_end is not None:
        t0 = pd.Timestamp(time_start).date() if time_start is not None else "..."
        t1 = pd.Timestamp(time_end).date() if time_end is not None else "latest"
        period_note = f"{t0} – {t1}"

    site_label = ""
    bulk_stats: dict[str, Any] = {}

    if plot_bulk:
        pair, info = load_whacs_bulk_hs_dp(
            (lat, lon),
            whacs_folder,
            site_index=site_index,
            aggregation=aggregation,
            time_start=time_start,
            time_end=time_end,
            max_distance_km=max_distance_km,
        )
        result["site_info"] = info
        site_label = (
            f"WHACS site {info['site_index']} ({info['lat']:.3f}, {info['lon']:.3f}), "
            f"{info['distance_km']:.2f} km from ({lat:.3f}, {lon:.3f})"
        )
        if pair is None:
            print("SKIP WHACS bulk wave rose: no data")
        else:
            h_vals, d_vals = _aligned_hs_dp_numpy(*pair)
            if h_vals.size == 0:
                print("SKIP WHACS bulk wave rose: no valid hs/dp")
            else:
                sector_deg = 360.0 / int(n_dir_bins)
                n_samples, e_total = _plot_hs_dp_dual_wave_rose(
                    h_vals,
                    d_vals,
                    site_label=f"WHACS bulk — {site_label}",
                    period_note=period_note,
                    aggregation=str(aggregation),
                    footnote=(
                        f"WHACS bulk hs/dp | {int(h_vals.size):,} samples | "
                        f"left: % per {sector_deg:g}° sector; "
                        f"right: % of ΣHs^{energy_power:g} energy per sector"
                    ),
                    n_dir_bins=n_dir_bins,
                    hs_range=hs_range,
                    hs_bin_m=hs_bin_m,
                    energy_power=energy_power,
                    figsize=figsize_bulk,
                    shoreline_orientation_deg=shoreline_orientation_deg,
                    show_orientation_arrows=show_orientation_arrows,
                    longshore_offset_deg=longshore_offset_deg,
                    arrow_length_frac=arrow_length_frac,
                    radial_range=radial_range,
                    energy_radial_range=energy_radial_range,
                    occurrence_radial_range=occurrence_radial_range,
                )
                bulk_stats = {
                    "n_samples": n_samples,
                    "energy_total_hs_power": e_total,
                }
        result["bulk"] = bulk_stats

    if plot_partitions:
        series, pinfo = load_whacs_partition_hs_dp(
            (lat, lon),
            whacs_folder,
            partition_ids=partition_ids,
            site_index=result.get("site_info", {}).get("site_index", site_index),
            aggregation=aggregation,
            time_start=time_start,
            time_end=time_end,
            max_distance_km=max_distance_km,
        )
        if "site_info" not in result:
            result["site_info"] = pinfo
        if not site_label:
            site_label = (
                f"WHACS site {pinfo['site_index']} ({pinfo['lat']:.3f}, {pinfo['lon']:.3f}), "
                f"{pinfo['distance_km']:.2f} km from ({lat:.3f}, {lon:.3f})"
            )

        partition_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for pid, (hs_da, dp_da) in series.items():
            h_vals, d_vals = _aligned_hs_dp_numpy(hs_da, dp_da)
            if h_vals.size == 0:
                print(f"SKIP WHACS partition {pid}: no valid phs/pdp")
                continue
            partition_arrays[pid] = (h_vals, d_vals)

        if partition_arrays:
            pids_loaded = sorted(partition_arrays.keys())
            sector_deg = 360.0 / int(n_dir_bins)
            n_total = sum(int(partition_arrays[p][0].size) for p in pids_loaded)
            stats = _plot_hs_dp_partition_grid_wave_rose(
                partition_arrays,
                partition_ids=pids_loaded,
                site_label=site_label,
                period_note=period_note,
                aggregation=str(aggregation),
                footnote=(
                    f"WHACS phs/pdp partitions {pids_loaded} | {n_total:,} samples | "
                    f"left: % per {sector_deg:g}° sector; "
                    f"right: % of ΣHs^{energy_power:g} energy per sector"
                ),
                n_dir_bins=n_dir_bins,
                hs_range=hs_range,
                hs_bin_m=hs_bin_m,
                energy_power=energy_power,
                figsize=figsize_partitions,
                shoreline_orientation_deg=shoreline_orientation_deg,
                show_orientation_arrows=show_orientation_arrows,
                longshore_offset_deg=longshore_offset_deg,
                arrow_length_frac=arrow_length_frac,
                radial_range=radial_range,
                energy_radial_range=energy_radial_range,
                occurrence_radial_range=occurrence_radial_range,
            )
            result["partitions"] = stats
            result["partition_ids"] = pids_loaded
        else:
            print("SKIP WHACS partition wave roses: no partition data loaded")
            result["partitions"] = {}

    return result


# --- Buoy vs hindcast longshore transport (Qs) --------------------------------


def _site_note_from_info(coordinate: tuple[float, float], info: dict) -> str:
    lat, lon = coordinate
    path_bit = ""
    if info.get("path"):
        path_bit = f" | {Path(info['path']).name}"
    return (
        f"site {info['site_index']} ({info['lat']:.3f}, {info['lon']:.3f}), "
        f"{info['distance_km']:.2f} km from ({lat:.3f}, {lon:.3f}){path_bit}"
    )


def _qs_formula_label(
    qs_formula: QsFormula,
    K: float,
    *,
    wave_direction: str = "dp",
) -> str:
    dir_name = _normalize_wave_direction(wave_direction).upper()
    if qs_formula == "deep":
        return f"deep-water Qs (K₂={K:g}, {dir_name})"
    return f"CERC Qs (K={K:g}, {dir_name})"


def _compute_qs_series(
    hs: xr.DataArray,
    dp: xr.DataArray,
    shoreline_orientation_deg: float,
    *,
    tp: xr.DataArray | None = None,
    K: float = DEFAULT_LONGSHORE_K,
    qs_formula: QsFormula = "cerc",
) -> xr.DataArray:
    qs = _region_qs_batch(
        hs,
        dp,
        shoreline_orientation_deg,
        K=K,
        tp=tp,
        qs_formula=qs_formula,
    )
    qs.name = "Qs"
    return qs


def load_kma_combined_qs(
    coordinate: tuple[float, float],
    shoreline_orientation_deg: float,
    *,
    qs_formula: QsFormula = "cerc",
    aggregation: str = "daily",
    time_start=None,
    time_end=None,
    K: float = DEFAULT_LONGSHORE_K,
    cluster_cases_root: str | Path,
    kma_bmu_csv: str | Path | None = None,
    combine_with_binwaves: bool = True,
    historical_folder: str | Path | None = None,
    historical_dataset: HistoricalDataset = "merged_grids",
    project_root: str | Path | None = None,
) -> tuple[xr.DataArray | None, dict]:
    """Qs from BinWaves + KMA cluster combined bulk waves at one coordinate."""
    from utils import kma_cluster_swan as kcs

    qs_formula = str(qs_formula).lower()  # type: ignore[assignment]
    if qs_formula not in ("cerc", "deep"):
        raise ValueError(f"qs_formula must be 'cerc' or 'deep'; got {qs_formula!r}")

    root = project_root or kcs._default_project_root()
    bmu_csv = kma_bmu_csv or kcs.DEFAULT_KMA_BMU_CSV
    hist_folder = resolve_historical_folder(
        historical_folder=historical_folder,
        historical_dataset=historical_dataset,
        project_root=root,
    )

    t_start = pd.Timestamp(time_start) if time_start is not None else None
    t_end = pd.Timestamp(time_end) if time_end is not None else None
    if t_start is None or t_end is None:
        print("SKIP KMA combined Qs: time_start and time_end are required")
        return None, {}

    bmu_csv_path = resolve_path(bmu_csv, root)
    bmu_series = kcs.load_bmu_assignments(bmu_csv_path)
    bmu_start, bmu_end = bmu_series.index[0], bmu_series.index[-1]
    t_start = max(t_start, bmu_start)
    t_end = min(t_end, bmu_end + pd.Timedelta(hours=3))
    if t_start > t_end:
        print("SKIP KMA combined Qs: no overlap with KMA BMU assignments")
        return None, {}

    try:
        hs_a, tp_a, dp_a, info = kcs.build_kma_combined_wave_dataarrays(
            coordinate,
            time_start=t_start,
            time_end=t_end,
            aggregation=aggregation,
            cluster_cases_root=cluster_cases_root,
            kma_bmu_csv=bmu_csv,
            historical_folder=hist_folder,
            combine_with_binwaves=combine_with_binwaves,
            project_root=root,
        )
    except Exception as exc:
        print(f"SKIP KMA combined Qs: {exc}")
        return None, {}

    if hs_a.sizes.get("time", 0) == 0:
        print("SKIP KMA combined Qs: no overlapping times after aggregation")
        return None, {}

    if qs_formula == "deep":
        qs = _compute_qs_series(
            hs_a, dp_a, shoreline_orientation_deg, tp=tp_a, K=K, qs_formula=qs_formula
        )
    else:
        qs = _compute_qs_series(
            hs_a, dp_a, shoreline_orientation_deg, K=K, qs_formula=qs_formula
        )
    info = dict(info)
    info["qs_formula"] = qs_formula
    info["n_points"] = int(qs.sizes.get("time", qs.size))
    return qs, info


def load_qs_historical_with_partitions(
    coordinate: tuple[float, float],
    shoreline_orientation_deg: float,
    *,
    qs_formula: QsFormula = "cerc",
    aggregation: str = "daily",
    time_start=None,
    time_end=None,
    K: float = DEFAULT_LONGSHORE_K,
    historical_folder: str | Path | None = None,
    historical_dataset: HistoricalDataset = "merged_grids",
    partitions_folder: str | Path | None = None,
    partitions_dataset: HistoricalDataset = "merged_grids",
    partition_ids: Sequence[int] | None = None,
    include_bulk: bool = True,
    include_partitions_sum: bool = True,
    include_binwaves_reference: bool | None = None,
    binwaves_reference_folder: str | Path | None = None,
    wave_direction: str = "dp",
    project_root: str | Path | None = None,
) -> tuple[dict[str, tuple[xr.DataArray, dict]], str]:
    """
    Historical Qs at one coordinate from bulk and/or wave partitions.

    ``qs_formula``: ``\"cerc\"`` (default) or ``\"deep\"`` (requires Tp from hindcast).
    ``wave_direction``: ``\"dp\"`` (peak direction) or ``\"dm\"`` (mean direction).
    """
    qs_formula = str(qs_formula).lower()  # type: ignore[assignment]
    if qs_formula not in ("cerc", "deep"):
        raise ValueError(f"qs_formula must be 'cerc' or 'deep'; got {qs_formula!r}")
    wave_direction = _normalize_wave_direction(wave_direction)

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

    series: dict[str, tuple[xr.DataArray, dict]] = {}
    site_note = ""

    def _load_qs_triplet(
        folder: str | Path,
        hs_name: str,
        dp_name: str,
        tp_name: str,
    ) -> tuple[xr.DataArray | None, dict]:
        folder = Path(resolve_path(folder, project_root))
        try:
            hs_path = _resolve_variable_nc_path(folder, hs_name, project_root)
            dp_path = _resolve_variable_nc_path(folder, dp_name, project_root)
        except FileNotFoundError:
            return None, {}
        hs_da, info = load_timeseries_at_coordinate(
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
        tp_da: xr.DataArray | None = None
        if qs_formula == "deep" and tp_name:
            try:
                tp_path = _resolve_variable_nc_path(folder, tp_name, project_root)
            except FileNotFoundError:
                print(f"SKIP {hs_name}: missing {tp_name} for deep-water Qs")
                return None, {}
            tp_da, _ = load_timeseries_at_coordinate(
                tp_path,
                coordinate,
                variable=tp_name,
                aggregation=aggregation,
                time_start=time_start,
                time_end=time_end,
                project_root=project_root,
            )
        hs_a, dp_a = xr.align(hs_da, dp_da, join="inner")
        if hs_a.sizes.get("time", 0) == 0:
            return None, {}
        if qs_formula == "deep":
            if tp_da is None:
                return None, {}
            hs_a, dp_a, tp_a = xr.align(hs_a, dp_a, tp_da, join="inner")
            if hs_a.sizes.get("time", 0) == 0:
                return None, {}
            qs = _compute_qs_series(
                hs_a, dp_a, shoreline_orientation_deg, tp=tp_a, K=K, qs_formula=qs_formula
            )
        else:
            qs = _compute_qs_series(
                hs_a, dp_a, shoreline_orientation_deg, K=K, qs_formula=qs_formula
            )
        info = dict(info)
        info["n_points"] = int(qs.sizes.get("time", qs.size))
        info["qs_formula"] = qs_formula
        info["wave_direction"] = wave_direction
        return qs, info

    if include_bulk:
        hs_name, tp_name, dp_name = _bulk_file_triplet(
            qs_formula, wave_direction=wave_direction
        )
        bulk_label = _historical_bulk_series_label(historical_folder, project_root)
        qs_bulk, info_bulk = _load_qs_triplet(hist_folder, hs_name, dp_name, tp_name)
        if qs_bulk is not None:
            series[bulk_label] = (qs_bulk, info_bulk)
            site_note = _site_note_from_info(coordinate, info_bulk)
        else:
            print(f"SKIP {bulk_label}: no overlapping hs/dp/tp times for Qs")

        if include_binwaves_reference is None:
            include_binwaves_reference = _uses_kma_merged_grids_folder(
                historical_folder, project_root
            )
        if include_binwaves_reference:
            ref_folder = _resolve_binwaves_reference_folder(
                historical_folder=historical_folder,
                historical_dataset=historical_dataset,
                partitions_folder=partitions_folder,
                partitions_dataset=partitions_dataset,
                binwaves_reference_folder=binwaves_reference_folder,
                project_root=project_root,
            )
            if ref_folder is not None:
                qs_bw, info_bw = _load_qs_triplet(ref_folder, hs_name, dp_name, tp_name)
                if qs_bw is not None:
                    series[BINWAVES_BULK_LABEL] = (qs_bw, info_bw)
                else:
                    print(
                        f"SKIP {BINWAVES_BULK_LABEL}: no overlapping hs/dp/tp times "
                        f"in {ref_folder}"
                    )

    pids = (
        [int(i) for i in partition_ids]
        if partition_ids is not None
        else _partition_ids_from_folder(part_folder, project_root)
    )
    if partition_ids is not None:
        print(f"Using requested partition_ids: {pids}")
    partition_qs: list[xr.DataArray] = []
    for pid in pids:
        hs_name, tp_name, dp_name = _partition_file_triplet(
            pid, qs_formula, wave_direction=wave_direction
        )
        qs_part, info_p = _load_qs_triplet(part_folder, hs_name, dp_name, tp_name)
        if qs_part is None:
            print(f"SKIP partition {pid}: missing data or no overlap for Qs")
            continue
        label = f"partition {pid}"
        series[label] = (qs_part, info_p)
        partition_qs.append(qs_part.rename(label))
        if not site_note:
            site_note = _site_note_from_info(coordinate, info_p)

    if include_partitions_sum and partition_qs:
        aligned = xr.align(*partition_qs, join="inner")
        if aligned and aligned[0].sizes.get("time", 0) > 0:
            summed = aligned[0].copy()
            for da in aligned[1:]:
                summed = summed + da
            summed.name = "Qs"
            ref_info = series[next(iter(series))][1]
            sum_info = {
                "site_index": ref_info["site_index"],
                "lat": ref_info["lat"],
                "lon": ref_info["lon"],
                "distance_km": ref_info["distance_km"],
                "n_points": int(summed.sizes.get("time", summed.size)),
                "path": str(part_folder),
                "aggregation": str(aggregation).lower(),
                "nc_var": "sum(partitions)",
                "qs_formula": qs_formula,
            }
            series["partitions sum"] = (summed, sum_info)
        else:
            print("SKIP partitions sum: no overlapping time among partitions")

    if partition_ids is not None:
        allowed = {f"partition {int(i)}" for i in partition_ids}
        keep = {
            _historical_bulk_series_label(historical_folder, project_root),
            BINWAVES_BULK_LABEL,
            "partitions sum",
        } | allowed
        if any(_is_historical_bulk_series_key(k) for k in series):
            keep |= {k for k in series if _is_historical_bulk_series_key(k)}
        series = {label: payload for label, payload in series.items() if label in keep}

    return series, site_note


_BUOY_TP_COLUMNS = ("Tp_Buoy", "TP_Buoy", "Tp", "PeakPeriod_Buoy", "Tm_Buoy")


def load_buoy_qs(
    buoy_id: str,
    shoreline_orientation_deg: float,
    *,
    qs_formula: QsFormula = "cerc",
    buoy_data_dir: str | Path = DEFAULT_BUOY_DATA_DIR,
    aggregation: str = "daily",
    time_start=None,
    time_end=None,
    K: float = DEFAULT_LONGSHORE_K,
) -> tuple[xr.DataArray | None, dict]:
    """
    NDBC buoy Qs from ``Hs_Buoy`` / ``Dir_Buoy`` (and Tp column if ``qs_formula='deep'``).
    """
    qs_formula = str(qs_formula).lower()  # type: ignore[assignment]
    if qs_formula not in ("cerc", "deep"):
        raise ValueError(f"qs_formula must be 'cerc' or 'deep'; got {qs_formula!r}")

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

    tp: xr.DataArray | None = None
    if qs_formula == "deep":
        tp_col = next((c for c in _BUOY_TP_COLUMNS if c in df.columns), None)
        if tp_col is None:
            print(
                f"SKIP buoy {buoy_id}: qs_formula='deep' needs a Tp column "
                f"(tried {_BUOY_TP_COLUMNS})"
            )
            return None, {}
        tp = _series_to_dataarray(df[tp_col], name="tp")
        if time_start is not None or time_end is not None:
            tp = tp.sel(time=slice(time_start, time_end))
        tp = apply_aggregation(tp, "tp", aggregation)

    if tp is not None:
        hs_a, dp_a, tp_a = xr.align(hs, dp, tp, join="inner")
        if hs_a.sizes.get("time", 0) == 0:
            print(f"SKIP buoy {buoy_id}: no overlapping Hs/Dir/Tp after aggregation")
            return None, {}
        qs = _compute_qs_series(
            hs_a, dp_a, shoreline_orientation_deg, tp=tp_a, K=K, qs_formula=qs_formula
        )
    else:
        hs_a, dp_a = xr.align(hs, dp, join="inner")
        if hs_a.sizes.get("time", 0) == 0:
            print(f"SKIP buoy {buoy_id}: no overlapping Hs/Dir after aggregation")
            return None, {}
        qs = _compute_qs_series(
            hs_a, dp_a, shoreline_orientation_deg, K=K, qs_formula=qs_formula
        )

    info = {
        "buoy_id": str(buoy_id),
        "path": str(pkl_path) if pkl_path else "",
        "aggregation": str(aggregation).lower(),
        "n_points": int(qs.sizes.get("time", qs.size)),
        "qs_formula": qs_formula,
        "time_start": str(time_start) if time_start is not None else None,
        "time_end": str(time_end) if time_end is not None else None,
    }
    return qs, info


def plot_longshore_transport_for_buoys(
    buoy_ids: str | Sequence[str],
    *,
    qs_formula: QsFormula = "cerc",
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
    include_binwaves_reference: bool | None = None,
    binwaves_reference_folder: str | Path | None = None,
    include_gcm_models: bool = False,
    models: Sequence[str] | None = None,
    scenarios: Mapping[str, dict] | None = None,
    cumulative_reference_start=None,
    time_highlights: Sequence[Any] | None = None,
    wave_direction: str = "dp",
    kma_cluster_cases_root: str | Path | None = None,
    kma_bmu_csv: str | Path | None = None,
    combine_kma_with_binwaves: bool = True,
) -> list[BuoyHistoricalMatch]:
    """
    Compare buoy Qs with merged hindcast Qs at the nearest grid site.

    Supports ``qs_formula='cerc'`` (default) or ``'deep'`` (uses hindcast/buoy Tp).
    ``wave_direction``: ``\"dp\"`` (peak direction) or ``\"dm\"`` (mean direction) for hindcast bulk.
    GCM overlays (if enabled) remain CERC-only.

    ``time_highlights``: optional shaded vertical bands on Qs (and cumulative Qs) plots.

    ``kma_cluster_cases_root``: when set, add a **BinWaves + KMA clusters** Qs series
    reconstructed from static ``CASES_ONLY_WIND/{bmu:03d}`` SWAN runs (hourly BMU mapping).
    """
    from utils import kma_cluster_swan as kcs

    qs_formula = str(qs_formula).lower()  # type: ignore[assignment]
    if qs_formula not in ("cerc", "deep"):
        raise ValueError(f"qs_formula must be 'cerc' or 'deep'; got {qs_formula!r}")
    wave_direction = _normalize_wave_direction(wave_direction)

    _warn_native(aggregation)
    ids = [buoy_ids] if isinstance(buoy_ids, str) else [str(b) for b in buoy_ids]
    if not ids:
        raise ValueError("buoy_ids must contain at least one buoy ID")

    if include_gcm_models and qs_formula != "cerc":
        print(
            "WARN: GCM scenario Qs overlay uses CERC only; "
            f"hindcast/buoy curves use qs_formula={qs_formula!r}"
        )

    formula_note = _qs_formula_label(qs_formula, K, wave_direction=wave_direction)

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
    bulk_label = _historical_bulk_series_label(hist_folder, project_root)
    binwaves_ref_folder = _resolve_binwaves_reference_folder(
        historical_folder=hist_folder,
        partitions_folder=part_folder,
        partitions_dataset=partitions_dataset,
        binwaves_reference_folder=binwaves_reference_folder,
        project_root=project_root,
    )
    if include_binwaves_reference is None:
        include_binwaves_reference = binwaves_ref_folder is not None
    ref_note = (
        f" | BinWaves reference: {binwaves_ref_folder}"
        if include_binwaves_reference and binwaves_ref_folder is not None
        else ""
    )
    print(f"Buoy Qs hindcast folder: {hist_folder} | bulk series: {bulk_label!r}{ref_note}")

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
            qs_formula=qs_formula,
            aggregation=aggregation,
            time_start=history_time_start,
            time_end=buoy_time_end,
            K=K,
            historical_folder=hist_folder,
            partitions_folder=part_folder,
            partition_ids=pids,
            include_bulk=include_historical_bulk,
            include_partitions_sum=include_partitions_sum if include_partitions else False,
            include_binwaves_reference=include_binwaves_reference,
            binwaves_reference_folder=binwaves_reference_folder,
            wave_direction=wave_direction,
            project_root=project_root,
        )
        series.update(hist_series)

        if (
            kma_cluster_cases_root is not None
            and not _uses_kma_merged_grids_folder(hist_folder, project_root)
        ):
            kma_label = (
                kcs.BINWAVES_PLUS_KMA_LABEL
                if combine_kma_with_binwaves
                else "KMA cluster SWAN"
            )
            qs_kma, info_kma = load_kma_combined_qs(
                coord,
                shore_deg,
                qs_formula=qs_formula,
                aggregation=aggregation,
                time_start=history_time_start,
                time_end=buoy_time_end,
                K=K,
                cluster_cases_root=kma_cluster_cases_root,
                kma_bmu_csv=kma_bmu_csv,
                combine_with_binwaves=combine_kma_with_binwaves,
                historical_folder=hist_folder,
                project_root=project_root,
            )
            if qs_kma is not None:
                series[kma_label] = (qs_kma, info_kma)

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
            qs_formula=qs_formula,
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

        bulk_series_folders: dict[str, Path] = {bulk_label: hist_folder}
        if BINWAVES_BULK_LABEL in series and binwaves_ref_folder is not None:
            bulk_series_folders[BINWAVES_BULK_LABEL] = binwaves_ref_folder
        scenarios_cfg = _scenarios_cfg_for_qs_plot(
            series,
            gcm_scenarios=gcm_scenarios,
            historical_folder=hist_folder,
            project_root=project_root,
            bulk_series_folders=bulk_series_folders,
        )
        kma_labels = {kcs.BINWAVES_PLUS_KMA_LABEL, "KMA cluster SWAN"}
        scenarios_cfg = [
            (
                label,
                {
                    **cfg,
                    "color": "turquoise",
                    "lw": 1.1,
                    "alpha": 0.9,
                    "zorder": 5,
                    "ls": "-",
                },
            )
            if label in kma_labels
            else (label, cfg)
            for label, cfg in scenarios_cfg
        ]

        hindcast_note = (
            f"{bulk_label} & {BINWAVES_BULK_LABEL}"
            if BINWAVES_BULK_LABEL in series
            else bulk_label
        )
        title = (
            f"{formula_note} ({aggregation}) — buoy {match.buoy_id} vs {hindcast_note} "
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
                if cumulative_reference_start is not None:
                    hist_key = next(
                        (k for k in series if _is_historical_bulk_series_key(k)),
                        None,
                    )
                    if hist_key is not None:
                        try:
                            hist_bulk_da = series[hist_key][0]
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


def plot_partition_hs_tp_dp_timeseries(
    coordinates: Sequence[float] | Sequence[Sequence[float]],
    *,
    qs_formula: QsFormula = "cerc",
    partition_ids: Sequence[int] | None = None,
    time_start=None,
    time_end=None,
    aggregation: str = "daily",
    include_bulk: bool = True,
    include_buoy: bool = False,
    overlay_buoy_on_bulk: bool = True,
    buoy_id: str | None = None,
    buoys: Mapping[str, Any] | None = None,
    buoy_data_dir: str | Path = DEFAULT_BUOY_DATA_DIR,
    buoy_max_distance_km: float | None = 75.0,
    layout: Literal["grid", "overlay"] = "grid",
    grid_orientation: Literal["rows_are_partitions", "rows_are_variables"] = "rows_are_variables",
    include_sediment_transport_row: bool = False,
    shoreline_orientation_deg: float | Sequence[float] | Mapping[tuple[float, float], float] | None = None,
    K: float = DEFAULT_LONGSHORE_K,
    static: bool = True,
    interactive: bool = False,
    bulk_axis_scope: Literal["per_figure", "all_figures"] = "all_figures",
    time_highlights: Sequence[Any] | None = None,
    swan_timeseries_path: str | Path | None = None,
    swan_timeseries_folder: str | Path | None = None,
    overlay_swan_on_bulk: bool = True,
    kma_cluster_cases_root: str | Path | None = None,
    kma_bmu_csv: str | Path | None = None,
    combine_kma_with_binwaves: bool = True,
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
    Partition Hs / Tp / Dp grid (and optional Qs row) at hindcast coordinates.

    ``qs_formula``: ``'cerc'`` or ``'deep'`` for the optional Qs row.

    ``include_buoy=True``: load NDBC buoy when a pickle exists.

    ``overlay_buoy_on_bulk=True`` (default): hindcast bulk + buoy on the **same**
    column (black = hindcast, red = buoy). Set ``False`` for a separate buoy column.

    ``time_highlights``: optional shaded vertical bands for one or more time windows
    (see ``gcm_comparison.plot_partition_hs_tp_dp_timeseries``).

    ``swan_timeseries_path``: optional SWAN point bulk NetCDF (``Hsig``/``Tps``/``pdir``).
    Overlaid on the hindcast bulk column by default (``overlay_swan_on_bulk=True``).

    ``kma_cluster_cases_root`` + ``kma_bmu_csv``: reconstruct hourly SWAN bulk from
    static KMA cluster cases (``CASES_ONLY_WIND/000`` … ``149``), mapping each hour to
  its BMU. When ``combine_kma_with_binwaves=True`` (default), builds
    **BinWaves + KMA clusters** and overlays it on the bulk column.
    """
    import importlib
    from utils import gcm_comparison as gc
    from utils import kma_cluster_swan as kcs

    importlib.reload(gc)
    importlib.reload(kcs)

    if kma_cluster_cases_root is not None and not _uses_kma_merged_grids_folder(
        historical_folder, project_root
    ):
        coord = coordinates[0] if coordinates else None
        if coord is None:
            raise ValueError("kma_cluster_cases_root requires at least one coordinate")
        root = project_root or kcs._default_project_root()
        bmu_csv = kma_bmu_csv or kcs.DEFAULT_KMA_BMU_CSV
        hist_for_kma = historical_folder
        if combine_kma_with_binwaves and hist_for_kma is None:
            hist_for_kma = gc.resolve_historical_folder(
                historical_folder=historical_folder,
                historical_dataset=historical_dataset,
                project_root=root,
            )
        if combine_kma_with_binwaves:
            swan_timeseries_path = kcs.build_binwaves_plus_kma_swan_netcdf(
                coord,
                time_start=time_start,
                time_end=time_end,
                cluster_cases_root=kma_cluster_cases_root,
                kma_bmu_csv=bmu_csv,
                historical_folder=hist_for_kma,
                project_root=root,
            )
            swan_bulk_label = swan_bulk_label or kcs.BINWAVES_PLUS_KMA_LABEL
        else:
            df, _info = kcs.build_kma_cluster_swan_dataframe(
                coord,
                time_start=time_start,
                time_end=time_end,
                cluster_cases_root=kma_cluster_cases_root,
                kma_bmu_csv=bmu_csv,
                combine_with_binwaves=False,
                project_root=root,
            )
            lat, lon, _, _ = kcs._normalize_coordinate(coord)
            out_dir = gc.resolve_path(kcs.DEFAULT_OUTPUT_DIR, root)
            t0 = pd.Timestamp(time_start).strftime("%Y%m%d%H")
            t1 = pd.Timestamp(time_end).strftime("%Y%m%d%H")
            swan_timeseries_path = kcs.write_swan_point_netcdf(
                df,
                out_dir / kcs._swan_point_filename(lat, lon, suffix=f"_windonly_{t0}_{t1}"),
                lat=lat,
                lon=lon,
            )
            swan_bulk_label = swan_bulk_label or "KMA cluster SWAN"

    plot_root = project_root or (kcs._default_project_root() if kma_cluster_cases_root else None)

    return gc.plot_partition_hs_tp_dp_timeseries(
        coordinates,
        qs_formula=qs_formula,
        partition_ids=partition_ids,
        time_start=time_start,
        time_end=time_end,
        aggregation=aggregation,
        include_bulk=include_bulk,
        include_buoy=include_buoy,
        overlay_buoy_on_bulk=overlay_buoy_on_bulk,
        buoy_id=buoy_id,
        buoys=buoys,
        buoy_data_dir=buoy_data_dir,
        buoy_max_distance_km=buoy_max_distance_km,
        layout=layout,
        grid_orientation=grid_orientation,
        include_sediment_transport_row=include_sediment_transport_row,
        shoreline_orientation_deg=shoreline_orientation_deg,
        K=K,
        static=static,
        interactive=interactive,
        bulk_axis_scope=bulk_axis_scope,
        time_highlights=time_highlights,
        swan_timeseries_path=swan_timeseries_path,
        swan_timeseries_folder=swan_timeseries_folder,
        overlay_swan_on_bulk=overlay_swan_on_bulk,
        swan_bulk_label=swan_bulk_label,
        partitions_folder=partitions_folder,
        partitions_dataset=partitions_dataset,
        historical_folder=historical_folder,
        historical_dataset=historical_dataset,
        include_binwaves_reference=include_binwaves_reference,
        binwaves_reference_folder=binwaves_reference_folder,
        project_root=plot_root,
    )


def plot_buoy_bulk_validation_scatter(*args, **kwargs):
    """Re-export from ``utils.gcm_comparison``."""
    return _gcm_comparison.plot_buoy_bulk_validation_scatter(*args, **kwargs)