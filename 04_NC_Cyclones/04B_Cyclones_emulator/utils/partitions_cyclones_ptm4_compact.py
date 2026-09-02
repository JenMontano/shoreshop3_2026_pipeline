"""Compact PTM4 cyclone partition workflow.

Goals:
- Keep cyclone-by-cyclone methodology.
- Reduce file explosion by saving all variables in one file per grid+cyclone.
- Reuse efficient crop/merge strategy (isobath filtering + smooth grid merge).
"""

from __future__ import annotations

import gc
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from postprocessing_binwaves_bmus import (
    CIRCULAR_BLEND_VARS,
    _polygon_from_sites,
    _smooth_grid_weight,
)

DEFAULT_GRID_NAMES = ("grid1", "grid2", "grid3", "grid4")
GRID_TO_POINT = {"grid1": "point_1", "grid2": "point_2", "grid3": "point_3", "grid4": "point_4"}

# Direction-like partition vars for cyclone compact files
CYCLONE_CIRCULAR_VARS = frozenset(set(CIRCULAR_BLEND_VARS) | {"pdp0", "pdp1", "dp", "dm"})


@dataclass(frozen=True)
class SmoothMergePlan:
    """Precomputed site union + blend weights shared by all cyclones."""

    lon: np.ndarray  # (n_merged,)
    lat: np.ndarray  # (n_merged,)
    weights: np.ndarray  # (n_grids, n_merged)
    site_idx: np.ndarray  # (n_grids, n_merged) int; -1 = unused
    grid_names: tuple[str, ...]


def _safe_save_netcdf(ds_out: xr.Dataset, output_file: Path) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_file.parent / f".{output_file.name}.tmp"
    tmp.unlink(missing_ok=True)

    encoding = {}
    for name in ds_out.data_vars:
        var = ds_out[name]
        if np.issubdtype(var.dtype, np.floating):
            if var.dtype != np.float32:
                ds_out[name] = var.astype(np.float32)
            encoding[name] = {"dtype": "float32", "zlib": True, "complevel": 1, "shuffle": True}

    attempts = [
        {"engine": None, "format": None, "encoding": encoding},
        {"engine": "netcdf4", "format": None, "encoding": {}},
        {"engine": "h5netcdf", "format": None, "encoding": {}},
        {"engine": "scipy", "format": "NETCDF3_64BIT", "encoding": {}},
    ]
    last_error: Exception | None = None
    for attempt in attempts:
        tmp.unlink(missing_ok=True)
        try:
            ds_out.to_netcdf(tmp, engine=attempt["engine"], format=attempt["format"], encoding=attempt["encoding"])
            last_error = None
            break
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Failed writing {output_file}: {last_error}") from last_error

    if tmp.stat().st_size < 10_000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Suspiciously small output: {tmp}")
    output_file.unlink(missing_ok=True)
    tmp.replace(output_file)


def _clean_output_da(da: xr.DataArray) -> xr.DataArray:
    """Drop non-dimension coords (e.g. scalar 'part') before dataset assembly."""
    # Keep only dimension coordinates to avoid MergeError on conflicting scalar coords.
    return da.reset_coords(drop=True)


def _get_depth_fn(gebco: xr.Dataset):
    depth_var = None
    for var in gebco.data_vars:
        v = var.lower()
        if "elevation" in v or "depth" in v or "bathymetry" in v or v == "z":
            depth_var = var
            break
    if depth_var is None:
        depth_var = list(gebco.data_vars)[0]

    lon_name = next((c for c in gebco.coords if "lon" in c.lower() or c == "x"), None)
    lat_name = next((c for c in gebco.coords if "lat" in c.lower() or c == "y"), None)
    if lon_name is None or lat_name is None:
        raise ValueError("Could not find lon/lat coordinates in GEBCO file")

    depth = gebco[depth_var]
    if float(depth.min()) < 0:
        depth = np.abs(depth)

    def depth_at(lon_deg: float, lat_deg: float) -> float:
        return float(depth.sel({lon_name: lon_deg, lat_name: lat_deg}, method="nearest").values)

    return depth_at


def _site_lon_lat(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    lon = None
    lat = None
    if "coord_x" in ds.coords and "coord_y" in ds.coords:
        lon = np.asarray(ds["coord_x"].values).reshape(-1)
        lat = np.asarray(ds["coord_y"].values).reshape(-1)
    elif "lon" in ds.coords and "lat" in ds.coords:
        lon = np.asarray(ds["lon"].values).reshape(-1)
        lat = np.asarray(ds["lat"].values).reshape(-1)
    elif "lon" in ds.data_vars and "lat" in ds.data_vars:
        lon = np.asarray(ds["lon"].values).reshape(-1)
        lat = np.asarray(ds["lat"].values).reshape(-1)
    if lon is None or lat is None or len(lon) != len(lat):
        raise ValueError("Could not read per-site lon/lat from dataset")
    return lon, lat


def _prepare_isobath_tree(points_geojson_file: Path):
    import geopandas as gpd

    gdf = gpd.read_file(points_geojson_file)
    pts = gdf[gdf.geometry.geom_type == "Point"].copy()
    if pts.empty:
        raise ValueError(f"No Point geometries in {points_geojson_file}")
    keep_xy = np.column_stack([pts.geometry.x.values, pts.geometry.y.values])
    return keep_xy, cKDTree(keep_xy), {(float(x), float(y)) for x, y in keep_xy}


def _mask_sites_to_points(lon: np.ndarray, lat: np.ndarray, keep_tree: cKDTree, keep_set: set[tuple[float, float]], tol: float):
    site_mask = np.array([(float(x), float(y)) in keep_set for x, y in zip(lon, lat)], dtype=bool)
    if not np.all(site_mask):
        candidate_idx = np.where(~site_mask)[0]
        candidate_xy = np.column_stack([lon[candidate_idx], lat[candidate_idx]])
        distances, _ = keep_tree.query(candidate_xy, distance_upper_bound=tol)
        site_mask[candidate_idx[np.isfinite(distances)]] = True
    return site_mask


def _wind_for_cyclone(wind_ds: xr.Dataset, grid_name: str, cyclone_id: int, n_steps: int, time_coord) -> tuple[xr.DataArray, xr.DataArray]:
    if "Windv_x" not in wind_ds.data_vars or "Windv_y" not in wind_ds.data_vars:
        raise ValueError("Wind dataset must include Windv_x and Windv_y")
    point_dim = "point" if "point" in wind_ds.dims else next((d for d in wind_ds.dims if "point" in d.lower()), None)
    if point_dim is None:
        raise ValueError("Wind dataset must include a point-like dimension")

    point_name = GRID_TO_POINT[grid_name]
    pt_idx = int(point_name.split("_")[-1]) - 1

    try:
        vx = wind_ds["Windv_x"].sel(**{point_dim: point_name}, method="nearest").values
        vy = wind_ds["Windv_y"].sel(**{point_dim: point_name}, method="nearest").values
    except Exception:
        vx = wind_ds["Windv_x"].isel(**{point_dim: pt_idx}).values
        vy = wind_ds["Windv_y"].isel(**{point_dim: pt_idx}).values

    vx = np.asarray(vx)
    vy = np.asarray(vy)
    if vx.ndim >= 2 and cyclone_id < vx.shape[0]:
        u = np.asarray(vx[cyclone_id]).reshape(-1)
        v = np.asarray(vy[cyclone_id]).reshape(-1)
    else:
        u = vx.reshape(-1)
        v = vy.reshape(-1)
        start = cyclone_id * n_steps
        end = start + n_steps
        if end <= len(u):
            u = u[start:end]
            v = v[start:end]
        else:
            u = u[:n_steps]
            v = v[:n_steps]

    if len(u) < n_steps:
        pad = n_steps - len(u)
        u = np.pad(u, (0, pad), mode="edge")
        v = np.pad(v, (0, pad), mode="edge")
    else:
        u = u[:n_steps]
        v = v[:n_steps]

    wspd = np.sqrt(u**2 + v**2)
    wdir = (270 - np.degrees(np.arctan2(v, u))) % 360.0
    wspd_da = xr.DataArray(wspd, dims=["time"], coords={"time": time_coord})
    wdir_da = xr.DataArray(wdir, dims=["time"], coords={"time": time_coord})
    return wspd_da, wdir_da


def discover_reconstructed_cyclones(project_root: Path, grid_name: str) -> list[tuple[int, Path]]:
    """Find reconstructed spectra files for one grid sorted by cyclone id."""
    project_root = Path(project_root)
    rx = re.compile(rf"^reconstructed_spectra_{re.escape(grid_name)}_cyclone_(\d+)\.nc$")
    files = []
    for p in (project_root / grid_name / "outputs").glob(f"reconstructed_spectra_{grid_name}_cyclone_*.nc"):
        m = rx.match(p.name)
        if m:
            files.append((int(m.group(1)), p))
    return sorted(files, key=lambda x: x[0])


def build_compact_ptm4_partitions(
    project_root: Path,
    output_dir: Path | None = None,
    grid_names: Sequence[str] = DEFAULT_GRID_NAMES,
    *,
    wind_data_path: Path | None = None,
    gebco_file: Path | None = None,
    isobath_geojson: Path | None = None,
    cyclone_start_id: int | None = None,
    cyclone_end_id: int | None = None,
    max_cyclones_per_grid: int | None = None,
    point_match_tolerance: float = 1e-6,
    skip_existing: bool = True,
) -> list[Path]:
    """Create one compact PTM4 file per grid+cyclone with all variables."""
    project_root = Path(project_root)
    output_dir = Path(output_dir or project_root / "outputs" / "partitions_cyclones_compact_ptm4")

    import sys
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from paths import GEBCO_FILE, ISOBATH_GEOJSON, predicted_emu_tracks

    wind_data_path = Path(wind_data_path or predicted_emu_tracks("ssp245"))
    gebco_file = Path(gebco_file or GEBCO_FILE)
    isobath_geojson = Path(isobath_geojson or ISOBATH_GEOJSON)

    output_dir.mkdir(parents=True, exist_ok=True)
    keep_xy, keep_tree, keep_set = _prepare_isobath_tree(isobath_geojson)
    print(f"Isobath points loaded: {len(keep_xy)}")

    wind_ds = xr.open_dataset(wind_data_path)
    gebco = xr.open_dataset(gebco_file)
    depth_at = _get_depth_fn(gebco)

    saved: list[Path] = []
    try:
        for grid_name in grid_names:
            rec_files = discover_reconstructed_cyclones(project_root, grid_name)
            if cyclone_start_id is not None:
                rec_files = [it for it in rec_files if it[0] >= cyclone_start_id]
            if cyclone_end_id is not None:
                rec_files = [it for it in rec_files if it[0] <= cyclone_end_id]
            if max_cyclones_per_grid is not None:
                rec_files = rec_files[:max_cyclones_per_grid]

            if not rec_files:
                print(f"[{grid_name}] No reconstructed cyclone files found.")
                continue
            print(f"[{grid_name}] Processing {len(rec_files)} cyclone files")

            cached_keep_idx = None
            cached_depth = None
            cached_lon = None
            cached_lat = None

            for cid, src in tqdm(rec_files, desc=f"{grid_name}: PTM4 compact"):
                out = output_dir / f"cyclone_{cid}_{grid_name}_allvars_ptm4.nc"
                if skip_existing and out.is_file():
                    saved.append(out)
                    continue

                with xr.open_dataset(src) as ds:
                    efth_var = "efth" if "efth" in ds.data_vars else list(ds.data_vars)[0]
                    spectra = ds.rename({efth_var: "efth"})
                    if "cyclone_id" in spectra.dims and spectra.sizes["cyclone_id"] == 1:
                        spectra = spectra.isel(cyclone_id=0, drop=True)

                    lon, lat = _site_lon_lat(spectra)
                    if cached_keep_idx is None:
                        mask = _mask_sites_to_points(lon, lat, keep_tree, keep_set, point_match_tolerance)
                        keep_idx = np.where(mask)[0]
                        if keep_idx.size == 0:
                            raise ValueError(f"No isobath-matching sites for {grid_name}; increase tolerance")
                        cached_keep_idx = keep_idx
                        cached_lon = lon[keep_idx].astype(np.float32)
                        cached_lat = lat[keep_idx].astype(np.float32)
                        cached_depth = np.array([depth_at(float(lo), float(la)) for lo, la in zip(cached_lon, cached_lat)], dtype=np.float32)
                        print(f"[{grid_name}] keeping {len(keep_idx)}/{len(lon)} sites")

                    spectra = spectra.isel(site=cached_keep_idx)
                    n_time = spectra.sizes["time"]
                    n_site = spectra.sizes["site"]

                    dpt = xr.DataArray(
                        np.broadcast_to(cached_depth[np.newaxis, :], (n_time, n_site)),
                        dims=["time", "site"],
                        coords={"time": spectra.time, "site": spectra.site},
                        name="dpt",
                    )
                    spectra = spectra.assign(dpt=dpt)

                    wspd, wdir = _wind_for_cyclone(wind_ds, grid_name, cid, n_time, spectra.time)
                    wspd_exp = xr.DataArray(
                        np.broadcast_to(wspd.values[:, np.newaxis], (n_time, n_site)),
                        dims=["time", "site"],
                        coords={"time": spectra.time, "site": spectra.site},
                    )
                    wdir_exp = xr.DataArray(
                        np.broadcast_to(wdir.values[:, np.newaxis], (n_time, n_site)),
                        dims=["time", "site"],
                        coords={"time": spectra.time, "site": spectra.site},
                    )

                    spec_full = spectra.spec
                    hs = spec_full.hs()
                    tp = spec_full.tp()
                    tm02 = spec_full.tm02()
                    dp = spec_full.dpm()
                    dm = spec_full.dm()

                    dspart = spectra.spec.partition.ptm4(wspd_exp, wdir_exp, spectra.dpt)
                    if hasattr(dspart, "chunks") and dspart.chunks:
                        dspart = dspart.load()
                    n_parts = int(min(2, dspart.sizes.get("part", 2)))

                    out_vars = {
                        "hs": _clean_output_da(hs),
                        "tp": _clean_output_da(tp),
                        "tm02": _clean_output_da(tm02),
                        "dp": _clean_output_da(dp),
                        "dm": _clean_output_da(dm),
                    }
                    for p in range(n_parts):
                        part_ds = xr.Dataset({"efth": dspart.isel(part=p)})
                        part_spec = part_ds.spec
                        out_vars[f"phs{p}"] = _clean_output_da(part_spec.hs())
                        out_vars[f"ptp{p}"] = _clean_output_da(part_spec.tp())
                        out_vars[f"pdp{p}"] = _clean_output_da(part_spec.dpm())
                        out_vars[f"spr{p}"] = _clean_output_da(part_spec.dspr())

                    ds_out = xr.Dataset(out_vars)
                    ds_out = ds_out.assign_coords(
                        lon=("site", cached_lon),
                        lat=("site", cached_lat),
                        cyclone_id=cid,
                    )
                    ds_out.attrs["grid"] = grid_name
                    ds_out.attrs["partition_method"] = "ptm4"
                    ds_out.attrs["source_file"] = str(src)

                    _safe_save_netcdf(ds_out, out)
                    saved.append(out)

                gc.collect()
    finally:
        wind_ds.close()
        gebco.close()
        gc.collect()

    return saved


def discover_compact_cyclone_ids(compact_dir: Path, grid_names: Sequence[str] = DEFAULT_GRID_NAMES, *, intersection: bool = True) -> list[int]:
    """Discover cyclone IDs in compact per-grid files."""
    compact_dir = Path(compact_dir)
    rx = re.compile(r"^cyclone_(\d+)_(grid\d+)_allvars_ptm4\.nc$")
    ids_by_grid: dict[str, set[int]] = {g: set() for g in grid_names}
    for nc in compact_dir.glob("cyclone_*_grid*_allvars_ptm4.nc"):
        m = rx.match(nc.name)
        if not m:
            continue
        cid = int(m.group(1))
        grid = m.group(2)
        if grid in ids_by_grid:
            ids_by_grid[grid].add(cid)
    sets = [ids_by_grid[g] for g in grid_names if ids_by_grid[g]]
    if not sets:
        return []
    ids = set.intersection(*sets) if intersection else set.union(*sets)
    return sorted(ids)


def build_smooth_merge_plan(
    grid_lons: Sequence[np.ndarray],
    grid_lats: Sequence[np.ndarray],
    grid_names: Sequence[str],
    *,
    steepness: float = 2.0,
    blend_buffer_km: float = 30.0,
    tolerance_deg: float = 0.001,
) -> SmoothMergePlan:
    """Build site-union + soft weights once (geometry is identical across cyclones)."""
    from collections import defaultdict

    from shapely.geometry import Point

    n_grids = len(grid_lons)
    polygons = [_polygon_from_sites(np.asarray(lon), np.asarray(lat)) for lon, lat in zip(grid_lons, grid_lats)]
    trees = [
        cKDTree(np.column_stack((np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))))
        for lon, lat in zip(grid_lons, grid_lats)
    ]

    coord_to_grids: dict[tuple[float, float], list[tuple[int, int]]] = defaultdict(list)
    all_coords: list[tuple[float, float]] = []
    # Keep a flat XY array for O(n) nearest-existing checks without rebuilding trees
    existing_xy = np.empty((0, 2), dtype=np.float64)

    for grid_idx, (lon, lat) in enumerate(zip(grid_lons, grid_lats)):
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        for site_idx, (lo, la) in enumerate(zip(lon, lat)):
            if existing_xy.size:
                dist = np.hypot(existing_xy[:, 0] - lo, existing_xy[:, 1] - la)
                j = int(np.argmin(dist))
                if dist[j] < tolerance_deg:
                    coord_to_grids[all_coords[j]].append((grid_idx, site_idx))
                    continue
            coord_key = (round(float(lo), 6), round(float(la), 6))
            coord_to_grids[coord_key].append((grid_idx, site_idx))
            all_coords.append(coord_key)
            existing_xy = np.vstack([existing_xy, [[float(lo), float(la)]]])

    n_merged = len(all_coords)
    weights = np.zeros((n_grids, n_merged), dtype=np.float64)
    site_idx = np.full((n_grids, n_merged), -1, dtype=np.int32)
    lon_out = np.empty(n_merged, dtype=np.float64)
    lat_out = np.empty(n_merged, dtype=np.float64)

    for m, (lo, la) in enumerate(tqdm(all_coords, desc="Build merge plan", leave=False)):
        lon_out[m] = lo
        lat_out[m] = la
        point = Point(lo, la)
        owners = {g: s for g, s in coord_to_grids[(lo, la)]}
        any_w = False
        for g in range(n_grids):
            w = _smooth_grid_weight(point, polygons[g], steepness=steepness, blend_buffer_km=blend_buffer_km)
            if w <= 0.0:
                continue
            if g in owners:
                s = owners[g]
            else:
                _, s = trees[g].query([lo, la])
                s = int(s)
            weights[g, m] = w
            site_idx[g, m] = s
            any_w = True
        if not any_w:
            g0, s0 = coord_to_grids[(lo, la)][0]
            weights[g0, m] = 1.0
            site_idx[g0, m] = s0

    return SmoothMergePlan(
        lon=lon_out.astype(np.float32),
        lat=lat_out.astype(np.float32),
        weights=weights,
        site_idx=site_idx,
        grid_names=tuple(str(g) for g in grid_names),
    )


def _blend_with_plan(
    arrays: Sequence[np.ndarray],
    plan: SmoothMergePlan,
    *,
    circular: bool,
) -> np.ndarray:
    """Apply precomputed weights to per-grid (time, site) arrays -> (time, n_merged)."""
    n_grids, n_merged = plan.weights.shape
    n_time = int(arrays[0].shape[0])
    stacked = np.full((n_grids, n_time, n_merged), np.nan, dtype=np.float64)
    for g, arr in enumerate(arrays):
        idx = plan.site_idx[g]
        valid = idx >= 0
        if not np.any(valid):
            continue
        stacked[g][:, valid] = np.asarray(arr, dtype=np.float64)[:, idx[valid]]

    w = plan.weights[:, None, :]  # (g, 1, m)
    finite = np.isfinite(stacked)
    w_eff = np.where(finite & (w > 0.0), w, 0.0)
    w_sum = w_eff.sum(axis=0)
    ok = w_sum > 0.0

    if circular:
        rad = np.deg2rad(np.nan_to_num(stacked, nan=0.0))
        sin_sum = (w_eff * np.sin(rad)).sum(axis=0)
        cos_sum = (w_eff * np.cos(rad)).sum(axis=0)
        out = np.rad2deg(np.arctan2(sin_sum, cos_sum)) % 360.0
    else:
        out = (w_eff * np.nan_to_num(stacked, nan=0.0)).sum(axis=0) / np.maximum(w_sum, 1e-12)

    return np.where(ok, out, np.nan).astype(np.float32)


def _merge_one_cyclone_fast(
    cid: int,
    compact_dir: str,
    output_dir: str,
    grid_names: tuple[str, ...],
    plan_payload: dict,
    skip_existing: bool,
) -> str | None:
    """Worker-friendly merge of one cyclone using a pickled plan payload."""
    compact_dir_p = Path(compact_dir)
    output_dir_p = Path(output_dir)
    out = output_dir_p / f"cyclone_{cid}_merged_allvars_ptm4.nc"
    if skip_existing and out.is_file() and out.stat().st_size > 100_000:
        # Heuristic: old empty files were ~81kB; good files are ~1MB
        return str(out)

    plan = SmoothMergePlan(
        lon=plan_payload["lon"],
        lat=plan_payload["lat"],
        weights=plan_payload["weights"],
        site_idx=plan_payload["site_idx"],
        grid_names=tuple(plan_payload["grid_names"]),
    )

    grid_paths = [compact_dir_p / f"cyclone_{cid}_{g}_allvars_ptm4.nc" for g in grid_names]
    grid_paths = [p for p in grid_paths if p.is_file()]
    if len(grid_paths) != len(grid_names):
        # Require all grids used to build the plan
        return None

    datasets = [xr.open_dataset(p) for p in grid_paths]
    try:
        sample = datasets[0]
        variables = [v for v in sample.data_vars if v not in ("lon", "lat", "coord_x", "coord_y", "site_id")]
        time = np.asarray(sample.time.values)

        merged_vars = {}
        for var in variables:
            arrays = []
            for ds in datasets:
                if var not in ds:
                    arrays.append(np.full((len(time), ds.sizes["site"]), np.nan, dtype=np.float64))
                else:
                    arrays.append(np.asarray(ds[var].values))
            circular = var in CYCLONE_CIRCULAR_VARS
            merged_vars[var] = (("time", "site"), _blend_with_plan(arrays, plan, circular=circular))

        ds_out = xr.Dataset(
            merged_vars,
            coords={
                "time": time,
                "site": np.arange(plan.lon.size),
                "lon": ("site", plan.lon),
                "lat": ("site", plan.lat),
                "cyclone_id": cid,
            },
            attrs={
                "partition_method": "ptm4",
                "merge_method": "smooth_buffer_blend_fast",
                "grids": ", ".join(plan.grid_names),
            },
        )
        _safe_save_netcdf(ds_out, out)
        return str(out)
    finally:
        for ds in datasets:
            ds.close()
        gc.collect()


def merge_compact_cyclones_smooth(
    compact_dir: Path,
    output_dir: Path | None = None,
    grid_names: Sequence[str] = DEFAULT_GRID_NAMES,
    cyclone_ids: Sequence[int] | None = None,
    *,
    steepness: float = 2.0,
    blend_buffer_km: float = 30.0,
    tolerance_deg: float = 0.001,
    skip_existing: bool = True,
    n_jobs: int = 1,
) -> list[Path]:
    """Create one merged all-variable file per cyclone from compact per-grid files.

    Unlike BinWaves (one merge per variable for the whole archive), cyclones need
    one merge per storm. Site geometry is identical across storms, so soft-blend
    weights are built once and then applied with vectorized NumPy. Same blend math
    as ``merge_grids_smooth``, without recomputing site-by-site for every
    cyclone×variable.

    ``n_jobs>1`` uses processes (HDF5/netCDF is not thread-safe).
    """
    compact_dir = Path(compact_dir)
    output_dir = Path(output_dir or compact_dir.parent / "merged_cyclones_compact_ptm4")
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_names_t = tuple(grid_names)

    ids = discover_compact_cyclone_ids(compact_dir, grid_names_t, intersection=True)
    if cyclone_ids is not None:
        wanted = set(int(c) for c in cyclone_ids)
        ids = [c for c in ids if c in wanted]
    if not ids:
        print("No common cyclone IDs to merge")
        return []

    # Geometry from first cyclone with all grids present
    ref_cid = ids[0]
    ref_paths = [compact_dir / f"cyclone_{ref_cid}_{g}_allvars_ptm4.nc" for g in grid_names_t]
    if not all(p.is_file() for p in ref_paths):
        raise FileNotFoundError(f"Missing reference grid files for cyclone {ref_cid}")

    print(f"Building merge plan from cyclone {ref_cid} ({len(grid_names_t)} grids)...")
    ref_lons, ref_lats = [], []
    for p in ref_paths:
        with xr.open_dataset(p) as ds:
            ref_lons.append(np.asarray(ds.lon.values))
            ref_lats.append(np.asarray(ds.lat.values))
    plan = build_smooth_merge_plan(
        ref_lons,
        ref_lats,
        grid_names_t,
        steepness=steepness,
        blend_buffer_km=blend_buffer_km,
        tolerance_deg=tolerance_deg,
    )
    print(f"Merge plan ready: {plan.lon.size} sites")

    plan_payload = {
        "lon": plan.lon,
        "lat": plan.lat,
        "weights": plan.weights,
        "site_idx": plan.site_idx,
        "grid_names": list(plan.grid_names),
    }

    # Drop tiny/empty leftovers from the old buggy merge so skip_existing won't keep them
    if skip_existing:
        for cid in ids:
            out = output_dir / f"cyclone_{cid}_merged_allvars_ptm4.nc"
            if out.is_file() and out.stat().st_size < 100_000:
                out.unlink(missing_ok=True)

    merged_files: list[Path] = []
    todo = []
    for cid in ids:
        out = output_dir / f"cyclone_{cid}_merged_allvars_ptm4.nc"
        if skip_existing and out.is_file() and out.stat().st_size >= 100_000:
            merged_files.append(out)
        else:
            todo.append(cid)

    print(f"Merging {len(todo)} cyclones ({len(merged_files)} already done), n_jobs={n_jobs}")
    if not todo:
        return merged_files

    if n_jobs <= 1:
        for cid in tqdm(todo, desc="Merging cyclones"):
            path = _merge_one_cyclone_fast(
                cid,
                str(compact_dir),
                str(output_dir),
                grid_names_t,
                plan_payload,
                False,
            )
            if path:
                merged_files.append(Path(path))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futs = {
                pool.submit(
                    _merge_one_cyclone_fast,
                    cid,
                    str(compact_dir),
                    str(output_dir),
                    grid_names_t,
                    plan_payload,
                    False,
                ): cid
                for cid in todo
            }
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Merging cyclones"):
                path = fut.result()
                if path:
                    merged_files.append(Path(path))

    return sorted(merged_files, key=lambda p: p.name)
