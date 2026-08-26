"""Merge per-grid BinWaves+BMUS bulk NetCDFs and build webpage GeoJSON statistics."""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

SHORESHOP_UTILS = Path("/lustre/geocean/WORK/users/montanoj/personal/ShoreShop2026/utils")
if str(SHORESHOP_UTILS) not in sys.path:
    sys.path.insert(0, str(SHORESHOP_UTILS))

from outputs_grids import compute_distance_to_border, merge_multiple_grids  # noqa: E402

DEFAULT_PROJECT_ROOT = Path("/lustre/geocean/WORK/users/montanoj/personal/Wind_Metamodel")
DEFAULT_GRID_NAMES = ("grid1", "grid2", "grid3", "grid4")
BINWAVES_TAG = "BinWaves_BMUS"
PRIORITY_VARIABLES = ("hs", "tp", "dm", "dp", "tm02", "phs0", "ptp0", "dp0")
CIRCULAR_BLEND_VARS = frozenset({"dp", "dm", "dp0"})
WEBPAGE_VARS = ("hs", "tp", "dp", "dm")
WEBPAGE_OPTIONAL_VARS = ("tm02",)
MONTH_NAMES = (
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
)
SEASON_MONTHS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}


def grid_binwaves_bmus_file(project_root: Path, grid_name: str, var_name: str) -> Path:
    """Path to ``{var}_grid{N}_BinWaves_BMUS.nc`` for one grid."""
    return (
        project_root
        / grid_name
        / "outputs"
        / "BinWaves_BMUS"
        / f"{var_name}_{grid_name}_{BINWAVES_TAG}.nc"
    )


def grid_name_from_binwaves_file(path: Path) -> str:
    return grid_name_from_path(path)


def grid_name_from_path(path: Path) -> str:
    """Extract grid name from ``{var}_gridN_*.nc`` or ``{var}_gridN_points500m.nc``."""
    stem = path.stem.replace("_points500m", "").replace(f"_{BINWAVES_TAG}", "")
    return stem.rsplit("_", 1)[-1]


def cropped_500m_file(output_dir: Path, grid_name: str, var_name: str) -> Path:
    """Path to cropped 500 m file: ``{var}_gridN_points500m.nc``."""
    return output_dir / f"{var_name}_{grid_name}_points500m.nc"


def discover_cropped_500m_variables(
    cropped_dir: Path,
    grid_names: Sequence[str] = DEFAULT_GRID_NAMES,
) -> list[str]:
    """Variables present in the cropped 500 m directory."""
    cropped_dir = Path(cropped_dir)
    found: set[str] = set()
    for grid_name in grid_names:
        for nc in cropped_dir.glob(f"*_{grid_name}_points500m.nc"):
            var_name = nc.stem.replace(f"_{grid_name}_points500m", "")
            if var_name:
                found.add(var_name)
    priority = [v for v in PRIORITY_VARIABLES if v in found]
    other = sorted(v for v in found if v not in PRIORITY_VARIABLES)
    return priority + other


def crop_binwaves_bmus_to_points_geojson(
    project_root: Path = DEFAULT_PROJECT_ROOT,
    output_dir: Path | None = None,
    points_geojson_file: Path | str = "inputs/water_level_statistics.geojson",
    grid_names: Sequence[str] = DEFAULT_GRID_NAMES,
    variables: Sequence[str] = PRIORITY_VARIABLES,
    *,
    point_match_tolerance: float = 1e-6,
    skip_existing: bool = True,
    verbose: bool = True,
) -> list[Path]:
    """
    Crop per-grid BinWaves+BMUS NetCDFs to the 500 m reference points in a GeoJSON.

    Writes ``{var}_gridN_points500m.nc`` into ``outputs/cropped_500m_binwaves_bmus/``.
    """
    import geopandas as gpd
    from scipy.spatial import cKDTree

    project_root = Path(project_root)
    output_dir = Path(output_dir or project_root / "outputs" / "cropped_500m_binwaves_bmus")
    points_geojson_file = Path(points_geojson_file)
    if not points_geojson_file.is_absolute():
        points_geojson_file = project_root / points_geojson_file

    output_dir.mkdir(parents=True, exist_ok=True)
    if not points_geojson_file.is_file():
        raise FileNotFoundError(f"Points GeoJSON not found: {points_geojson_file}")

    keep_gdf = gpd.read_file(points_geojson_file)
    keep_points = keep_gdf[keep_gdf.geometry.geom_type == "Point"].copy()
    if keep_points.empty:
        raise ValueError(f"GeoJSON has no Point geometries: {points_geojson_file}")

    keep_xy = np.column_stack([keep_points.geometry.x.values, keep_points.geometry.y.values])
    keep_set = {(float(x), float(y)) for x, y in keep_xy}
    keep_tree = cKDTree(keep_xy)

    if verbose:
        print(f"Cropping BinWaves+BMUS to 500 m points")
        print(f"  Output: {output_dir}")
        print(f"  Points: {points_geojson_file} ({len(keep_xy)} sites)")
        print(f"  Variables: {list(variables)}")

    saved: list[Path] = []
    tasks = [
        (grid_name, var_name)
        for grid_name in grid_names
        for var_name in variables
    ]

    for grid_name, var_name in tqdm(tasks, desc="Cropping BinWaves+BMUS", disable=not verbose):
        src = grid_binwaves_bmus_file(project_root, grid_name, var_name)
        dst = cropped_500m_file(output_dir, grid_name, var_name)
        if skip_existing and dst.is_file():
            if verbose:
                print(f"  Skip {dst.name}")
            saved.append(dst)
            continue
        if not src.is_file():
            if verbose:
                print(f"  Missing source: {src}")
            continue

        with xr.open_dataset(src, chunks={"time": 1000}) as ds:
            lon = lat = None
            if "lon" in ds.coords or "lon" in ds.data_vars:
                lon = np.asarray(ds["lon"].values).flatten()
            elif "coord_x" in ds.coords or "coord_x" in ds.data_vars:
                lon = np.asarray(ds["coord_x"].values).flatten()
            if "lat" in ds.coords or "lat" in ds.data_vars:
                lat = np.asarray(ds["lat"].values).flatten()
            elif "coord_y" in ds.coords or "coord_y" in ds.data_vars:
                lat = np.asarray(ds["coord_y"].values).flatten()

            if lon is None or lat is None or len(lon) != len(lat):
                if verbose:
                    print(f"  Skipping {src.name}: no lon/lat")
                continue

            site_mask = np.array([(float(x), float(y)) in keep_set for x, y in zip(lon, lat)], dtype=bool)
            if not np.all(site_mask):
                candidate_idx = np.where(~site_mask)[0]
                candidate_xy = np.column_stack([lon[candidate_idx], lat[candidate_idx]])
                distances, _ = keep_tree.query(candidate_xy, distance_upper_bound=point_match_tolerance)
                site_mask[candidate_idx[np.isfinite(distances)]] = True

            n_kept = int(site_mask.sum())
            if n_kept == 0:
                if verbose:
                    print(f"  No matching sites for {src.name}")
                continue

            subset = ds.isel(site=site_mask)
            if "coord_x" in subset.variables and "lon" not in subset.variables:
                subset = subset.rename({"coord_x": "lon", "coord_y": "lat"})
            subset = subset.drop_vars(
                [v for v in ("coord_x", "coord_y", "site_id") if v in subset.variables],
                errors="ignore",
            )
            subset.attrs = {}

            if verbose:
                print(f"  {src.name}: keeping {n_kept}/{len(site_mask)} sites -> {dst.name}")
            subset.to_netcdf(dst)
            saved.append(dst)

    return saved


def merge_all_binwaves_bmus_smooth_from_cropped(
    cropped_dir: Path,
    output_dir: Path | None = None,
    grid_names: Sequence[str] = DEFAULT_GRID_NAMES,
    variables: Sequence[str] | None = None,
    *,
    steepness: float = 2.0,
    blend_buffer_km: float = 30.0,
    tolerance_deg: float = 0.001,
    skip_existing: bool = True,
    output_suffix: str = "_500m",
) -> dict[str, Path]:
    """Smooth-merge cropped 500 m per-grid files into ``{var}{output_suffix}.nc``."""
    cropped_dir = Path(cropped_dir)
    output_dir = Path(output_dir or cropped_dir.parent / "merged_500m_binwaves_bmus")
    output_dir.mkdir(parents=True, exist_ok=True)

    if variables is None:
        variables = discover_cropped_500m_variables(cropped_dir, grid_names)
    if not variables:
        raise ValueError(f"No cropped 500 m variables found in {cropped_dir}")

    results: dict[str, Path] = {}
    for var_name in variables:
        final_path = output_dir / f"{var_name}{output_suffix}.nc"
        if skip_existing and final_path.is_file():
            print(f"Skip {var_name}: {final_path.name} exists")
            results[var_name] = final_path
            continue

        grid_files = []
        for grid_name in grid_names:
            p = cropped_500m_file(cropped_dir, grid_name, var_name)
            if p.is_file():
                grid_files.append(p)
            else:
                print(f"  Missing {grid_name}: {p.name}")

        if not grid_files:
            print(f"No cropped files for {var_name}; skipping")
            continue

        blend_kind = "circular" if var_name in CIRCULAR_BLEND_VARS else "linear"
        print(f"Smooth merge {var_name} ({blend_kind}) from {len(grid_files)} grid(s)...")
        merge_grids_smooth(
            grid_files,
            var_name=var_name,
            steepness=steepness,
            blend_buffer_km=blend_buffer_km,
            tolerance_deg=tolerance_deg,
            output_file=final_path,
        )
        results[var_name] = final_path
        print(f"  -> {final_path}")

    return results


def discover_binwaves_bmus_variables(
    project_root: Path,
    grid_names: Sequence[str] = DEFAULT_GRID_NAMES,
) -> list[str]:
    """Variables present in at least one grid's ``outputs/BinWaves_BMUS`` folder."""
    found: set[str] = set()
    for grid_name in grid_names:
        folder = project_root / grid_name / "outputs" / "BinWaves_BMUS"
        if not folder.is_dir():
            continue
        for nc in folder.glob(f"*_{grid_name}_{BINWAVES_TAG}.nc"):
            stem = nc.stem.replace(f"_{grid_name}_{BINWAVES_TAG}", "")
            found.add(stem)
    priority = [v for v in PRIORITY_VARIABLES if v in found]
    other = sorted(v for v in found if v not in PRIORITY_VARIABLES)
    return priority + other


def audit_grid_binwaves_bmus(
    project_root: Path,
    grid_names: Sequence[str] = DEFAULT_GRID_NAMES,
) -> pd.DataFrame:
    """Summary table: which variables exist per grid."""
    variables = discover_binwaves_bmus_variables(project_root, grid_names)
    rows = []
    for grid_name in grid_names:
        for var in variables:
            p = grid_binwaves_bmus_file(project_root, grid_name, var)
            rows.append(
                {
                    "grid": grid_name,
                    "variable": var,
                    "exists": p.is_file(),
                    "path": str(p),
                    "size_mb": round(p.stat().st_size / 1e6, 1) if p.is_file() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def validate_netcdf_file(file_path: Path, var_name: str | None = None) -> tuple[bool, str | None]:
    try:
        with xr.open_dataset(file_path) as ds:
            _ = ds.sizes
            if var_name and var_name in ds.data_vars:
                _ = ds[var_name].isel(site=0, time=0).values
        return True, None
    except Exception as exc:
        return False, str(exc)


def cleanup_corrupted_temp_files(directory: Path, var_name: str | None = None) -> int:
    deleted = 0
    for temp_file in directory.glob("temp_*.nc"):
        ok, _ = validate_netcdf_file(temp_file, var_name)
        if not ok:
            temp_file.unlink(missing_ok=True)
            deleted += 1
    return deleted


def _drop_merge_extras(ds: xr.Dataset) -> xr.Dataset:
    drop = [
        v
        for v in ds.data_vars
        if v in ("coord_x", "coord_y", "site_id") or v.startswith("weight_")
    ]
    if drop:
        ds = ds.drop_vars(drop)
    ds.attrs = {}
    return ds


def _save_merged_netcdf(current_merged: xr.Dataset, final_output_file: Path) -> None:
    encoding: dict = {}
    valid_keys = {"zlib", "complevel", "shuffle", "fletcher32", "contiguous", "chunksizes", "dtype", "_FillValue"}
    for name in current_merged.data_vars:
        var = current_merged[name]
        enc = dict(var.encoding) if getattr(var, "encoding", None) else {}
        if np.issubdtype(var.dtype, np.floating):
            if var.dtype != np.float32:
                current_merged[name] = var.astype(np.float32)
            # complevel=1 writes several times faster than 4 for a small size cost
            enc.update(dtype="float32", zlib=True, complevel=1, shuffle=True)
            if var.ndim == 2:
                enc["chunksizes"] = tuple(min(s, c) for s, c in zip(var.shape, (8760, 512)))
        encoding[name] = {k: v for k, v in enc.items() if k in valid_keys}
    for coord in current_merged.coords:
        enc = dict(current_merged[coord].encoding) if getattr(current_merged[coord], "encoding", None) else {}
        encoding.setdefault(coord, {k: v for k, v in enc.items() if k in valid_keys})

    temp_output = final_output_file.parent / f".{final_output_file.name}.tmp"
    temp_output.unlink(missing_ok=True)
    # On some Lustre/HDF5 setups, netCDF4 writes can intermittently fail with
    # "NetCDF: HDF error". Try HDF-compatible backends first, then optional
    # NetCDF3 fallback with explicit time encoding.
    scipy_encoding: dict = {}
    if "time" in current_merged.coords:
        time_vals = pd.to_datetime(current_merged["time"].values)
        t0 = pd.Timestamp(time_vals[0]).floor("s")
        secs = ((time_vals - t0) / np.timedelta64(1, "s")).astype(np.int64)
        if np.all((secs >= np.iinfo(np.int32).min) & (secs <= np.iinfo(np.int32).max)):
            scipy_encoding["time"] = {
                "dtype": "int32",
                "units": f"seconds since {t0.strftime('%Y-%m-%d %H:%M:%S')}",
                "calendar": "proleptic_gregorian",
            }

    write_attempts = [
        {"engine": None, "format": None, "encoding": encoding},
        {"engine": "netcdf4", "format": None, "encoding": {}},
        {"engine": "h5netcdf", "format": None, "encoding": {}},
        {"engine": "scipy", "format": "NETCDF3_64BIT", "encoding": scipy_encoding},
    ]
    last_error: Exception | None = None
    for attempt in write_attempts:
        temp_output.unlink(missing_ok=True)
        try:
            current_merged.to_netcdf(
                temp_output,
                engine=attempt["engine"],
                format=attempt["format"],
                encoding=attempt["encoding"],
            )
            last_error = None
            break
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        temp_output.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to write merged NetCDF {final_output_file}: {last_error}") from last_error

    if temp_output.stat().st_size < 100_000:
        temp_output.unlink(missing_ok=True)
        raise RuntimeError(f"Merged file suspiciously small: {temp_output}")
    final_output_file.unlink(missing_ok=True)
    temp_output.replace(final_output_file)


def merge_binwaves_bmus_variable(
    var_name: str,
    grid_files: list[Path],
    output_dir: Path,
    *,
    steepness: float = 10.0,
    tolerance_deg: float = 0.001,
) -> Path:
    """Sequentially merge one variable across grids (grid1+grid2 → +grid3 → +grid4)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    final_output_file = output_dir / f"{var_name}_merged_all.nc"
    if final_output_file.exists():
        return final_output_file

    with xr.open_dataset(grid_files[0]) as ds_sample:
        data_vars = [v for v in ds_sample.data_vars if v not in ("lon", "lat", "coord_x", "coord_y", "site_id")]
        actual_var = var_name if var_name in data_vars else data_vars[0]

    all_times: list[pd.Timestamp] = []
    for grid_file in grid_files:
        with xr.open_dataset(grid_file) as ds_temp:
            all_times.extend(pd.to_datetime(ds_temp.time.values).tolist())
    all_times_idx = pd.DatetimeIndex(sorted(set(all_times)))
    common_time = xr.DataArray(all_times_idx, dims=["time"], name="time")

    current_merged: xr.Dataset | None = None
    ds_grid = None
    ds_grid_aligned = None

    try:
        for merge_step, grid_file in enumerate(grid_files):
            grid_name = grid_name_from_binwaves_file(grid_file)
            if merge_step == 0:
                ds_grid = xr.open_dataset(grid_file)
                ds_grid_aligned = ds_grid.reindex(time=common_time, method=None)
                current_merged = _drop_merge_extras(ds_grid_aligned)
                ds_grid.close()
                ds_grid = None
                ds_grid_aligned = None
                gc.collect()
                continue

            temp_current = output_dir / f"temp_{var_name}_merged_step{merge_step}.nc"
            temp_next = output_dir / f"temp_{var_name}_{grid_name}_step{merge_step}.nc"

            ds_grid = xr.open_dataset(grid_file)
            ds_grid_aligned = ds_grid.reindex(time=common_time, method=None)
            current_merged.to_netcdf(temp_current)
            current_merged.close()
            current_merged = None
            ds_grid_aligned.to_netcdf(temp_next)
            ds_grid.close()
            ds_grid = None
            ds_grid_aligned = None
            gc.collect()

            current_merged = merge_multiple_grids(
                grid_files=[temp_current, temp_next],
                var_name=actual_var,
                steepness=steepness,
                tolerance_deg=tolerance_deg,
                output_file=None,
                use_quality_checks=False,
            )
            current_merged = _drop_merge_extras(current_merged)
            temp_current.unlink(missing_ok=True)
            temp_next.unlink(missing_ok=True)
            gc.collect()

        if current_merged is None:
            raise RuntimeError(f"No merged dataset produced for {var_name}")

        _save_merged_netcdf(current_merged, final_output_file)
        current_merged.close()
        return final_output_file
    finally:
        for obj in (current_merged, ds_grid, ds_grid_aligned):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        gc.collect()


def merge_all_binwaves_bmus(
    project_root: Path = DEFAULT_PROJECT_ROOT,
    output_dir: Path | None = None,
    grid_names: Sequence[str] = DEFAULT_GRID_NAMES,
    variables: Sequence[str] | None = None,
    *,
    steepness: float = 10.0,
    tolerance_deg: float = 0.001,
    skip_existing: bool = True,
) -> dict[str, Path]:
    """Merge all discovered variables; returns map var → output path."""
    project_root = Path(project_root)
    output_dir = Path(output_dir or project_root / "outputs" / "merged_grids_binwaves_bmus")
    output_dir.mkdir(parents=True, exist_ok=True)

    if variables is None:
        variables = discover_binwaves_bmus_variables(project_root, grid_names)
    if not variables:
        raise ValueError("No BinWaves_BMUS variables found under any grid.")

    cleanup_corrupted_temp_files(output_dir)
    results: dict[str, Path] = {}

    for var_name in variables:
        final_path = output_dir / f"{var_name}_merged_all.nc"
        if skip_existing and final_path.is_file():
            print(f"Skip {var_name}: {final_path.name} exists")
            results[var_name] = final_path
            continue

        grid_files = []
        for grid_name in grid_names:
            p = grid_binwaves_bmus_file(project_root, grid_name, var_name)
            if p.is_file():
                grid_files.append(p)
            else:
                print(f"  Missing {grid_name}: {p.name}")

        if not grid_files:
            print(f"No files for {var_name}; skipping")
            continue

        print(f"Merging {var_name} from {len(grid_files)} grid(s)...")
        results[var_name] = merge_binwaves_bmus_variable(
            var_name,
            grid_files,
            output_dir,
            steepness=steepness,
            tolerance_deg=tolerance_deg,
        )
        print(f"  -> {results[var_name]}")

    return results


def _circular_weighted_blend_deg(angles_deg: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted circular mean along axis 0 (one value per timestep)."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.maximum(w.sum(), 1e-12)
    rad = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    sin_sum = np.tensordot(w, np.sin(rad), axes=(0, 0))
    cos_sum = np.tensordot(w, np.cos(rad), axes=(0, 0))
    return np.rad2deg(np.arctan2(sin_sum, cos_sum)) % 360.0


def _linear_weighted_blend(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted arithmetic mean along axis 0 (one value per timestep)."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.maximum(w.sum(), 1e-12)
    return np.tensordot(w, np.asarray(values, dtype=np.float64), axes=(0, 0))


def _polygon_from_sites(lon: np.ndarray, lat: np.ndarray):
    from scipy.spatial import ConvexHull
    from shapely.geometry import Polygon

    pts = np.unique(np.column_stack((lon, lat)), axis=0)
    if len(pts) >= 3:
        hull = ConvexHull(pts)
        ring = [[float(v[0]), float(v[1])] for v in pts[hull.vertices]]
        ring.append(ring[0])
        return Polygon(ring)
    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())
    buf = 0.01
    return Polygon([
        [lon_min - buf, lat_min - buf],
        [lon_max + buf, lat_min - buf],
        [lon_max + buf, lat_max + buf],
        [lon_min - buf, lat_max + buf],
        [lon_min - buf, lat_min - buf],
    ])


def _smooth_grid_weight(
    point,
    polygon,
    *,
    steepness: float,
    blend_buffer_km: float,
) -> float:
    inside = polygon.contains(point) or polygon.touches(point)
    if inside:
        dist_km = compute_distance_to_border(point, polygon) * 111.0
        return float(1.0 / (1.0 + np.exp(-steepness * dist_km)))
    dist_km = polygon.distance(point) * 111.0
    if dist_km > blend_buffer_km:
        return 0.0
    t = dist_km / blend_buffer_km
    return float(0.5 * np.cos(0.5 * np.pi * t) ** 2)


def _common_time_index(grid_files: list[Path]) -> xr.DataArray:
    all_times: list[pd.Timestamp] = []
    for grid_file in grid_files:
        with xr.open_dataset(grid_file) as ds_temp:
            all_times.extend(pd.to_datetime(ds_temp.time.values).tolist())
    return xr.DataArray(pd.DatetimeIndex(sorted(set(all_times))), dims=["time"], name="time")


def merge_grids_smooth(
    grid_files: list[Path],
    var_name: str = "hs",
    *,
    steepness: float = 2.0,
    blend_buffer_km: float = 30.0,
    tolerance_deg: float = 0.001,
    output_file: Path | None = None,
) -> xr.Dataset:
    """Grid merge with soft sigmoid weights, buffer blending, and circular mean for directions."""
    from collections import defaultdict

    from scipy.spatial import cKDTree
    from shapely.geometry import Point

    grid_files = [Path(p) for p in grid_files]
    common_time = _common_time_index(grid_files)

    with xr.open_dataset(grid_files[0]) as ds_sample:
        data_vars = [v for v in ds_sample.data_vars if v not in ("lon", "lat", "coord_x", "coord_y", "site_id")]
        actual_var = var_name if var_name in data_vars else data_vars[0]

    circular = actual_var in CIRCULAR_BLEND_VARS
    datasets: list[xr.Dataset] = []
    polygons = []
    grid_names: list[str] = []
    grid_coords: list[tuple[np.ndarray, np.ndarray]] = []
    trees: list[cKDTree] = []

    for grid_file in grid_files:
        ds = xr.open_dataset(grid_file, chunks={"time": 10_000})
        ds = ds.reindex(time=common_time, method=None)
        datasets.append(ds)
        grid_names.append(grid_name_from_path(grid_file))
        lon = ds.lon.values.astype(np.float64)
        lat = ds.lat.values.astype(np.float64)
        grid_coords.append((lon, lat))
        polygons.append(_polygon_from_sites(lon, lat))
        trees.append(cKDTree(np.column_stack((lon, lat))))

    coord_to_grids: dict[tuple[float, float], list[tuple[int, int]]] = defaultdict(list)
    all_coords: list[tuple[float, float]] = []

    for grid_idx, (lon, lat) in enumerate(grid_coords):
        for site_idx, (lo, la) in enumerate(zip(lon, lat)):
            coord_key = (round(float(lo), 6), round(float(la), 6))
            found = False
            for existing in coord_to_grids:
                dist = np.hypot(existing[0] - lo, existing[1] - la)
                if dist < tolerance_deg:
                    coord_to_grids[existing].append((grid_idx, site_idx))
                    found = True
                    break
            if not found:
                coord_to_grids[coord_key].append((grid_idx, site_idx))
                all_coords.append(coord_key)

    merged_data: list[np.ndarray] = []
    merged_coords: list[list[float]] = []
    blend_label = "circular_mean_with_buffer" if circular else "linear_mean_with_buffer"

    for lon, lat in tqdm(all_coords, desc=f"  {actual_var}: smooth merge"):
        point = Point(lon, lat)
        series_list: list[np.ndarray] = []
        weight_list: list[float] = []

        for grid_idx in range(len(datasets)):
            w = _smooth_grid_weight(
                point,
                polygons[grid_idx],
                steepness=steepness,
                blend_buffer_km=blend_buffer_km,
            )
            if w <= 0.0:
                continue

            site_idx = None
            for g_idx, s_idx in coord_to_grids[(lon, lat)]:
                if g_idx == grid_idx:
                    site_idx = s_idx
                    break
            if site_idx is None:
                _, site_idx = trees[grid_idx].query([lon, lat])

            da = datasets[grid_idx][actual_var].isel(site=int(site_idx))
            series = da.compute().values if hasattr(da.data, "compute") else da.values
            series_list.append(series)
            weight_list.append(w)

        if not series_list:
            grid_idx, site_idx = coord_to_grids[(lon, lat)][0]
            da = datasets[grid_idx][actual_var].isel(site=site_idx)
            blended = da.compute().values if hasattr(da.data, "compute") else da.values
        elif len(series_list) == 1:
            blended = series_list[0]
        else:
            stacked = np.stack(series_list)
            weights_arr = np.array(weight_list)
            blended = (
                _circular_weighted_blend_deg(stacked, weights_arr)
                if circular
                else _linear_weighted_blend(stacked, weights_arr)
            )

        merged_data.append(blended)
        merged_coords.append([lon, lat])

    times = common_time.values
    merged_arr = np.asarray(merged_data)
    coords_arr = np.asarray(merged_coords)

    ds_out = xr.Dataset(
        {
            actual_var: (["time", "site"], merged_arr.T.astype(np.float32), {
                "long_name": datasets[0][actual_var].attrs.get("long_name", actual_var),
                "units": datasets[0][actual_var].attrs.get("units", ""),
            }),
            "lon": (["site"], coords_arr[:, 0].astype(np.float32)),
            "lat": (["site"], coords_arr[:, 1].astype(np.float32)),
        },
        coords={"time": times, "site": np.arange(len(merged_arr))},
        attrs={
            "grids": ", ".join(grid_names),
            "variable": actual_var,
            "steepness": steepness,
            "blend_buffer_km": blend_buffer_km,
            "blend_method": blend_label,
        },
    )

    for ds in datasets:
        ds.close()
    gc.collect()

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        _save_merged_netcdf(ds_out, output_file)
        print(f"Saved {output_file}")

    return ds_out


def merge_all_binwaves_bmus_smooth(
    project_root: Path = DEFAULT_PROJECT_ROOT,
    output_dir: Path | None = None,
    grid_names: Sequence[str] = DEFAULT_GRID_NAMES,
    variables: Sequence[str] | None = None,
    *,
    steepness: float = 2.0,
    blend_buffer_km: float = 30.0,
    tolerance_deg: float = 0.001,
    skip_existing: bool = True,
) -> dict[str, Path]:
    """Merge all variables with smooth buffer blending (circular for dp/dm/dp0, linear otherwise)."""
    project_root = Path(project_root)
    output_dir = Path(output_dir or project_root / "outputs" / "merged_grids_binwaves_bmus")
    output_dir.mkdir(parents=True, exist_ok=True)

    if variables is None:
        variables = discover_binwaves_bmus_variables(project_root, grid_names)
    if not variables:
        raise ValueError("No BinWaves_BMUS variables found under any grid.")

    results: dict[str, Path] = {}

    for var_name in variables:
        final_path = output_dir / f"{var_name}_merged_all.nc"
        if skip_existing and final_path.is_file():
            print(f"Skip {var_name}: {final_path.name} exists")
            results[var_name] = final_path
            continue

        grid_files = []
        for grid_name in grid_names:
            p = grid_binwaves_bmus_file(project_root, grid_name, var_name)
            if p.is_file():
                grid_files.append(p)
            else:
                print(f"  Missing {grid_name}: {p.name}")

        if not grid_files:
            print(f"No files for {var_name}; skipping")
            continue

        blend_kind = "circular" if var_name in CIRCULAR_BLEND_VARS else "linear"
        print(f"Smooth merge {var_name} ({blend_kind}) from {len(grid_files)} grid(s)...")
        merge_grids_smooth(
            grid_files,
            var_name=var_name,
            steepness=steepness,
            blend_buffer_km=blend_buffer_km,
            tolerance_deg=tolerance_deg,
            output_file=final_path,
        )
        results[var_name] = final_path
        print(f"  -> {final_path}")

    return results


def _site_coords(ds: xr.Dataset, site_dim: str) -> tuple[np.ndarray, np.ndarray]:
    lon = lat = None
    for name in ("lon", "longitude", "coord_x", "x"):
        if name in ds.coords and site_dim in ds[name].dims:
            lon = ds[name].values
            break
        if name in ds.data_vars and site_dim in ds[name].dims:
            lon = ds[name].values
            break
    for name in ("lat", "latitude", "coord_y", "y"):
        if name in ds.coords and site_dim in ds[name].dims:
            lat = ds[name].values
            break
        if name in ds.data_vars and site_dim in ds[name].dims:
            lat = ds[name].values
            break
    if lon is None or lat is None:
        raise RuntimeError("Could not find per-site lon/lat in merged dataset")
    return lon, lat


def _circular_mean_deg(da: xr.DataArray, dim: str) -> xr.DataArray:
    rad = np.deg2rad(da)
    return (np.rad2deg(np.arctan2(np.sin(rad).mean(dim=dim, skipna=True), np.cos(rad).mean(dim=dim, skipna=True))) % 360)


def generate_webpage_geojson(
    merged_dir: Path,
    output_dir: Path,
    *,
    merged_suffix: str = "_merged_all",
    time_start: str | None = None,
    time_end: str | None = None,
) -> dict[str, Path]:
    """
    Write ``wave_statistics_all.geojson``, ``wave_statistics_hs.geojson``,
    ``wave_statistics_dp.geojson`` (same layout as ``inputs/webpage_examples``).
    """
    merged_dir = Path(merged_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {var: merged_dir / f"{var}{merged_suffix}.nc" for var in WEBPAGE_VARS}
    optional_paths = {
        var: merged_dir / f"{var}{merged_suffix}.nc" for var in WEBPAGE_OPTIONAL_VARS
    }
    missing = [v for v, p in paths.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing merged files for webpage GeoJSON:\n  "
            + "\n  ".join(str(paths[v]) for v in missing)
        )

    optional_vars = [v for v, p in optional_paths.items() if p.is_file()]
    skipped_optional = [v for v in WEBPAGE_OPTIONAL_VARS if v not in optional_vars]
    if skipped_optional:
        print(
            "WARN: skipping optional webpage vars (merged file not found): "
            + ", ".join(skipped_optional)
        )

    load_vars = tuple(WEBPAGE_VARS) + tuple(optional_vars)
    datasets = {var: xr.open_dataset(paths.get(var) or optional_paths[var]) for var in load_vars}
    try:
        var_names = {
            var: (var if var in datasets[var].data_vars else list(datasets[var].data_vars)[0])
            for var in load_vars
        }
        hs = datasets["hs"][var_names["hs"]]
        time_dim = "time"
        site_dim = [d for d in hs.dims if d != time_dim][0]

        if time_start or time_end:
            hs = hs.sel(time=slice(time_start, time_end))
        time_vals = pd.to_datetime(hs[time_dim].values)
        t0 = time_vals[0].strftime("%Y-%m-%d")
        t1 = time_vals[-1].strftime("%Y-%m-%d")

        lon, lat = _site_coords(datasets["hs"], site_dim)
        n_sites = hs.sizes[site_dim]

        def sel_time(da: xr.DataArray) -> xr.DataArray:
            return da.sel(time=slice(time_start, time_end)) if (time_start or time_end) else da

        tp = sel_time(datasets["tp"][var_names["tp"]])
        dp = sel_time(datasets["dp"][var_names["dp"]])
        dm = sel_time(datasets["dm"][var_names["dm"]])
        tm02_mean = None
        if "tm02" in datasets:
            tm02 = sel_time(datasets["tm02"][var_names["tm02"]])
            tm02_mean = tm02.mean(dim=time_dim, skipna=True)

        hs_mean = hs.mean(dim=time_dim, skipna=True)
        hs_max = hs.max(dim=time_dim, skipna=True)
        hs_95 = hs.quantile(0.95, dim=time_dim, skipna=True)
        tp_mean = tp.mean(dim=time_dim, skipna=True)
        dp_mean = _circular_mean_deg(dp, time_dim)
        dm_mean = _circular_mean_deg(dm, time_dim)

        # --- wave_statistics_all.geojson ---
        all_features = []
        for idx in tqdm(range(n_sites), desc="wave_statistics_all"):
            def _val(da):
                v = float(da.isel({site_dim: idx}).values)
                return v if np.isfinite(v) else None

            props = {
                "id": f"{idx:07d}",
                "dataset_index": int(idx),
                "time_start": t0,
                "time_end": t1,
                "Hs_mean": _val(hs_mean),
                "Hs_max": _val(hs_max),
                "Hs_95": _val(hs_95),
                "Tp_mean": _val(tp_mean),
                "Dp_mean": _val(dp_mean),
                "Dm_mean": _val(dm_mean),
            }
            if tm02_mean is not None:
                props["Tm02_mean"] = _val(tm02_mean)
            all_features.append(
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": {"type": "Point", "coordinates": [float(lon[idx]), float(lat[idx])]},
                }
            )
        all_path = output_dir / "wave_statistics_all.geojson"
        all_path.write_text(json.dumps({"type": "FeatureCollection", "features": all_features}, indent=2))

        # --- wave_statistics_hs.geojson (monthly + seasonal climatology) ---
        hs_month = hs.groupby(f"{time_dim}.month").mean(dim=time_dim, skipna=True)
        hs_season = {}
        for season, months in SEASON_MONTHS.items():
            avail = [m for m in months if int(m) in hs_month["month"].values]
            hs_season[season] = hs_month.sel(month=avail).mean(dim="month", skipna=True) if avail else None

        hs_features = []
        for idx in tqdm(range(n_sites), desc="wave_statistics_hs"):
            monthly = {}
            for m_idx, m_name in enumerate(MONTH_NAMES, start=1):
                if m_idx in hs_month["month"].values:
                    monthly[m_name] = float(hs_month.sel(month=m_idx).isel({site_dim: idx}).values)
                else:
                    monthly[m_name] = np.nan
            seasonal = {
                s: (float(da.isel({site_dim: idx}).values) if da is not None else np.nan)
                for s, da in hs_season.items()
            }
            props = {"id": f"{idx:07d}", "Hs_mean": float(hs_mean.isel({site_dim: idx}).values)}
            props.update({f"Hs_mean_{m}": monthly[m] for m in MONTH_NAMES})
            props.update({f"Hs_mean_{s}": seasonal[s] for s in SEASON_MONTHS})
            hs_features.append(
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": {"type": "Point", "coordinates": [float(lon[idx]), float(lat[idx])]},
                }
            )
        hs_path = output_dir / "wave_statistics_hs.geojson"
        hs_path.write_text(json.dumps({"type": "FeatureCollection", "features": hs_features}, indent=2))

        # --- wave_statistics_dp.geojson (circular monthly + seasonal) ---
        dp_rad = np.deg2rad(dp)
        dp_sin = np.sin(dp_rad)
        dp_cos = np.cos(dp_rad)
        dp_month_sin = dp_sin.groupby(f"{time_dim}.month").mean(dim=time_dim, skipna=True)
        dp_month_cos = dp_cos.groupby(f"{time_dim}.month").mean(dim=time_dim, skipna=True)
        dp_month = (np.rad2deg(np.arctan2(dp_month_sin, dp_month_cos)) % 360)
        dp_season = {}
        for season, months in SEASON_MONTHS.items():
            avail = [m for m in months if int(m) in dp_month_sin["month"].values]
            if avail:
                sin_s = dp_month_sin.sel(month=avail).mean(dim="month", skipna=True)
                cos_s = dp_month_cos.sel(month=avail).mean(dim="month", skipna=True)
                dp_season[season] = (np.rad2deg(np.arctan2(sin_s, cos_s)) % 360)
            else:
                dp_season[season] = None

        dp_features = []
        for idx in tqdm(range(n_sites), desc="wave_statistics_dp"):
            monthly = {}
            for m_idx, m_name in enumerate(MONTH_NAMES, start=1):
                if m_idx in dp_month["month"].values:
                    monthly[m_name] = float(dp_month.sel(month=m_idx).isel({site_dim: idx}).values)
                else:
                    monthly[m_name] = np.nan
            seasonal = {
                s: (float(da.isel({site_dim: idx}).values) if da is not None else np.nan)
                for s, da in dp_season.items()
            }
            props = {"id": f"{idx:07d}", "Dp_mean": float(dp_mean.isel({site_dim: idx}).values)}
            props.update({f"Dp_mean_{m}": monthly[m] for m in MONTH_NAMES})
            props.update({f"Dp_mean_{s}": seasonal[s] for s in SEASON_MONTHS})
            dp_features.append(
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": {"type": "Point", "coordinates": [float(lon[idx]), float(lat[idx])]},
                }
            )
        dp_path = output_dir / "wave_statistics_dp.geojson"
        dp_path.write_text(json.dumps({"type": "FeatureCollection", "features": dp_features}, indent=2))

        return {"all": all_path, "hs": hs_path, "dp": dp_path}
    finally:
        for ds in datasets.values():
            ds.close()
