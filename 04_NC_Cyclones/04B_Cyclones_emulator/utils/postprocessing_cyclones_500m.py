"""Crop and smooth-merge cyclone partition outputs on 500 m points.

This module reuses the efficient methodology from the newer post-processing flow:
- coordinate matching with KDTree
- robust NetCDF writing
- smooth multi-grid blending (circular for directional variables)

It intentionally excludes BMUs and GeoJSON statistics generation.
"""

from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from postprocessing_binwaves_bmus import CIRCULAR_BLEND_VARS, merge_grids_smooth

DEFAULT_GRID_NAMES = ("grid1", "grid2", "grid3", "grid4")
PARTITION_PATTERN = re.compile(
    r"^(?P<var>[a-z0-9]+)_(?P<grid>grid\d+)_cyclone_(?P<id>\d+)\.nc$"
)


def discover_partition_variables(
    partitions_dir: Path,
    grid_names: Sequence[str] = DEFAULT_GRID_NAMES,
) -> list[str]:
    """Return sorted variables available in per-cyclone partition files."""
    partitions_dir = Path(partitions_dir)
    found: set[str] = set()
    for nc in partitions_dir.glob("*.nc"):
        m = PARTITION_PATTERN.match(nc.name)
        if not m:
            continue
        if m.group("grid") in grid_names:
            found.add(m.group("var"))
    return sorted(found)


def available_cyclone_ids_for_var(
    partitions_dir: Path,
    var_name: str,
    grid_names: Sequence[str] = DEFAULT_GRID_NAMES,
    *,
    intersection: bool = True,
) -> list[int]:
    """Return cyclone IDs for a variable across grids.

    If intersection=True, returns IDs present in all grids.
    Otherwise returns union across grids.
    """
    partitions_dir = Path(partitions_dir)
    ids_by_grid: dict[str, set[int]] = {}
    for grid in grid_names:
        ids = set()
        for nc in partitions_dir.glob(f"{var_name}_{grid}_cyclone_*.nc"):
            m = PARTITION_PATTERN.match(nc.name)
            if m:
                ids.add(int(m.group("id")))
        ids_by_grid[grid] = ids

    if not ids_by_grid:
        return []

    sets = list(ids_by_grid.values())
    merged = set.intersection(*sets) if intersection else set.union(*sets)
    return sorted(merged)


def _site_lon_lat(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    lon = None
    lat = None
    for name in ("lon", "coord_x", "longitude", "x"):
        if name in ds.coords and "site" in ds[name].dims:
            lon = np.asarray(ds[name].values).reshape(-1)
            break
        if name in ds.data_vars and "site" in ds[name].dims:
            lon = np.asarray(ds[name].values).reshape(-1)
            break
    for name in ("lat", "coord_y", "latitude", "y"):
        if name in ds.coords and "site" in ds[name].dims:
            lat = np.asarray(ds[name].values).reshape(-1)
            break
        if name in ds.data_vars and "site" in ds[name].dims:
            lat = np.asarray(ds[name].values).reshape(-1)
            break
    if lon is None or lat is None or len(lon) != len(lat):
        raise ValueError("Could not determine per-site lon/lat coordinates.")
    return lon, lat


def _safe_save_netcdf(ds_out: xr.Dataset, output_file: Path) -> None:
    """Write NetCDF with temporary file and fallback engines."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_file.parent / f".{output_file.name}.tmp"
    temp_output.unlink(missing_ok=True)

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
        temp_output.unlink(missing_ok=True)
        try:
            ds_out.to_netcdf(
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
        raise RuntimeError(f"Failed writing {output_file}: {last_error}") from last_error

    if temp_output.stat().st_size < 10_000:
        temp_output.unlink(missing_ok=True)
        raise RuntimeError(f"Suspiciously small output file: {temp_output}")

    output_file.unlink(missing_ok=True)
    temp_output.replace(output_file)


def crop_partitions_to_points_geojson(
    project_root: Path,
    partitions_dir: Path | None = None,
    output_dir: Path | None = None,
    points_geojson_file: Path | str = "inputs/isobath_10m_points_500m.geojson",
    grid_names: Sequence[str] = DEFAULT_GRID_NAMES,
    variables: Sequence[str] | None = None,
    cyclone_ids: Sequence[int] | None = None,
    *,
    point_match_tolerance: float = 1e-6,
    skip_existing: bool = True,
    verbose: bool = True,
) -> list[Path]:
    """Crop cyclone partition files to reference 500 m points."""
    import geopandas as gpd

    project_root = Path(project_root)
    partitions_dir = Path(partitions_dir or project_root / "outputs" / "partitions_cyclones")
    output_dir = Path(output_dir or project_root / "outputs" / "cropped_500m_partitions_cyclones")
    points_geojson_file = Path(points_geojson_file)
    if not points_geojson_file.is_absolute():
        points_geojson_file = project_root / points_geojson_file

    if variables is None:
        variables = discover_partition_variables(partitions_dir, grid_names)

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

    cyclone_filter = set(int(c) for c in cyclone_ids) if cyclone_ids is not None else None
    saved: list[Path] = []

    tasks: list[tuple[str, str, int, Path]] = []
    for nc in sorted(partitions_dir.glob("*.nc")):
        m = PARTITION_PATTERN.match(nc.name)
        if not m:
            continue
        var_name = m.group("var")
        grid_name = m.group("grid")
        cid = int(m.group("id"))
        if var_name not in variables or grid_name not in grid_names:
            continue
        if cyclone_filter is not None and cid not in cyclone_filter:
            continue
        tasks.append((var_name, grid_name, cid, nc))

    if verbose:
        print(f"Cropping partition files to 500 m points")
        print(f"  Input: {partitions_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Files selected: {len(tasks)}")

    for var_name, grid_name, cid, src in tqdm(tasks, desc="Cropping cyclone partitions", disable=not verbose):
        dst = output_dir / f"{var_name}_{grid_name}_cyclone_{cid}.nc"
        if skip_existing and dst.is_file():
            saved.append(dst)
            continue

        with xr.open_dataset(src, chunks={"time": 10_000}) as ds:
            lon, lat = _site_lon_lat(ds)
            site_mask = np.array([(float(x), float(y)) in keep_set for x, y in zip(lon, lat)], dtype=bool)
            if not np.all(site_mask):
                candidate_idx = np.where(~site_mask)[0]
                candidate_xy = np.column_stack([lon[candidate_idx], lat[candidate_idx]])
                distances, _ = keep_tree.query(candidate_xy, distance_upper_bound=point_match_tolerance)
                site_mask[candidate_idx[np.isfinite(distances)]] = True

            n_kept = int(site_mask.sum())
            if n_kept == 0:
                if verbose:
                    print(f"  No matching sites in {src.name}")
                continue

            subset = ds.isel(site=site_mask)
            if "coord_x" in subset.variables and "lon" not in subset.variables:
                subset = subset.rename({"coord_x": "lon", "coord_y": "lat"})
            subset = subset.drop_vars(
                [v for v in ("coord_x", "coord_y", "site_id") if v in subset.variables],
                errors="ignore",
            )
            subset.attrs = {}
            if "cyclone_id" not in subset.coords:
                subset = subset.assign_coords(cyclone_id=cid)

            _safe_save_netcdf(subset, dst)
            saved.append(dst)

    return saved


def merge_all_cyclone_partitions_smooth(
    cropped_dir: Path,
    output_dir: Path | None = None,
    grid_names: Sequence[str] = DEFAULT_GRID_NAMES,
    variables: Sequence[str] | None = None,
    cyclone_ids: Sequence[int] | None = None,
    *,
    steepness: float = 2.0,
    blend_buffer_km: float = 30.0,
    tolerance_deg: float = 0.001,
    skip_existing: bool = True,
    output_suffix: str = "_merged_cyclones_500m",
) -> dict[str, Path]:
    """Smooth-merge cropped per-cyclone files and concatenate by cyclone_id."""
    cropped_dir = Path(cropped_dir)
    output_dir = Path(output_dir or cropped_dir.parent / "merged_500m_partitions_cyclones")
    output_dir.mkdir(parents=True, exist_ok=True)

    if variables is None:
        variables = discover_partition_variables(cropped_dir, grid_names)
    if not variables:
        raise ValueError(f"No variables found in {cropped_dir}")

    cyclone_filter = set(int(c) for c in cyclone_ids) if cyclone_ids is not None else None
    results: dict[str, Path] = {}

    for var_name in variables:
        final_path = output_dir / f"{var_name}{output_suffix}.nc"
        if skip_existing and final_path.is_file():
            results[var_name] = final_path
            print(f"Skip {var_name}: {final_path.name} exists")
            continue

        ids = available_cyclone_ids_for_var(cropped_dir, var_name, grid_names, intersection=True)
        if cyclone_filter is not None:
            ids = [cid for cid in ids if cid in cyclone_filter]
        if not ids:
            print(f"No cyclone IDs available for {var_name}; skipping")
            continue

        merged_by_cid: list[xr.Dataset] = []
        blend_kind = "circular" if var_name in CIRCULAR_BLEND_VARS else "linear"
        print(f"Merging {var_name} ({blend_kind}) for {len(ids)} cyclone IDs")

        for cid in tqdm(ids, desc=f"  {var_name}: cyclones"):
            grid_files: list[Path] = []
            for grid_name in grid_names:
                p = cropped_dir / f"{var_name}_{grid_name}_cyclone_{cid}.nc"
                if p.is_file():
                    grid_files.append(p)
            if not grid_files:
                continue

            ds_merged = merge_grids_smooth(
                grid_files=grid_files,
                var_name=var_name,
                steepness=steepness,
                blend_buffer_km=blend_buffer_km,
                tolerance_deg=tolerance_deg,
                output_file=None,
            )
            ds_merged = ds_merged.expand_dims(cyclone_id=[cid])
            merged_by_cid.append(ds_merged)

            if len(merged_by_cid) % 20 == 0:
                gc.collect()

        if not merged_by_cid:
            print(f"Could not merge any cyclone for {var_name}; skipping")
            continue

        combined = xr.concat(merged_by_cid, dim="cyclone_id").sortby("cyclone_id")
        _safe_save_netcdf(combined, final_path)

        for ds in merged_by_cid:
            try:
                ds.close()
            except Exception:
                pass
        gc.collect()

        results[var_name] = final_path
        print(f"  -> {final_path}")

    return results
