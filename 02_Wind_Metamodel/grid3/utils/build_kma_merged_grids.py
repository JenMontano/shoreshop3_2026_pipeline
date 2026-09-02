"""
Build ShoreShop ``cropped_variables``-format bulk NetCDFs with BinWaves + KMA cluster SWAN.

Reads BinWaves bulk from ``cropped_variables`` (e.g. ``hs_grid1_masked.nc``) and writes
combined fields with the same site list, coordinates, chunking, and variable layout::

    phs0_grid{N}_BinWaves_BMUS.nc   (BMU merged into wind-sea partition)
    ptp0_grid{N}_BinWaves_BMUS.nc
    dp0_grid{N}_BinWaves_BMUS.nc
    hs_grid{N}_BinWaves_BMUS.nc
    tp_grid{N}_BinWaves_BMUS.nc
    dp_grid{N}_BinWaves_BMUS.nc
    dm_grid{N}_BinWaves_BMUS.nc
    tm02_grid{N}_BinWaves_BMUS.nc

Example::

    python -m utils.build_kma_merged_grids \\
        --input-folder ../../01_BinWaves/outputs/cropped_variables \\
        --output-folder outputs/BinWaves_BMUS \\
        --grid-id 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import loadmat

from utils.kma_cluster_swan import (
    DEFAULT_CLUSTER_CASES_ROOT,
    DEFAULT_KMA_BMU_CSV,
    PARTITION_IDS,
    _default_project_root,
    bmu_for_hourly_times,
    combine_partitions_with_bmu_arrays,
    find_cluster_mat,
    find_reference_cluster_mat,
    infer_n_clusters,
    load_bmu_assignments,
)

DEFAULT_INPUT_FOLDER = str(
    Path(__file__).resolve().parents[3] / "01_BinWaves" / "outputs" / "cropped_variables"
)
DEFAULT_OUTPUT_FOLDER = "outputs/BinWaves_BMUS"
DEFAULT_GRID_ID = 1
DEFAULT_TIME_CHUNK = 8760  # ~1 year of hourly steps (processing batch size)
DEFAULT_NETCDF_CHUNKS = (4392, 454)  # ShoreShop cropped_variables layout for grid1
DEFAULT_NETCDF_COMPLEVEL = 4

BULK_STEMS = ("hs", "tp", "dp", "dm", "tm02")
PARTITION0_OUTPUTS = (("phs0", "hs"), ("ptp0", "tp"), ("dp0", "dp"))


def _cropped_filename(stem: str, grid_id: int) -> str:
    return f"{stem}_grid{grid_id}_masked.nc"


def _output_filename(stem: str, grid_id: int) -> str:
    return f"{stem}_grid{grid_id}_BinWaves_BMUS.nc"


def _cropped_bulk_paths(folder: Path, grid_id: int) -> dict[str, Path]:
    return {stem: folder / _cropped_filename(stem, grid_id) for stem in BULK_STEMS}


def _cropped_partition_paths(folder: Path, grid_id: int) -> dict[int, tuple[Path, Path, Path]]:
    gid = f"grid{grid_id}"
    out: dict[int, tuple[Path, Path, Path]] = {}
    for pid in PARTITION_IDS:
        phs = folder / f"phs{pid}_{gid}_masked.nc"
        ptp = folder / f"ptp{pid}_{gid}_masked.nc"
        dp_part = folder / f"dp{pid}_{gid}_masked.nc"
        if phs.is_file() and ptp.is_file() and dp_part.is_file():
            out[pid] = (phs, ptp, dp_part)
    return out


def _resolve_bulk_nc_var_name(expected: str, ds: xr.Dataset) -> str | None:
    """Match ShoreShop variable names (e.g. phs0 stored as ``hs``)."""
    if expected in ds.data_vars:
        return expected
    for name in ds.data_vars:
        return name
    return None


def _read_time_site_chunk(
    ds: xr.Dataset,
    var: str,
    chunk_times: pd.DatetimeIndex,
    site_idx: np.ndarray,
) -> np.ndarray:
    """Return array shaped ``(time, site)`` regardless of on-disk dim order."""
    da = ds[var].sel(time=chunk_times).isel(site=site_idx)
    if da.dims == ("time", "site"):
        return da.values.astype(np.float32)
    if da.dims == ("site", "time"):
        return da.values.astype(np.float32).T
    raise ValueError(f"unexpected dims {da.dims} for {var!r} in {getattr(ds, 'encoding', {}).get('source', 'dataset')}")


def _swan_indices_for_sites(
    site_lons: np.ndarray,
    site_lats: np.ndarray,
    ref_mat: Path,
) -> tuple[np.ndarray, np.ndarray]:
    data = loadmat(ref_mat, squeeze_me=True)
    lon2d = np.asarray(data["Xp"], dtype=float)
    lat2d = np.asarray(data["Yp"], dtype=float)
    n_sites = len(site_lons)
    iy = np.empty(n_sites, dtype=int)
    ix = np.empty(n_sites, dtype=int)
    for s in range(n_sites):
        dist2d = (lon2d - site_lons[s]) ** 2 + (lat2d - site_lats[s]) ** 2
        flat = int(np.nanargmin(dist2d))
        iy[s], ix[s] = np.unravel_index(flat, dist2d.shape)
    return iy, ix


def build_cluster_site_lookup(
    cluster_root: Path,
    site_lons: np.ndarray,
    site_lats: np.ndarray,
    *,
    ref_mat: Path | None = None,
    n_clusters: int = 250,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Preload cluster SWAN bulk at every hindcast site for each BMU id.

    Returns ``(hs, tp, dp, dm, tm)`` arrays with shape ``(n_clusters, n_sites)`` indexed
    by cluster id (0..n_clusters-1), plus the list of BMUs with available output.
    """
    cluster_root = Path(cluster_root)
    ref_mat = ref_mat or find_reference_cluster_mat(cluster_root)
    iy, ix = _swan_indices_for_sites(site_lons, site_lats, ref_mat)

    n_sites = len(site_lons)
    cluster_hs = np.full((n_clusters, n_sites), np.nan, dtype=np.float32)
    cluster_tp = np.full((n_clusters, n_sites), np.nan, dtype=np.float32)
    cluster_dp = np.full((n_clusters, n_sites), np.nan, dtype=np.float32)
    cluster_dm = np.full((n_clusters, n_sites), np.nan, dtype=np.float32)
    cluster_tm = np.full((n_clusters, n_sites), np.nan, dtype=np.float32)
    loaded: list[int] = []

    for cid in range(n_clusters):
        mat_path = find_cluster_mat(cluster_root, cid)
        if mat_path is None:
            continue
        data = loadmat(mat_path, squeeze_me=True)
        if "Hsig" not in data:
            continue
        hs2d = np.asarray(data["Hsig"], dtype=float)
        cluster_hs[cid, :] = hs2d[iy, ix].astype(np.float32)
        if "TPsmoo" in data:
            cluster_tp[cid, :] = np.asarray(data["TPsmoo"], dtype=float)[iy, ix].astype(np.float32)
        if "PkDir" in data:
            cluster_dp[cid, :] = np.asarray(data["PkDir"], dtype=float)[iy, ix].astype(np.float32)
        if "Dir" in data:
            cluster_dm[cid, :] = np.asarray(data["Dir"], dtype=float)[iy, ix].astype(np.float32)
        if "Tm02" in data:
            cluster_tm[cid, :] = np.asarray(data["Tm02"], dtype=float)[iy, ix].astype(np.float32)
        loaded.append(cid)

    print(
        f"Cluster lookup: {len(loaded)}/{n_clusters} BMUs loaded for {n_sites} sites "
        f"(ref mat {ref_mat.name})"
    )
    missing = [cid for cid in range(n_clusters) if cid not in loaded]
    if missing:
        print(
            f"WARN missing cluster output for BMUs: "
            f"{missing[:20]}{'...' if len(missing) > 20 else ''}"
        )
    return cluster_hs, cluster_tp, cluster_dp, cluster_dm, cluster_tm, loaded


def _create_partition_output_dataset(
    template: xr.Dataset,
    template_path: Path,
    var_name: str,
    time_index: pd.DatetimeIndex,
) -> xr.Dataset:
    """Clone cropped_variables partition layout: ``(site, time)`` with ``part`` coord."""
    n_time = len(time_index)
    n_site = template.sizes["site"]
    var_attrs: dict = {}
    with nc.Dataset(template_path) as ds:
        if var_name in ds.variables:
            var_attrs = {
                a: ds.variables[var_name].getncattr(a)
                for a in ds.variables[var_name].ncattrs()
                if a != "_FillValue"
            }
    data = np.full((n_site, n_time), np.nan, dtype=np.float32)
    coords: dict = {
        "site": template["site"],
        "time": time_index,
        "lon": template["lon"],
        "lat": template["lat"],
    }
    if "part" in template.coords:
        coords["part"] = template["part"]
    return xr.Dataset(
        {var_name: (("site", "time"), data, var_attrs)},
        coords=coords,
        attrs={},
    )


def _create_cropped_output_dataset(
    template: xr.Dataset,
    template_path: Path,
    var_name: str,
    time_index: pd.DatetimeIndex,
) -> xr.Dataset:
    """Clone cropped_variables layout: lon/lat coords, empty global attrs."""
    n_time = len(time_index)
    n_site = template.sizes["site"]
    var_attrs: dict = {}
    with nc.Dataset(template_path) as ds:
        if var_name in ds.variables:
            var_attrs = {
                a: ds.variables[var_name].getncattr(a)
                for a in ds.variables[var_name].ncattrs()
                if a != "_FillValue"
            }
    data = np.full((n_time, n_site), np.nan, dtype=np.float32)
    return xr.Dataset(
        {var_name: (("time", "site"), data, var_attrs)},
        coords={
            "time": time_index,
            "site": template["site"],
            "lon": template["lon"],
            "lat": template["lat"],
        },
        attrs={},
    )


def _read_on_disk_chunksizes(nc_path: Path, variable: str) -> tuple[int, int] | None:
    """Read NetCDF chunk layout from disk (xarray encoding is often empty)."""
    with nc.Dataset(nc_path, "r") as ds:
        if variable not in ds.variables:
            return None
        layout = ds.variables[variable].chunking()
        if layout == "contiguous":
            return None
        return tuple(int(x) for x in layout)


def _resolve_netcdf_chunks(
    template_path: Path,
    template: xr.Dataset,
    variable: str,
    *,
    n_time: int,
    n_sites: int,
    netcdf_chunks: tuple[int, int] | None = None,
    dim_order: tuple[str, str] = ("time", "site"),
) -> tuple[int, int]:
    """Pick output chunk layout; default matches ShoreShop cropped_variables."""
    if netcdf_chunks is not None:
        c0, c1 = netcdf_chunks
        if dim_order == ("time", "site"):
            return (min(int(c0), n_time), min(int(c1), n_sites))
        return (min(int(c0), n_sites), min(int(c1), n_time))

    ref = None
    if variable in template:
        ref = template[variable].encoding.get("chunksizes")
    if ref is None or len(ref) != 2:
        ref = _read_on_disk_chunksizes(template_path, variable)
    if ref is not None and len(ref) == 2:
        if dim_order == ("time", "site"):
            return (min(int(ref[0]), n_time), min(int(ref[1]), n_sites))
        return (min(int(ref[0]), n_sites), min(int(ref[1]), n_time))
    t0, s0 = DEFAULT_NETCDF_CHUNKS
    if dim_order == ("time", "site"):
        return (min(t0, n_time), min(s0, n_sites))
    return (min(s0, n_sites), min(t0, n_time))


def _cropped_output_encoding(
    template_path: Path,
    var_name: str,
    *,
    chunksizes: tuple[int, int],
    complevel: int = DEFAULT_NETCDF_COMPLEVEL,
) -> dict:
    """NetCDF encoding matching ShoreShop cropped_variables layout."""
    encoding: dict = {
        var_name: {
            "zlib": True,
            "complevel": complevel,
            "dtype": "float32",
            "chunksizes": chunksizes,
            "_FillValue": np.float32(np.nan),
        }
    }
    with nc.Dataset(template_path, "r") as ds:
        for vname in ("lon", "lat", "site", "time"):
            if vname not in ds.variables:
                continue
            var = ds.variables[vname]
            enc: dict = {}
            if vname in ("lon", "lat"):
                enc["dtype"] = "float64"
                enc["_FillValue"] = np.float64(np.nan)
            elif vname == "site":
                enc["dtype"] = "int64"
            elif vname == "time":
                for a in ("units", "calendar"):
                    if a in var.ncattrs():
                        enc[a] = var.getncattr(a)
            encoding[vname] = enc
    return encoding


def _write_netcdf_chunk(
    path: Path,
    var_name: str,
    t0: int,
    t1: int,
    arr: np.ndarray,
    handles: dict[Path, nc.Dataset],
    *,
    site_time: bool = False,
) -> None:
    """
    Write one time slice to an existing NetCDF via netCDF4.

    ``arr`` is always ``(time, site)``; partition files on disk use ``(site, time)``.
    """
    ds = handles.get(path)
    if ds is None:
        ds = nc.Dataset(path, "r+")
        handles[path] = ds
    values = np.asarray(arr, dtype=np.float32)
    if site_time:
        ds.variables[var_name][:, t0:t1] = values.T
    else:
        ds.variables[var_name][t0:t1, :] = values
    ds.sync()


def _verify_output_has_data(out_paths: dict[str, Path], *, time_chunk: int) -> None:
    """Fail fast if chunk writes did not persist."""
    with xr.open_dataset(out_paths["hs"]) as ds:
        sample = ds["hs"].isel(time=slice(0, min(time_chunk, ds.sizes["time"]))).values
    if not np.isfinite(sample).any():
        raise RuntimeError(
            "Output NetCDF contains no finite values after first chunk — "
            "chunk writes did not persist"
        )
    frac = float(np.isfinite(sample).mean())
    print(f"Verified hs chunk 0: {frac * 100:.1f}% finite samples")


def _verify_output_chunks(
    chunk_specs: dict[str, tuple[Path, str, tuple[int, int]]],
) -> None:
    """Confirm on-disk chunk layout — no separate rechunk pass should be needed."""
    for label, (path, var_name, expected_chunks) in chunk_specs.items():
        on_disk = _read_on_disk_chunksizes(path, var_name)
        if on_disk != expected_chunks:
            raise RuntimeError(
                f"{path.name}: expected on-disk chunks {expected_chunks}, got {on_disk}"
            )
        print(f"Verified {label} on-disk chunks {on_disk}")


def _verify_output_matches_template(
    out_path: Path,
    template_path: Path,
    var_name: str,
    *,
    site_idx: np.ndarray | None = None,
) -> None:
    """Check dims, coords, and site list match the ShoreShop cropped template."""
    with xr.open_dataset(template_path) as tpl_raw, xr.open_dataset(out_path) as out:
        tpl = tpl_raw.isel(site=site_idx) if site_idx is not None else tpl_raw
        if out.sizes["site"] != tpl.sizes["site"]:
            raise RuntimeError(
                f"{out_path.name}: site count {out.sizes['site']} != template {tpl.sizes['site']}"
            )
        if list(out.coords) != list(tpl.coords):
            raise RuntimeError(
                f"{out_path.name}: coords {list(out.coords)} != template {list(tpl.coords)}"
            )
        if not np.array_equal(out["site"].values, tpl["site"].values):
            raise RuntimeError(f"{out_path.name}: site ids differ from template")
        if var_name not in out.data_vars:
            raise RuntimeError(f"{out_path.name}: missing variable {var_name!r}")
        if var_name not in tpl:
            raise RuntimeError(f"{out_path.name}: variable {var_name!r} not in template")
        if out[var_name].dims != tpl[var_name].dims:
            raise RuntimeError(
                f"{out_path.name}: expected dims {tpl[var_name].dims}, got {out[var_name].dims}"
            )
    print(f"Verified {out_path.name} matches template layout ({tpl.sizes['site']} sites)")


def build_merged_grids_binwaves_kma(
    *,
    input_folder: str | Path = DEFAULT_INPUT_FOLDER,
    output_folder: str | Path = DEFAULT_OUTPUT_FOLDER,
    grid_id: int = DEFAULT_GRID_ID,
    cluster_cases_root: str | Path = DEFAULT_CLUSTER_CASES_ROOT,
    kma_bmu_csv: str | Path = DEFAULT_KMA_BMU_CSV,
    project_root: str | Path | None = None,
    time_chunk: int = DEFAULT_TIME_CHUNK,
    netcdf_chunks: tuple[int, int] | None = None,
    time_start=None,
    time_end=None,
    site_indices: np.ndarray | None = None,
    with_dm: bool = True,
    with_tm02: bool = True,
    with_partition0: bool = True,
    overwrite: bool = False,
    n_clusters: int | None = None,
) -> Path:
    """
    Write ``{var}_grid{N}_BinWaves_BMUS.nc`` bulk files with BinWaves + KMA cluster combined waves.

    Input and output follow the ShoreShop ``cropped_variables`` schema (same seapoints,
    lon/lat coords, chunking, and variable names).
    """
    project_root = Path(project_root or _default_project_root())
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    if not output_folder.is_absolute():
        output_folder = project_root / output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    bulk_in = _cropped_bulk_paths(input_folder, grid_id)
    hs_in = bulk_in["hs"]
    if not hs_in.is_file():
        raise FileNotFoundError(hs_in)

    partition_inputs = _cropped_partition_paths(input_folder, grid_id)
    if len(partition_inputs) < len(PARTITION_IDS):
        missing = [pid for pid in PARTITION_IDS if pid not in partition_inputs]
        raise FileNotFoundError(
            f"Partition files phs/ptp/dp missing for ids {missing} in {input_folder} "
            f"(grid{grid_id})"
        )

    out_paths: dict[str, Path] = {
        stem: output_folder / _output_filename(stem, grid_id) for stem in ("hs", "tp", "dp")
    }
    if with_dm:
        out_paths["dm"] = output_folder / _output_filename("dm", grid_id)
    if with_tm02:
        out_paths["tm02"] = output_folder / _output_filename("tm02", grid_id)
    if with_partition0:
        for stem, _var in PARTITION0_OUTPUTS:
            out_paths[stem] = output_folder / _output_filename(stem, grid_id)

    if all(p.is_file() for p in out_paths.values()) and not overwrite:
        print(f"Output already exists in {output_folder} (use --overwrite to rebuild)")
        return output_folder

    cluster_root = Path(cluster_cases_root)
    if not cluster_root.is_absolute():
        cluster_root = project_root / cluster_root
    bmu_csv = Path(kma_bmu_csv)
    if not bmu_csv.is_absolute():
        bmu_csv = project_root / bmu_csv

    hs_in = bulk_in["hs"]
    chunk_specs: dict[str, tuple[Path, str, tuple[int, int]]] = {}
    with xr.open_dataset(hs_in) as hs_tpl:
        time_index = pd.DatetimeIndex(hs_tpl.time.values)
        if time_start is not None:
            time_index = time_index[time_index >= pd.Timestamp(time_start)]
        if time_end is not None:
            time_index = time_index[time_index <= pd.Timestamp(time_end)]
        if time_index.empty:
            raise ValueError("No timesteps selected")

        site_dim = "site"
        all_site_idx = np.arange(hs_tpl.sizes[site_dim], dtype=int)
        if site_indices is not None:
            site_idx = np.asarray(site_indices, dtype=int)
        else:
            site_idx = all_site_idx

        site_lons = np.asarray(hs_tpl["lon"].isel({site_dim: site_idx}).values, dtype=float)
        site_lats = np.asarray(hs_tpl["lat"].isel({site_dim: site_idx}).values, dtype=float)
        n_sites = len(site_idx)
        n_time = len(time_index)

        tpl_sub = hs_tpl.isel({site_dim: site_idx})
        eff_netcdf_chunks = _resolve_netcdf_chunks(
            hs_in,
            hs_tpl,
            "hs",
            n_time=n_time,
            n_sites=n_sites,
            netcdf_chunks=netcdf_chunks,
        )
        if not all(p.is_file() for p in out_paths.values()) or overwrite:
            for var, path in out_paths.items():
                if var in dict(PARTITION0_OUTPUTS):
                    continue
                template_path = bulk_in.get(var, bulk_in["hs"])
                if not template_path.is_file():
                    template_path = bulk_in["hs"]
                ds_out = _create_cropped_output_dataset(tpl_sub, template_path, var, time_index)
                ds_out.to_netcdf(
                    path,
                    encoding=_cropped_output_encoding(
                        template_path, var, chunksizes=eff_netcdf_chunks
                    ),
                )
                chunk_specs[var] = (path, var, eff_netcdf_chunks)
                print(f"Created {path} (chunks {eff_netcdf_chunks})")

            if with_partition0:
                part0_templates = {
                    "phs0": partition_inputs[0][0],
                    "ptp0": partition_inputs[0][1],
                    "dp0": partition_inputs[0][2],
                }
                for stem, var in PARTITION0_OUTPUTS:
                    tpl_path = part0_templates[stem]
                    with xr.open_dataset(tpl_path) as part_tpl:
                        part_tpl_sub = part_tpl.isel({site_dim: site_idx})
                        part_chunks = _resolve_netcdf_chunks(
                            tpl_path,
                            part_tpl,
                            var,
                            n_time=n_time,
                            n_sites=n_sites,
                            netcdf_chunks=netcdf_chunks,
                            dim_order=("site", "time"),
                        )
                        path = out_paths[stem]
                        ds_out = _create_partition_output_dataset(
                            part_tpl_sub, tpl_path, var, time_index
                        )
                        ds_out.to_netcdf(
                            path,
                            encoding=_cropped_output_encoding(
                                tpl_path, var, chunksizes=part_chunks
                            ),
                        )
                        chunk_specs[stem] = (path, var, part_chunks)
                        print(f"Created {path} (chunks {part_chunks})")

    eff_n_clusters = n_clusters or infer_n_clusters(cluster_root, bmu_csv)
    print(f"KMA clusters: {eff_n_clusters} BMUs (cases root {cluster_root})")

    cluster_hs, cluster_tp, cluster_dp, _cluster_dm, _cluster_tm, _loaded = build_cluster_site_lookup(
        cluster_root, site_lons, site_lats, n_clusters=eff_n_clusters
    )

    bmu_series = load_bmu_assignments(bmu_csv)
    bmu_hourly = bmu_for_hourly_times(bmu_series, time_index).astype(int).to_numpy()

    print(
        f"Combining grid{grid_id}: {n_time:,} hours × {n_sites:,} sites "
        f"({time_chunk:,}-step chunks) → {output_folder}"
    )

    nc_handles: dict[Path, nc.Dataset] = {}
    try:
        for t0 in range(0, n_time, time_chunk):
            t1 = min(t0 + time_chunk, n_time)
            chunk_times = time_index[t0:t1]
            bmu_chunk = bmu_hourly[t0:t1]

            phs_bw: list[np.ndarray] = []
            pdir_bw: list[np.ndarray] = []
            ptp_bw: list[np.ndarray] = []
            for pid in PARTITION_IDS:
                phs_path, ptp_path, dp_part_path = partition_inputs[pid]
                with (
                    xr.open_dataset(phs_path) as ds_phs,
                    xr.open_dataset(ptp_path) as ds_ptp,
                    xr.open_dataset(dp_part_path) as ds_dp_part,
                ):
                    phs_var = _resolve_bulk_nc_var_name(f"phs{pid}", ds_phs) or "hs"
                    ptp_var = _resolve_bulk_nc_var_name(f"ptp{pid}", ds_ptp) or "tp"
                    dp_var = _resolve_bulk_nc_var_name(f"dp{pid}", ds_dp_part) or "dp"
                    phs_bw.append(_read_time_site_chunk(ds_phs, phs_var, chunk_times, site_idx))
                    ptp_bw.append(_read_time_site_chunk(ds_ptp, ptp_var, chunk_times, site_idx))
                    pdir_bw.append(_read_time_site_chunk(ds_dp_part, dp_var, chunk_times, site_idx))

            hs_w = cluster_hs[bmu_chunk, :]
            tp_w = cluster_tp[bmu_chunk, :]
            dp_w = cluster_dp[bmu_chunk, :]

            phs0_out, ptp0_out, dp0_out, hs_out, tp_out, dp_out, dm_out, tm_out = (
                combine_partitions_with_bmu_arrays(
                    hs_w,
                    tp_w,
                    dp_w,
                    phs_bw=phs_bw,
                    ptp_bw=ptp_bw,
                    pdir_bw=pdir_bw,
                )
            )

            write_vars: list[tuple[str, np.ndarray, Path, bool]] = [
                ("hs", hs_out, out_paths["hs"], False),
                ("tp", tp_out, out_paths["tp"], False),
                ("dp", dp_out, out_paths["dp"], False),
            ]
            if with_dm:
                write_vars.append(("dm", dm_out, out_paths["dm"], False))
            if with_tm02:
                write_vars.append(("tm02", tm_out, out_paths["tm02"], False))
            if with_partition0:
                write_vars.extend(
                    [
                        ("hs", phs0_out, out_paths["phs0"], True),
                        ("tp", ptp0_out, out_paths["ptp0"], True),
                        ("dp", dp0_out, out_paths["dp0"], True),
                    ]
                )

            for var_name, arr, path, site_time in write_vars:
                _write_netcdf_chunk(
                    path, var_name, t0, t1, arr, nc_handles, site_time=site_time
                )

            print(
                f"  wrote timesteps {t0:,}–{t1 - 1:,} / {n_time - 1:,} "
                f"({100.0 * t1 / n_time:.1f}%)"
            )
            if t0 == 0:
                _verify_output_has_data(out_paths, time_chunk=time_chunk)
    finally:
        for ds in nc_handles.values():
            ds.close()

    _verify_output_chunks(chunk_specs)
    for var, path in out_paths.items():
        if var in dict(PARTITION0_OUTPUTS):
            stem, nc_var = next((s, v) for s, v in PARTITION0_OUTPUTS if s == var)
            template_path = {
                "phs0": partition_inputs[0][0],
                "ptp0": partition_inputs[0][1],
                "dp0": partition_inputs[0][2],
            }[stem]
            _verify_output_matches_template(path, template_path, nc_var, site_idx=site_idx)
        else:
            template_path = bulk_in.get(var, bulk_in["hs"])
            _verify_output_matches_template(path, template_path, var, site_idx=site_idx)

    print(f"Done: {output_folder}")
    return output_folder


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build cropped_variables-format NetCDFs with BinWaves + KMA cluster SWAN bulk."
    )
    parser.add_argument(
        "--input-folder",
        default=DEFAULT_INPUT_FOLDER,
        help="Source cropped_variables folder (BinWaves bulk for the target grid)",
    )
    parser.add_argument(
        "--output-folder",
        default=DEFAULT_OUTPUT_FOLDER,
        help="Destination folder for combined hs/tp/dp/dm/tm02_grid{N}_BinWaves_BMUS.nc",
    )
    parser.add_argument(
        "--grid-id",
        type=int,
        default=DEFAULT_GRID_ID,
        help="ShoreShop grid id (outputs e.g. hs_grid1_BinWaves_BMUS.nc; inputs hs_grid1_masked.nc)",
    )
    parser.add_argument(
        "--cluster-cases-root",
        default=str(DEFAULT_CLUSTER_CASES_ROOT),
        help="CASES_ONLY_WIND root with 000..249/output.mat",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Number of KMA BMUs (default: infer from BMU CSV or case folders)",
    )
    parser.add_argument(
        "--kma-bmu-csv",
        default=str(DEFAULT_KMA_BMU_CSV),
        help="Hourly/3-hourly BMU assignment CSV",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root for relative paths (default: parent of utils/ with CASES_ONLY_WIND)",
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=DEFAULT_TIME_CHUNK,
        help="Timesteps per processing chunk (I/O batch size only; not NetCDF chunk layout)",
    )
    parser.add_argument(
        "--netcdf-chunks",
        default=None,
        help="Output time,site chunk sizes (default: match input cropped_variables, e.g. 4392,454)",
    )
    parser.add_argument("--time-start", default=None, help="Optional start (e.g. 2018-01-01)")
    parser.add_argument("--time-end", default=None, help="Optional end (e.g. 2018-12-31)")
    parser.add_argument(
        "--max-sites",
        type=int,
        default=None,
        help="Debug: only process the first N sites",
    )
    parser.add_argument(
        "--no-dm",
        action="store_true",
        help="Skip dm_grid{N}_BinWaves_BMUS.nc (energy-weighted mean direction)",
    )
    parser.add_argument(
        "--no-partition0",
        action="store_true",
        help="Skip phs0/ptp0/dp0_grid{N}_BinWaves_BMUS.nc partition-0 outputs",
    )
    parser.add_argument(
        "--no-tm02",
        action="store_true",
        help="Skip tm02_grid{N}_BinWaves_BMUS.nc (harmonic mean period)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing outputs")
    args = parser.parse_args(argv)

    site_idx = None
    if args.max_sites is not None:
        site_idx = np.arange(int(args.max_sites), dtype=int)

    nc_chunks = None
    if args.netcdf_chunks is not None:
        parts = [int(x.strip()) for x in str(args.netcdf_chunks).split(",")]
        if len(parts) != 2:
            raise ValueError("--netcdf-chunks must be time,site e.g. 4392,454")
        nc_chunks = (parts[0], parts[1])

    build_merged_grids_binwaves_kma(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        grid_id=args.grid_id,
        cluster_cases_root=args.cluster_cases_root,
        kma_bmu_csv=args.kma_bmu_csv,
        project_root=args.project_root,
        time_chunk=args.time_chunk,
        netcdf_chunks=nc_chunks,
        time_start=args.time_start,
        time_end=args.time_end,
        site_indices=site_idx,
        with_dm=not args.no_dm,
        with_tm02=not args.no_tm02,
        with_partition0=not args.no_partition0,
        overwrite=args.overwrite,
        n_clusters=args.n_clusters,
    )


if __name__ == "__main__":
    main()
