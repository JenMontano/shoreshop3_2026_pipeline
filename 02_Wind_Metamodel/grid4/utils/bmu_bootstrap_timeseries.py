from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters; supports numpy broadcasting."""
    r = 6371000.0
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


@dataclass(frozen=True)
class SiteSelection:
    ids: list[str]
    lats: np.ndarray  # (n_site,)
    lons: np.ndarray  # (n_site,)
    src_index: np.ndarray  # (n_site,) indices into source grid
    distance_m: np.ndarray  # (n_site,)


def _guess_lat_lon(ds: xr.Dataset) -> tuple[str, str]:
    for lat_name in ("lat", "latitude"):
        if lat_name in ds.coords or lat_name in ds:
            for lon_name in ("lon", "longitude"):
                if lon_name in ds.coords or lon_name in ds:
                    return lat_name, lon_name
    raise KeyError(f"Could not find lat/lon in dataset. coords={list(ds.coords)} vars={list(ds.data_vars)}")


def _guess_site_dim(ds: xr.Dataset, lat_name: str, lon_name: str) -> str:
    lat = ds[lat_name] if lat_name in ds else ds.coords[lat_name]
    lon = ds[lon_name] if lon_name in ds else ds.coords[lon_name]
    if lat.ndim == 1 and lon.ndim == 1 and lat.dims == lon.dims:
        return lat.dims[0]
    # common wave/wind point dims
    for cand in ("site", "seapoint", "point", "stations"):
        if cand in ds.dims:
            return cand
    raise ValueError("Could not infer point/site dimension")


def select_all_sites_from_merged_file(
    src_nc: str,
    *,
    selected_site_indices: Iterable[int] | int | None = None,
    selected_coordinate: tuple[float, float] | None = None,
) -> SiteSelection:
    """
    Build a SiteSelection from a merged-style NetCDF file with per-site lat/lon.

    Files in ``inputs/merged_500m`` already store the variable on the desired (site)
    grid with ``lat`` / ``lon`` data variables, so no GeoJSON or distance matching is
    needed: every site in the file is used by default.

    - ``selected_coordinate=(lat, lon)`` keeps only the nearest single site
      (``distance_m`` reports the haversine distance to that site).
    - ``selected_site_indices`` keeps only the given integer site indices into the
      file's ``site`` dimension.
    """
    with xr.open_dataset(src_nc) as ds:
        lat_name, lon_name = _guess_lat_lon(ds)
        _ = _guess_site_dim(ds, lat_name, lon_name)
        src_lat = (ds[lat_name] if lat_name in ds else ds.coords[lat_name]).values.astype(float)
        src_lon = (ds[lon_name] if lon_name in ds else ds.coords[lon_name]).values.astype(float)

    if src_lat.ndim != 1 or src_lon.ndim != 1 or src_lat.shape != src_lon.shape:
        raise ValueError(
            f"Expected 1-D lat/lon of equal length in {src_nc}; "
            f"got lat={src_lat.shape}, lon={src_lon.shape}"
        )

    n = int(src_lat.shape[0])
    src_index_all = np.arange(n, dtype=int)

    if selected_coordinate is not None:
        sel_lat, sel_lon = float(selected_coordinate[0]), float(selected_coordinate[1])
        d = haversine_m(sel_lat, sel_lon, src_lat, src_lon)
        i_pick = int(np.argmin(d))
        src_index = np.asarray([i_pick], dtype=int)
        distance_m = np.asarray([float(d[i_pick])], dtype=float)
    elif selected_site_indices is not None:
        if isinstance(selected_site_indices, (int, np.integer)):
            wanted = [int(selected_site_indices)]
        else:
            wanted = [int(x) for x in selected_site_indices]
        bad = [w for w in wanted if w < 0 or w >= n]
        if bad:
            raise ValueError(f"selected_site_indices out of range [0, {n - 1}]: {bad}")
        src_index = np.asarray(wanted, dtype=int)
        distance_m = np.zeros_like(src_index, dtype=float)
    else:
        src_index = src_index_all
        distance_m = np.zeros(n, dtype=float)

    ids = [str(int(i)) for i in src_index]
    lats = src_lat[src_index].astype(float)
    lons = src_lon[src_index].astype(float)
    return SiteSelection(ids=ids, lats=lats, lons=lons, src_index=src_index, distance_m=distance_m)


def _load_hist_bmus(
    waveclusters_and_pcs: pd.DataFrame | str,
    *,
    bmu_col: str = "kma_bmus",
    start_date: str | None = "1980-01-01",
) -> xr.DataArray:
    if isinstance(waveclusters_and_pcs, str):
        df = pd.read_csv(waveclusters_and_pcs, index_col=0, parse_dates=False)
        df.index = pd.to_datetime(df.index).floor("D")
    else:
        df = waveclusters_and_pcs.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index).floor("D")

    if start_date is not None:
        df = df.loc[pd.Timestamp(start_date) :]

    if bmu_col not in df.columns:
        raise KeyError(f"Missing '{bmu_col}' in waveclusters_and_pcs")

    da = df[bmu_col].astype(int).to_xarray().rename("cluster_id")
    if "time" not in da.dims:
        old = list(da.dims)[0]
        da = da.rename({old: "time"})
    return da


def _build_pool_indices(
    hist_cluster: np.ndarray,
    hist_month: np.ndarray | None,
    *,
    monthly_conditioning: bool,
) -> dict[tuple[int, int] | int, np.ndarray]:
    pools: dict[tuple[int, int] | int, np.ndarray] = {}
    cids = np.unique(hist_cluster.astype(int))
    if monthly_conditioning:
        if hist_month is None:
            raise ValueError("hist_month is required when monthly_conditioning=True")
        for cid in cids:
            m_arr = hist_month
            for m in range(1, 13):
                idx = np.where((hist_cluster == cid) & (m_arr == m))[0]
                if idx.size:
                    pools[(int(cid), int(m))] = idx
    else:
        for cid in cids:
            idx = np.where(hist_cluster == cid)[0]
            if idx.size:
                pools[int(cid)] = idx
    return pools


def bmu_monthly_bootstrap_hist_row_indices(
    hist_cluster: np.ndarray,
    hist_month: np.ndarray,
    sim_cluster: np.ndarray,
    sim_month: np.ndarray,
    *,
    monthly_conditioning: bool,
    seed: int = 0,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> np.ndarray:
    """
    Historical time-row index chosen for each simulated day (one index per sim day).
    Rows with no valid pool are marked -1 (caller should leave NaNs in output).
    """
    rng = np.random.default_rng(seed)
    pools = _build_pool_indices(hist_cluster, hist_month, monthly_conditioning=monthly_conditioning)

    n_out = len(sim_cluster)
    picks = np.full(n_out, -1, dtype=np.int64)

    it = enumerate(sim_cluster.astype(int))
    if show_progress and tqdm is not None:
        it = tqdm(it, total=n_out, desc=progress_desc or "Bootstrap", unit="day")

    for i, cid in it:
        if monthly_conditioning:
            key = (int(cid), int(sim_month[i]))
            idxs = pools.get(key)
            if idxs is None or idxs.size == 0:
                idxs = np.where(hist_cluster == cid)[0]
        else:
            idxs = pools.get(int(cid))
        if idxs is None or idxs.size == 0:
            continue
        picks[i] = int(rng.choice(idxs))

    return picks


def bmu_monthly_bootstrap_timeseries(
    values_hist: np.ndarray,
    hist_cluster: np.ndarray,
    hist_month: np.ndarray,
    sim_cluster: np.ndarray,
    sim_month: np.ndarray,
    *,
    monthly_conditioning: bool,
    seed: int = 0,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> np.ndarray:
    """
    Bootstrap values for each simulated time step by sampling a random historical time index
    from the same BMU (and month, if enabled). Sampling is shared across sites for a given
    simulated time step (i.e., pick one historical day and take all sites from that day).
    """
    picks = bmu_monthly_bootstrap_hist_row_indices(
        hist_cluster,
        hist_month,
        sim_cluster,
        sim_month,
        monthly_conditioning=monthly_conditioning,
        seed=seed,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )
    n_out, n_site = len(sim_cluster), int(values_hist.shape[1])
    out = np.full((n_out, n_site), np.nan, dtype=float)
    valid = picks >= 0
    if np.any(valid):
        out[valid, :] = values_hist[picks[valid], :]
    return out


def _site_ckpt_paths(out_path: str) -> tuple[str, str, str]:
    """Sidecar paths for within-file site checkpoints (meta json, pick indices, memmap data)."""
    base = out_path + ".site_ckpt"
    return (base + ".json", base + ".pick_idx.npy", base + ".dat")


def _atomic_write_json(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, path)


def _remove_site_ckpt(out_path: str) -> None:
    for p in _site_ckpt_paths(out_path):
        try:
            if os.path.isfile(p):
                os.remove(p)
        except OSError:
            pass


def _in_progress_marker_path(out_path: str) -> str:
    return out_path + ".in_progress.txt"


def _write_in_progress_marker(out_path: str, lines: list[str]) -> None:
    p = _in_progress_marker_path(out_path)
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    try:
        os.chmod(p, 0o644)
    except OSError:
        pass


def _clear_in_progress_marker(out_path: str) -> None:
    p = _in_progress_marker_path(out_path)
    try:
        if os.path.isfile(p):
            os.remove(p)
    except OSError:
        pass


def _dataarray_to_numpy_time_chunked(
    da: xr.DataArray,
    *,
    time_chunk: int | None,
    desc: str,
    show_progress: bool,
) -> np.ndarray:
    """
    Materialize ``da`` to numpy. If ``time_chunk`` is a positive int and ``time`` is a
    dimension, load in time slices so tqdm can show progress (large Lustre reads).
    """
    if time_chunk is None or int(time_chunk) <= 0:
        return np.asarray(da.values)
    if "time" not in da.dims:
        return np.asarray(da.values)
    nt = int(da.sizes["time"])
    if nt <= int(time_chunk):
        return np.asarray(da.values)
    blocks: list[np.ndarray] = []
    step = int(time_chunk)
    rng = range(0, nt, step)
    if show_progress and tqdm is not None:
        rng = tqdm(
            rng,
            total=(nt + step - 1) // step,
            desc=desc,
            unit="time_blk",
        )
    for t0 in rng:
        t1 = min(t0 + step, nt)
        blocks.append(np.asarray(da.isel(time=slice(t0, t1)).values))
    return np.concatenate(blocks, axis=0)


def create_bmu_bootstrap_waves_winds_for_merged_sites(
    *,
    hindcast_waves_winds_dir: str,
    simulated_daily_bmus: xr.Dataset,
    waveclusters_and_pcs: pd.DataFrame | str,
    output_dir: str,
    monthly_conditioning: bool = True,
    bmu_col: str = "kma_bmus",
    sim_idx: int = 0,
    sim_indices: Iterable[int] | int | None = None,
    seed: int = 0,
    seed_per_sim: bool = False,
    resample_rule: str = "1D",
    time_start: str | None = None,
    time_end: str | None = None,
    exclude_files: Iterable[str] = (),
    show_progress: bool = True,
    progress_desc: str = "BMU bootstrap",
    show_progress_per_variable: bool = False,
    skip_existing_outputs: bool = True,
    atomic_write: bool = True,
    site_checkpoint_every: int | None = None,
    resume_site_checkpoints: bool = False,
    materialize_time_chunk: int | None = 512,
    write_trace_vars: bool = True,
    include_variables: Iterable[str] | None = None,
    selected_variables: Iterable[str] | str | None = None,
    selected_site_indices: Iterable[int] | int | None = None,
    selected_coordinate: tuple[float, float] | None = None,
) -> SiteSelection:
    """
    Create BMU- (and month-) conditioned bootstrap time series for ALL NetCDF variables
    in ``hindcast_waves_winds_dir`` using the file-native site grid.

    Each input file (e.g. those in ``inputs/merged_500m``) is expected to already store
    one variable on the desired site grid with dims ``(time, site)`` and ``lat`` / ``lon``
    as per-site data variables, so no GeoJSON file or distance matching is required: by
    default every site in each file is bootstrapped. Site selection can still be
    narrowed with ``selected_coordinate`` (nearest single site) or
    ``selected_site_indices`` (integer indices into the file's ``site`` dimension).

    Logic matches the plots:
    - historical pool: daily-mean hindcast values grouped by historical BMU (and month)
    - simulated series: for each simulated day, pick a random historical day from the same group

    Outputs:
    - writes one NetCDF per input file into ``output_dir``, subset to the selected sites
      with coords: time, site, lat, lon and a single data variable (same name as input)
    - each input ``*.nc`` is one output file (typically one physical variable per file).
    - if ``skip_existing_outputs=True``, existing destination files are skipped so a
      restarted run continues after the last completed variable.
    - if ``atomic_write=True`` (default), each file is written to a sibling ``*.tmp``
      then renamed, so a crash during write does not leave a truncated NetCDF with the
      final name.
    - if ``site_checkpoint_every`` is a positive int (e.g. 100), bootstrap output for each
      variable is built in site strips of that width. Progress is flushed to sidecar files
      next to the intended output (``*.site_ckpt.json``, ``*.site_ckpt.pick_idx.npy``,
      ``*.site_ckpt.dat`` memmap). Set ``resume_site_checkpoints=True`` to continue a
      partially built variable after a crash (same parameters and hindcast data required).
    - ``materialize_time_chunk``: if a positive integer, the hindcast slice aligned to
      historical BMUs is read from disk in blocks of that many **time** steps and a
      tqdm bar shows ``time_blk`` progress (the slow step before any ``.site_ckpt`` files
      appear). Use ``None`` or ``0`` for a single in-memory read (often faster, no bar).

    Time window:
    - time_start/time_end filter the simulated BMU series (and therefore the output time axis).
    """
    hindcast_waves_winds_dir = os.path.abspath(os.path.expanduser(hindcast_waves_winds_dir))
    # Allow passing a single NetCDF file path for convenience (folder is its parent)
    single_only_basename: str | None = None
    if os.path.isfile(hindcast_waves_winds_dir) and hindcast_waves_winds_dir.endswith(".nc"):
        single_nc = hindcast_waves_winds_dir
        hindcast_waves_winds_dir = os.path.dirname(single_nc)
        single_only_basename = os.path.basename(single_nc)
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    # simulated BMUs are 1..K in ALR output; convert to 0..K-1 like historical kma_bmus
    if sim_indices is None:
        sim_indices = (int(sim_idx),)
    else:
        if isinstance(sim_indices, (int, np.integer)):
            sim_indices = (int(sim_indices),)
        else:
            sim_indices = tuple(int(i) for i in sim_indices)
        if len(sim_indices) == 0:
            raise ValueError("sim_indices must not be empty")

    sim_all = simulated_daily_bmus["evbmus_sims"].isel(n_sim=list(sim_indices))
    sim_time = pd.to_datetime(sim_all["time"].values).floor("D")
    sim_month = pd.to_datetime(sim_time).month
    # Force deterministic dim order: (time, n_sim)
    sim_all_t = sim_all.transpose("time", "n_sim")
    sim_cluster_all = sim_all_t.values.astype(int) - 1
    if time_start is not None or time_end is not None:
        t0 = pd.Timestamp(time_start) if time_start is not None else sim_time.min()
        t1 = pd.Timestamp(time_end) if time_end is not None else sim_time.max()
        mask = (sim_time >= t0) & (sim_time <= t1)
        sim_time = sim_time[mask]
        sim_cluster_all = sim_cluster_all[mask, :]
        sim_month = sim_month[mask]
    if len(sim_time) == 0:
        raise ValueError(
            "No simulated BMU times selected. "
            f"Check `time_start`/`time_end` ({time_start}..{time_end}) against "
            f"`simulated_daily_bmus` time range ({pd.to_datetime(sim_all['time'].values).min()}.."
            f"{pd.to_datetime(sim_all['time'].values).max()})."
        )

    hist_bmu_da = _load_hist_bmus(waveclusters_and_pcs, bmu_col=bmu_col)
    hist_time = pd.to_datetime(hist_bmu_da["time"].values).floor("D")

    if selected_variables is None:
        selected_variables = include_variables
    if isinstance(selected_variables, str):
        selected_variables = [selected_variables]
    include_set = None if selected_variables is None else {str(v).lower() for v in selected_variables}
    exclude_set = set(exclude_files or ())
    if single_only_basename is not None:
        files = [single_only_basename]
    else:
        files = sorted(
            [f for f in os.listdir(hindcast_waves_winds_dir) if f.endswith(".nc") and f not in exclude_set]
        )
    if include_set is not None:
        # Files are typically named like "{var}_500m.nc" or "{var}_NorthCarolina.nc"
        before_filter = list(files)
        files = [f for f in files if os.path.splitext(f)[0].split("_", 1)[0].lower() in include_set]
        matched_prefixes = {
            os.path.splitext(f)[0].split("_", 1)[0].lower() for f in files
        }
        missing_requested = sorted(include_set - matched_prefixes)
        if missing_requested:
            print(
                f"Warning: {len(missing_requested)} requested variable(s) have no hindcast "
                f"file in {hindcast_waves_winds_dir} (basename prefix before '_', e.g. "
                f"ptp0_500m.nc -> ptp0): {missing_requested}",
                flush=True,
            )
        if not files and before_filter:
            raise FileNotFoundError(
                f"No .nc files left after `selected_variables` filter in {hindcast_waves_winds_dir}. "
                f"Filter uses the part before the first underscore in each basename (e.g. "
                f"'hs_500m.nc' -> 'hs'). You passed selected_variables={selected_variables!r}. "
                f"Do not use a literal `[...]` placeholder in the notebook — use the real variable "
                f"names, e.g. ['hs','tp',...]. Example basenames present: {before_filter[:8]}{'...' if len(before_filter) > 8 else ''}"
            )
    if not files:
        raise FileNotFoundError(f"No .nc files found in {hindcast_waves_winds_dir}")

    it = files
    if show_progress and tqdm is not None:
        it = tqdm(files, desc=progress_desc, unit="file")

    first_selection: SiteSelection | None = None
    processed = 0
    skipped = 0
    skipped_existing = 0
    for fn in it:
        out_path = os.path.join(output_dir, fn)
        if skip_existing_outputs and os.path.isfile(out_path):
            try:
                if os.path.getsize(out_path) > 0:
                    skipped_existing += 1
                    _remove_site_ckpt(out_path)
                    _clear_in_progress_marker(out_path)
                    continue
            except OSError:
                pass

        src_path = os.path.join(hindcast_waves_winds_dir, fn)
        with xr.open_dataset(src_path) as ds:
            # only handle datasets that have a time dimension
            if "time" not in ds.dims and "time" not in ds.coords:
                skipped += 1
                continue

            # identify main variable: prefer one matching the filename prefix
            # (e.g. 'hs' for 'hs_500m.nc'); fall back to the first non-coord var.
            data_vars = list(ds.data_vars)
            data_vars = [v for v in data_vars if v not in {"latitude", "longitude", "lat", "lon", "projected_coordinate_system"}]
            if len(data_vars) == 0:
                skipped += 1
                continue
            prefix = os.path.splitext(fn)[0].split("_", 1)[0].lower()
            preferred = [v for v in data_vars if str(v).lower() == prefix]
            var = preferred[0] if preferred else data_vars[0]

            try:
                lat_name, lon_name = _guess_lat_lon(ds)
                site_dim = _guess_site_dim(ds, lat_name, lon_name)
            except Exception:
                # Some files may not be in point-grid format
                skipped += 1
                continue

            try:
                selection = select_all_sites_from_merged_file(
                    src_path,
                    selected_site_indices=selected_site_indices,
                    selected_coordinate=selected_coordinate,
                )
            except Exception as e:
                print(f"Skipping '{fn}': site selection failed ({e}).", flush=True)
                skipped += 1
                continue

            if first_selection is None:
                first_selection = selection
                n_total_sites = int(ds.sizes.get(site_dim, 0))
                kept = len(selection.ids)
                print(
                    f"Using {kept} / {n_total_sites} sites per file "
                    f"(no GeoJSON; all sites taken from each merged NetCDF).",
                    flush=True,
                )

            # subset to selected indices on the file's own site dim
            ds_sub = ds.isel({site_dim: xr.DataArray(selection.src_index, dims=(site_dim,))})

            _write_in_progress_marker(
                out_path,
                [
                    f"file={fn}",
                    f"var={var}",
                    f"n_sites={len(selection.src_index)}",
                    "stage=resample_daily_mean (lazy until materialize)",
                ],
            )
            print(
                f"[{fn}] Resampling to {resample_rule} means for {len(selection.src_index)} sites "
                f"(then aligning to historical BMUs). Marker: {_in_progress_marker_path(out_path)}",
                flush=True,
            )

            # daily aggregation
            da_src = ds_sub[var]
            try:
                da_daily = da_src.resample(time=resample_rule).mean()
            except RuntimeError as e:
                # On some HPC filesystems/backends (HDF5 on Lustre), lazy reads during
                # resample can trigger low-level HDF errors.
                if "HDF error" in str(e):
                    try:
                        da_daily = da_src.load().resample(time=resample_rule).mean()
                    except RuntimeError as e2:
                        if "HDF error" not in str(e2):
                            raise
                        # last resort: reopen with a different backend
                        try:
                            with xr.open_dataset(src_path, engine="h5netcdf") as ds2:
                                ds2_sub = ds2.isel(
                                    {site_dim: xr.DataArray(selection.src_index, dims=(site_dim,))}
                                )
                                da_daily = ds2_sub[var].load().resample(time=resample_rule).mean()
                        except Exception:
                            skipped += 1
                            _clear_in_progress_marker(out_path)
                            continue
                else:
                    raise
            da_daily["time"] = pd.to_datetime(da_daily["time"].values).floor("D")

            # align historical BMUs and daily hindcast values to build pools
            da_hist = da_daily.sel(time=hist_time, drop=True)
            # handle any missing times gracefully (numpy day precision)
            da_hist_days = pd.to_datetime(da_hist["time"].values).to_numpy(dtype="datetime64[D]")
            hist_days = pd.to_datetime(hist_time).to_numpy(dtype="datetime64[D]")
            common_time = np.intersect1d(da_hist_days, hist_days)
            if common_time.size == 0:
                _clear_in_progress_marker(out_path)
                continue
            da_hist = da_hist.sel(time=common_time)
            hist_sel = hist_bmu_da.sel(time=common_time)

            _write_in_progress_marker(
                out_path,
                [
                    f"file={fn}",
                    f"var={var}",
                    f"n_hist_times={len(common_time)}",
                    f"n_sites={da_hist.sizes.get('site', da_hist.sizes.get(site_dim, '?'))}",
                    "stage=materialize_hindcast_numpy",
                ],
            )
            show_mat = bool(show_progress or show_progress_per_variable)
            values_hist = _dataarray_to_numpy_time_chunked(
                da_hist,
                time_chunk=materialize_time_chunk,
                desc=f"{fn} hindcast→RAM",
                show_progress=show_mat,
            )
            if values_hist.ndim == 1:
                # if single site, promote to (time, site)
                values_hist = values_hist[:, None]
            hist_cluster = np.asarray(hist_sel.values).astype(int)
            hist_month = pd.to_datetime(common_time).month

            # simulate: one output series per simulation member
            n_time = len(sim_time)
            n_sim = sim_cluster_all.shape[1]
            n_site = int(values_hist.shape[1])

            if site_checkpoint_every is not None:
                if int(site_checkpoint_every) <= 0:
                    raise ValueError("site_checkpoint_every must be None or a positive integer.")
                ck_every = int(site_checkpoint_every)
                meta_path, pick_path, dat_path = _site_ckpt_paths(out_path)

                if not resume_site_checkpoints and any(os.path.isfile(p) for p in (meta_path, pick_path, dat_path)):
                    print(
                        f"Removing stale site checkpoint for '{fn}' (resume_site_checkpoints=False)."
                    )
                    _remove_site_ckpt(out_path)

                meta_expected = {
                    "schema": 2,
                    "src_basename": fn,
                    "n_time": n_time,
                    "n_site": n_site,
                    "n_sim": n_sim,
                    "n_hist_rows": int(values_hist.shape[0]),
                    "sim_indices": list(sim_indices),
                    "seed": int(seed),
                    "seed_per_sim": bool(seed_per_sim),
                    "monthly_conditioning": bool(monthly_conditioning),
                    "time_start": time_start,
                    "time_end": time_end,
                    "resample_rule": str(resample_rule),
                    "bmu_col": str(bmu_col),
                    "selection_kind": "merged_file",
                    "selected_site_indices": (
                        None
                        if selected_site_indices is None
                        else (
                            [int(selected_site_indices)]
                            if isinstance(selected_site_indices, (int, np.integer))
                            else [int(x) for x in selected_site_indices]
                        )
                    ),
                    "selected_coordinate": (
                        None if selected_coordinate is None
                        else [float(selected_coordinate[0]), float(selected_coordinate[1])]
                    ),
                }

                pick_idx: np.ndarray | None = None
                sites_done = 0
                mm: np.memmap | None = None

                if (
                    resume_site_checkpoints
                    and os.path.isfile(meta_path)
                    and os.path.isfile(pick_path)
                    and os.path.isfile(dat_path)
                ):
                    with open(meta_path, "r") as f:
                        meta_disk = json.load(f)
                    sites_done = int(meta_disk.get("sites_completed", 0))
                    meta_cmp = {k: v for k, v in meta_disk.items() if k != "sites_completed"}
                    if meta_cmp != meta_expected:
                        print(
                            f"Site checkpoint metadata mismatch for '{fn}'; restarting from scratch. "
                            f"(Expected keys/values differ from disk.)"
                        )
                        _remove_site_ckpt(out_path)
                        sites_done = 0
                    else:
                        pick_idx = np.load(pick_path)
                        if pick_idx.shape != (n_time, n_sim):
                            print(f"pick_idx shape mismatch for '{fn}'; restarting site checkpoint.")
                            _remove_site_ckpt(out_path)
                            pick_idx = None
                            sites_done = 0
                        else:
                            mm = np.memmap(dat_path, dtype=np.float64, mode="r+", shape=(n_time, n_site, n_sim))
                            if sites_done > n_site:
                                sites_done = 0
                            print(
                                f"Resuming '{fn}' from site {sites_done}/{n_site} "
                                f"(checkpoint every {ck_every} sites)."
                            )

                if pick_idx is None:
                    pick_idx = np.empty((n_time, n_sim), dtype=np.int64)
                    for j in range(n_sim):
                        pick_idx[:, j] = bmu_monthly_bootstrap_hist_row_indices(
                            hist_cluster,
                            hist_month,
                            sim_cluster_all[:, j],
                            sim_month,
                            monthly_conditioning=monthly_conditioning,
                            seed=(int(seed) + int(sim_indices[j])) if seed_per_sim else int(seed),
                            show_progress=bool(show_progress_per_variable),
                            progress_desc=f"{var} pick sim{sim_indices[j]} ({len(selection.ids)} sites)",
                        )
                    pick_tmp = pick_path.replace(".npy", "._tmp_save.npy", 1)
                    np.save(pick_tmp, pick_idx)
                    os.replace(pick_tmp, pick_path)

                if mm is None:
                    mm = np.memmap(dat_path, dtype=np.float64, mode="w+", shape=(n_time, n_site, n_sim))
                    mm[:] = np.nan
                    mm.flush()
                    _atomic_write_json(meta_path, {**meta_expected, "sites_completed": 0})
                    _write_in_progress_marker(
                        out_path,
                        [
                            f"file={fn}",
                            f"var={var}",
                            f"n_site={n_site}",
                            f"site_ckpt_every={ck_every}",
                            "stage=site_chunks_memmap (see .site_ckpt.*)",
                        ],
                    )

                site_bar = None
                show_sites = bool(show_progress or show_progress_per_variable)
                if show_sites and tqdm is not None:
                    site_bar = tqdm(
                        total=n_site,
                        initial=sites_done,
                        unit="site",
                        desc=f"{fn} {var}",
                        smoothing=0.05,
                    )
                for s0 in range(sites_done, n_site, ck_every):
                    s1 = min(s0 + ck_every, n_site)
                    for j in range(n_sim):
                        pj = pick_idx[:, j]
                        valid = pj >= 0
                        block = np.full((n_time, s1 - s0), np.nan, dtype=np.float64)
                        if np.any(valid):
                            block[valid, :] = values_hist[pj[valid], s0:s1]
                        mm[:, s0:s1, j] = block
                    mm.flush()
                    _atomic_write_json(meta_path, {**meta_expected, "sites_completed": int(s1)})
                    if site_bar is not None:
                        site_bar.update(s1 - s0)
                        site_bar.set_postfix_str(f"ckpt {s1}/{n_site}")
                if site_bar is not None:
                    site_bar.close()

                out_vals = np.asarray(mm, dtype=np.float64)
                del mm
            else:
                out_vals = np.full((n_time, n_site, n_sim), np.nan, dtype=float)
                for j in range(n_sim):
                    out_vals[:, :, j] = bmu_monthly_bootstrap_timeseries(
                        values_hist=values_hist,
                        hist_cluster=hist_cluster,
                        hist_month=hist_month,
                        sim_cluster=sim_cluster_all[:, j],
                        sim_month=sim_month,
                        monthly_conditioning=monthly_conditioning,
                        seed=(int(seed) + int(sim_indices[j])) if seed_per_sim else int(seed),
                        show_progress=bool(show_progress_per_variable),
                        progress_desc=f"{var} sim{sim_indices[j]} ({len(selection.ids)} sites)",
                    )

            if n_sim == 1:
                data_vars = {var: (("time", "site"), out_vals[:, :, 0])}
                coords_extra = {}
            else:
                data_vars = {var: (("time", "site", "n_sim"), out_vals)}
                coords_extra = {"n_sim": np.asarray(sim_indices, dtype=int)}

            out = xr.Dataset(
                data_vars,
                coords={
                    "time": sim_time,
                    "site": np.asarray(selection.ids, dtype="U"),
                    "lat": (("site",), np.asarray(selection.lats, dtype=float)),
                    "lon": (("site",), np.asarray(selection.lons, dtype=float)),
                    **coords_extra,
                },
                attrs={
                    "generated_by": "BMU bootstrap (random-member, merged-site grid)",
                    # netCDF4 attrs don't support boolean dtype; store as 0/1
                    "monthly_conditioning": int(bool(monthly_conditioning)),
                    "seed": int(seed),
                    "sim_indices": ",".join(str(i) for i in sim_indices),
                    "source_dir": hindcast_waves_winds_dir,
                    "source_file": fn,
                    "selection_kind": "merged_file",
                    "selected_coordinate": (
                        "" if selected_coordinate is None
                        else f"{float(selected_coordinate[0])},{float(selected_coordinate[1])}"
                    ),
                },
            )
            if write_trace_vars:
                out["distance_m"] = (("site",), np.asarray(selection.distance_m, dtype=float))
                out["src_index"] = (("site",), np.asarray(selection.src_index, dtype=int))

            try:
                if atomic_write:
                    tmp_path = out_path + ".tmp"
                    out.to_netcdf(tmp_path, mode="w")
                    os.replace(tmp_path, out_path)
                else:
                    out.to_netcdf(out_path, mode="w")
            except Exception:
                if site_checkpoint_every is not None:
                    meta_hint, _, _ = _site_ckpt_paths(out_path)
                    print(
                        f"NetCDF write failed for '{fn}'; leaving site checkpoint sidecars next to output "
                        f"(see '{meta_hint}') for resume with resume_site_checkpoints=True."
                    )
                raise
            else:
                _remove_site_ckpt(out_path)
                _clear_in_progress_marker(out_path)
                processed += 1

    parts = [f"Wrote {processed} files to '{output_dir}'", f"skipped (unreadable/format) {skipped}"]
    if skip_existing_outputs and skipped_existing:
        parts.append(f"skipped (already present) {skipped_existing}")
    print("Finished. " + ". ".join(parts) + ".")

    if first_selection is None:
        # nothing got processed; return an empty selection so callers can still introspect
        return SiteSelection(
            ids=[],
            lats=np.empty(0, dtype=float),
            lons=np.empty(0, dtype=float),
            src_index=np.empty(0, dtype=int),
            distance_m=np.empty(0, dtype=float),
        )
    return first_selection


def diagnose_bmu_bootstrap_alignment(
    *,
    hindcast_nc_path: str,
    generated_nc_path: str,
    simulated_daily_bmus: xr.Dataset,
    waveclusters_and_pcs: pd.DataFrame | str,
    variable: str = "hs",
    point_lat: float,
    point_lon: float,
    monthly_conditioning: bool = True,
    bmu_col: str = "kma_bmus",
    sim_indices: Iterable[int] | int | None = None,
    max_sims: int | None = 20,
) -> pd.DataFrame:
    """
    Debug helper to verify the bootstrap logic is sampling from the intended historical pools.

    It compares, for each (month, BMU), the historical pool distribution of `variable`
    against the generated values distribution conditioned on the simulated BMU sequence.

    Returns a DataFrame with pool/gen counts and summary stats per (month, bmu).
    """

    hindcast_nc_path = os.path.abspath(os.path.expanduser(hindcast_nc_path))
    generated_nc_path = os.path.abspath(os.path.expanduser(generated_nc_path))

    # Load generated site and choose nearest to target coordinate
    with xr.open_dataset(generated_nc_path) as ds_gen:
        if variable not in ds_gen.data_vars:
            raise KeyError(f"'{variable}' not found in generated file. Vars={list(ds_gen.data_vars)}")
        if "site" not in ds_gen.dims:
            raise KeyError(f"Generated file missing 'site' dim. Dims={dict(ds_gen.dims)}")
        if "time" not in ds_gen.dims:
            raise KeyError(f"Generated file missing 'time' dim. Dims={dict(ds_gen.dims)}")

        lat_name_g, lon_name_g = _guess_lat_lon(ds_gen)
        glats = ds_gen[lat_name_g].values.astype(float)
        glons = ds_gen[lon_name_g].values.astype(float)
        gen_site_idx = int(np.argmin(haversine_m(point_lat, point_lon, glats, glons) / 1000.0))
        gen_lat = float(glats[gen_site_idx])
        gen_lon = float(glons[gen_site_idx])

        gen_time = pd.to_datetime(ds_gen["time"].values).floor("D")
        gen_month = gen_time.month.values
        da_gen = ds_gen[variable].isel(site=gen_site_idx)
        if "n_sim" in da_gen.dims:
            gen_vals = da_gen.transpose("time", "n_sim").values
            gen_nsim = gen_vals.shape[1]
        else:
            gen_vals = da_gen.values.reshape((-1, 1))
            gen_nsim = 1

    # Historical daily values at nearest hindcast point to generated point
    with xr.open_dataset(hindcast_nc_path) as ds_hc:
        if variable not in ds_hc.data_vars:
            raise KeyError(f"'{variable}' not found in hindcast file. Vars={list(ds_hc.data_vars)}")
        lat_name_h, lon_name_h = _guess_lat_lon(ds_hc)
        site_dim_h = _guess_site_dim(ds_hc, lat_name_h, lon_name_h)
        hlats = ds_hc[lat_name_h].values.astype(float)
        hlons = ds_hc[lon_name_h].values.astype(float)
        hc_site_idx = int(np.argmin(haversine_m(gen_lat, gen_lon, hlats, hlons) / 1000.0))
        da_hc = ds_hc[variable].isel({site_dim_h: hc_site_idx})
        # daily aggregation same as generator (scalar mean; directional not handled here)
        da_hc_daily = da_hc.resample(time="1D").mean()
        da_hc_daily["time"] = pd.to_datetime(da_hc_daily["time"].values).floor("D")

    # historical BMUs
    hist_bmu_da = _load_hist_bmus(waveclusters_and_pcs, bmu_col=bmu_col)
    hist_time = pd.to_datetime(hist_bmu_da["time"].values).floor("D")

    # align hindcast daily values with historical BMUs by common days
    da_hist = da_hc_daily.sel(time=hist_time, drop=True)
    da_hist_days = pd.to_datetime(da_hist["time"].values).to_numpy(dtype="datetime64[D]")
    hist_days = pd.to_datetime(hist_time).to_numpy(dtype="datetime64[D]")
    common_days = np.intersect1d(da_hist_days, hist_days)
    if common_days.size == 0:
        raise ValueError("No overlapping days between hindcast daily series and historical BMU series.")
    da_hist = da_hist.sel(time=common_days)
    hist_sel = hist_bmu_da.sel(time=common_days)

    hist_vals = np.asarray(da_hist.values).astype(float)
    hist_bmu = np.asarray(hist_sel.values).astype(int)
    hist_month = pd.to_datetime(common_days).month

    # simulated BMUs for the generated time axis (must align to gen_time)
    sim_all = simulated_daily_bmus["evbmus_sims"].transpose("time", "n_sim")
    sim_time = pd.to_datetime(sim_all["time"].values).floor("D")
    # use only times present in generated output
    # (generated time is already the selected window)
    sim_mask = np.isin(sim_time.to_numpy(dtype="datetime64[D]"), gen_time.to_numpy(dtype="datetime64[D]"))
    if not np.any(sim_mask):
        raise ValueError("No overlap between simulated_daily_bmus time and generated time axis.")
    sim_all = sim_all.isel(time=np.where(sim_mask)[0])
    sim_time2 = pd.to_datetime(sim_all["time"].values).floor("D")
    # ensure same ordering as generated time
    order = pd.Index(sim_time2).get_indexer(pd.Index(gen_time))
    if np.any(order < 0):
        raise ValueError("Could not align simulated BMUs to generated time (missing days).")
    sim_all = sim_all.isel(time=order)

    if sim_indices is None:
        sim_indices = list(range(sim_all.sizes["n_sim"]))
    elif isinstance(sim_indices, (int, np.integer)):
        sim_indices = [int(sim_indices)]
    else:
        sim_indices = [int(i) for i in sim_indices]
    if max_sims is not None:
        sim_indices = sim_indices[: int(max_sims)]

    sim_bmu = (sim_all.isel(n_sim=sim_indices).values.astype(int) - 1)  # (time, n_sim_subset)
    if sim_bmu.shape[0] != gen_vals.shape[0]:
        raise ValueError(
            f"Time length mismatch: gen time={gen_vals.shape[0]} vs sim time={sim_bmu.shape[0]}"
        )

    # summarize per (month, bmu)
    rows = []
    months = range(1, 13) if monthly_conditioning else [0]
    for m in months:
        if monthly_conditioning:
            hist_mask_m = hist_month == m
            gen_mask_m = gen_month == m
        else:
            hist_mask_m = np.ones_like(hist_month, dtype=bool)
            gen_mask_m = np.ones_like(gen_month, dtype=bool)

        bmus = np.unique(sim_bmu[gen_mask_m, :])
        for cid in bmus:
            cid = int(cid)
            pool = hist_vals[hist_mask_m & (hist_bmu == cid)]
            pool = pool[np.isfinite(pool)]

            # generated values across chosen sims/time
            mask_sim = (sim_bmu == cid) & gen_mask_m[:, None]
            gen_samples = gen_vals[mask_sim]
            gen_samples = gen_samples[np.isfinite(gen_samples)]

            if pool.size == 0 and gen_samples.size == 0:
                continue

            def qstats(a):
                if a.size == 0:
                    return (np.nan, np.nan, np.nan)
                return (float(np.nanmean(a)), float(np.nanpercentile(a, 10)), float(np.nanpercentile(a, 90)))

            pool_mean, pool_p10, pool_p90 = qstats(pool)
            gen_mean, gen_p10, gen_p90 = qstats(gen_samples)
            rows.append(
                {
                    "month": int(m) if monthly_conditioning else None,
                    "bmu": cid,
                    "pool_n": int(pool.size),
                    "gen_n": int(gen_samples.size),
                    "pool_mean": pool_mean,
                    "pool_p10": pool_p10,
                    "pool_p90": pool_p90,
                    "gen_mean": gen_mean,
                    "gen_p10": gen_p10,
                    "gen_p90": gen_p90,
                    "mean_diff": float(gen_mean - pool_mean) if np.isfinite(gen_mean) and np.isfinite(pool_mean) else np.nan,
                }
            )

    df = pd.DataFrame(rows).sort_values(["month", "bmu"]).reset_index(drop=True)
    print(
        f"Diagnose bootstrap alignment for {variable} at target=({point_lat},{point_lon}). "
        f"Selected gen_site_idx={gen_site_idx} (lat/lon={gen_lat:.4f},{gen_lon:.4f}), "
        f"hindcast_site_idx={hc_site_idx}. Using {len(sim_indices)} sims (of {gen_nsim} in file)."
    )
    if not df.empty:
        print("Top |mean_diff| rows:")
        print(df.reindex(df["mean_diff"].abs().sort_values(ascending=False).head(10).index))
    return df

