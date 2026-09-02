"""Splice cyclone peaks into earth3_veg climate series — all sites, all shared vars.

Same rules as the single-site splice:
  - On ``time_emulation`` day, replace values at every climate site.
  - HS: max over the 24 cyclone hours.
  - Companion vars: value at the hour of max HS.
  - Skip a site for that cyclone if HS peak < that site's original HS minimum
    (HS mins taken from ``hs_500m.nc``; sites only in partition files use the
    nearest HS climate site).
  - If several cyclones share a day, keep the higher HS peak *per site*.

Work is done on the cyclone site axis (1356), then scattered onto each climate
file's own sites (1269 for bulk vars, 1302 for some partitions).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

DEFAULT_PROJECT = Path(__file__).resolve().parents[1]

import sys

if str(DEFAULT_PROJECT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_PROJECT))
from paths import earth3_baseline_dir, predicted_emu_tracks  # noqa: E402

# climate_file_stem -> (climate_data_var_name, cyclone_var_name, role)
# role: "peak_hs" = replace with max-HS; "at_peak" = value at hour of max HS
VAR_SPEC: dict[str, tuple[str, str, str]] = {
    "hs": ("hs", "hs", "peak_hs"),
    "tp": ("tp", "tp", "at_peak"),
    "tm02": ("tm02", "tm02", "at_peak"),
    "dp": ("dp", "dp", "at_peak"),
    "dm": ("dm", "dm", "at_peak"),
    "phs0": ("hs", "phs0", "at_peak"),
    "ptp0": ("tp", "ptp0", "at_peak"),
    "dp0": ("dp", "pdp0", "at_peak"),
    "phs1": ("hs", "phs1", "at_peak"),
    "ptp1": ("tp", "ptp1", "at_peak"),
    "dp1": ("dp", "pdp1", "at_peak"),
    "spr1": ("spr", "spr1", "at_peak"),
}

SCENARIOS = {
    "ssp245": {
        "original_dir": earth3_baseline_dir("ssp245"),
        "cyclone_dir": DEFAULT_PROJECT / "outputs" / "merged_cyclones_earh3_245",
        "catalog_nc": predicted_emu_tracks("ssp245", mf=True),
        "output_dir": DEFAULT_PROJECT / "outputs" / "earth3_ssp245_with_cyclones",
    },
    "ssp585": {
        "original_dir": earth3_baseline_dir("ssp585"),
        "cyclone_dir": DEFAULT_PROJECT / "outputs" / "merged_cyclones_earth3_585",
        "catalog_nc": predicted_emu_tracks("ssp585", mf=True),
        "output_dir": DEFAULT_PROJECT / "outputs" / "earth3_ssp585_with_cyclones",
    },
}

# Companion vars that also require value >= original site minimum
_MIN_FILTER_STEMS = frozenset({"tp", "ptp0", "ptp1", "tm02"})


@dataclass(frozen=True)
class SiteMap:
    cyc_index: np.ndarray  # (n_climate,) -> cyclone site
    max_dist_deg: float


def build_site_map(
    climate_lon: np.ndarray,
    climate_lat: np.ndarray,
    cyclone_lon: np.ndarray,
    cyclone_lat: np.ndarray,
    *,
    max_allowed_deg: float = 0.05,
) -> SiteMap:
    tree = cKDTree(np.column_stack([cyclone_lon, cyclone_lat]))
    dist, idx = tree.query(np.column_stack([climate_lon, climate_lat]))
    dist = np.asarray(dist, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64)
    if float(dist.max()) > max_allowed_deg:
        raise ValueError(
            f"Climate↔cyclone site mismatch: max dist={dist.max():.3e} deg "
            f"(allowed {max_allowed_deg})"
        )
    return SiteMap(cyc_index=idx, max_dist_deg=float(dist.max()))


def _resolve_variables(
    original_dir: Path,
    variables: tuple[str, ...] | None,
) -> list[tuple[str, str, str, str]]:
    """Return list of (stem, climate_var, cyclone_var, role) present on disk."""
    stems = list(VAR_SPEC) if variables is None else list(variables)
    out: list[tuple[str, str, str, str]] = []
    for stem in stems:
        if stem not in VAR_SPEC:
            raise KeyError(f"Unknown variable stem {stem!r}; known={sorted(VAR_SPEC)}")
        clim_var, cyc_var, role = VAR_SPEC[stem]
        path = original_dir / f"{stem}_500m.nc"
        if not path.is_file():
            print(f"  skip missing climate file: {path.name}")
            continue
        out.append((stem, clim_var, cyc_var, role))
    return out


def _load_catalog(catalog_nc: Path) -> tuple[np.ndarray, pd.DatetimeIndex]:
    with xr.open_dataset(catalog_nc) as cat:
        case_nums = np.asarray(cat.case_num.values)
        time_emul = pd.to_datetime(np.asarray(cat.time_emulation.values)).normalize()
    return case_nums, pd.DatetimeIndex(time_emul)


def _process_one_cyclone(
    cyc_path: str,
    cyc_vars: tuple[str, ...],
) -> dict[str, np.ndarray] | None:
    """Load one cyclone; return (24, n_cyc_site) arrays keyed by cyclone var name."""
    path = Path(cyc_path)
    if not path.is_file():
        return None
    with xr.open_dataset(path) as ds:
        missing = [v for v in cyc_vars if v not in ds]
        if missing:
            return None
        return {v: np.asarray(ds[v].values, dtype=np.float32) for v in cyc_vars}


def _hs_min_on_cyclone_axis(
    hs_min_climate: np.ndarray,
    hs_map: SiteMap,
    n_cyc: int,
    cyclone_lon: np.ndarray,
    cyclone_lat: np.ndarray,
    climate_lon: np.ndarray,
    climate_lat: np.ndarray,
) -> np.ndarray:
    """HS minimum on cyclone sites; fill unmapped sites from nearest climate HS site."""
    hs_min_cyc = np.full(n_cyc, np.nan, dtype=np.float64)
    hs_min_cyc[hs_map.cyc_index] = hs_min_climate

    missing = ~np.isfinite(hs_min_cyc)
    if missing.any():
        tree = cKDTree(np.column_stack([climate_lon, climate_lat]))
        dist, idx = tree.query(np.column_stack([cyclone_lon[missing], cyclone_lat[missing]]))
        hs_min_cyc[np.flatnonzero(missing)] = hs_min_climate[idx]
        print(
            f"  filled HS-min for {int(missing.sum())} cyclone sites "
            f"not in hs_500m (max dist={float(np.max(dist)):.3e} deg)"
        )
    return hs_min_cyc


def splice_all_sites_with_cyclones(
    *,
    scenario: str = "ssp245",
    original_dir: Path | None = None,
    cyclone_dir: Path | None = None,
    catalog_nc: Path | None = None,
    output_dir: Path | None = None,
    variables: tuple[str, ...] | None = None,
    skip_existing: bool = False,
    n_jobs: int = 1,
    max_cyclones: int | None = None,
) -> dict:
    """Copy climate ``{var}_500m.nc`` and splice cyclones at every site."""
    if scenario not in SCENARIOS:
        raise KeyError(f"scenario must be one of {sorted(SCENARIOS)}")
    cfg = SCENARIOS[scenario]
    original_dir = Path(original_dir or cfg["original_dir"])
    cyclone_dir = Path(cyclone_dir or cfg["cyclone_dir"])
    catalog_nc = Path(catalog_nc or cfg["catalog_nc"])
    output_dir = Path(output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    var_list = _resolve_variables(original_dir, variables)
    stems = [v[0] for v in var_list]
    if "hs" not in stems:
        raise ValueError("hs must be included (needed for peak-hour / min filter)")

    hs_path = original_dir / "hs_500m.nc"
    with xr.open_dataset(hs_path) as base:
        climate_lon = np.asarray(base.lon.values, dtype=np.float64)
        climate_lat = np.asarray(base.lat.values, dtype=np.float64)
        time_index = pd.DatetimeIndex(base.time.values).normalize()
        hs_clim = np.asarray(base.hs.values, dtype=np.float32)
        hs_min_climate = np.nanmin(hs_clim, axis=0).astype(np.float64)
        del hs_clim

    sample = next(cyclone_dir.glob("cyclone_*_merged_allvars_ptm4.nc"), None)
    if sample is None:
        raise FileNotFoundError(f"No cyclone files in {cyclone_dir}")
    with xr.open_dataset(sample) as cyc0:
        cyclone_lon = np.asarray(cyc0.lon.values, dtype=np.float64)
        cyclone_lat = np.asarray(cyc0.lat.values, dtype=np.float64)
        n_cyc = int(cyc0.sizes["site"])

    hs_map = build_site_map(climate_lon, climate_lat, cyclone_lon, cyclone_lat, max_allowed_deg=1e-3)
    hs_min_cyc = _hs_min_on_cyclone_axis(
        hs_min_climate, hs_map, n_cyc, cyclone_lon, cyclone_lat, climate_lon, climate_lat
    )

    # Per-climate-file site maps (1269 vs 1302) and optional min floors
    file_maps: dict[str, SiteMap] = {}
    var_min_climate: dict[str, np.ndarray] = {}
    for stem, clim_var, _cyc_var, role in var_list:
        with xr.open_dataset(original_dir / f"{stem}_500m.nc") as ds:
            lon = np.asarray(ds.lon.values, dtype=np.float64)
            lat = np.asarray(ds.lat.values, dtype=np.float64)
            file_maps[stem] = build_site_map(lon, lat, cyclone_lon, cyclone_lat)
            if stem in _MIN_FILTER_STEMS:
                var_min_climate[stem] = np.nanmin(
                    np.asarray(ds[clim_var].values, dtype=np.float64), axis=0
                )
            print(
                f"  {stem}_500m.nc: n_site={lon.size}  "
                f"map_max_dist={file_maps[stem].max_dist_deg:.2e} deg"
            )

    print(f"[{scenario}] cyclone sites={n_cyc}  hs climate sites={climate_lon.size}")
    print(f"[{scenario}] variables: {stems}")
    print(f"[{scenario}] output_dir: {output_dir}")

    out_paths = {stem: output_dir / f"{stem}_500m.nc" for stem, *_ in var_list}
    if skip_existing and all(p.is_file() and p.stat().st_size > 1_000_000 for p in out_paths.values()):
        print(f"[{scenario}] skip_existing: all outputs present")
        return {"outputs": out_paths, "skipped_all": True}

    case_nums, time_emul = _load_catalog(catalog_nc)
    day_pos = time_index.get_indexer(time_emul)
    cyc_vars = tuple(sorted({cyc_var for _, _, cyc_var, _ in var_list}))

    day_best_hs: dict[int, np.ndarray] = {}
    day_pack: dict[int, dict[str, np.ndarray]] = {}
    n_used = 0
    n_skip = 0
    n_below = 0

    jobs: list[tuple[int, int, str]] = []
    for i, (cid, dpos) in enumerate(zip(case_nums, day_pos)):
        if dpos < 0:
            n_skip += 1
            continue
        cyc_path = cyclone_dir / f"cyclone_{int(cid)}_merged_allvars_ptm4.nc"
        if not cyc_path.is_file():
            n_skip += 1
            continue
        jobs.append((i, int(dpos), str(cyc_path)))
        if max_cyclones is not None and len(jobs) >= int(max_cyclones):
            break

    def _merge_cyclone(day_idx: int, arrs: dict[str, np.ndarray]) -> tuple[int, int]:
        hs24 = arrs["hs"]
        finite = np.isfinite(hs24)
        if not finite.any():
            return 0, 0
        hs_fill = np.where(finite, hs24, -np.inf)
        ih = np.argmax(hs_fill, axis=0)
        cols = np.arange(n_cyc)
        hs_peak = hs24[ih, cols].astype(np.float64)
        valid = np.isfinite(hs_peak) & (hs_peak >= hs_min_cyc)
        n_b = int((np.isfinite(hs_peak) & ~valid).sum())
        if not valid.any():
            return 0, n_b

        if day_idx not in day_best_hs:
            day_best_hs[day_idx] = np.full(n_cyc, -np.inf, dtype=np.float64)
            day_pack[day_idx] = {
                stem: np.full(n_cyc, np.nan, dtype=np.float32) for stem, *_ in var_list
            }

        better = valid & (hs_peak > day_best_hs[day_idx])
        if not better.any():
            return 0, n_b

        day_best_hs[day_idx][better] = hs_peak[better]
        for stem, _clim_var, cyc_var, role in var_list:
            if role == "peak_hs":
                day_pack[day_idx][stem][better] = hs_peak[better].astype(np.float32)
            else:
                vals = arrs[cyc_var][ih, cols].astype(np.float32)
                day_pack[day_idx][stem][better] = np.where(
                    np.isfinite(vals[better]), vals[better], np.nan
                )
        return int(better.sum()), n_b

    print(f"[{scenario}] cyclones to process: {len(jobs)}  (catalog skipped: {n_skip})")

    if n_jobs <= 1:
        for _, day_idx, path in tqdm(jobs, desc=f"Splice {scenario}"):
            arrs = _process_one_cyclone(path, cyc_vars)
            if arrs is None:
                n_skip += 1
                continue
            n_sites, n_b = _merge_cyclone(day_idx, arrs)
            n_below += n_b
            if n_sites:
                n_used += 1
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = {
                ex.submit(_process_one_cyclone, path, cyc_vars): day_idx
                for _, day_idx, path in jobs
            }
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Splice {scenario}"):
                day_idx = futs[fut]
                arrs = fut.result()
                if arrs is None:
                    n_skip += 1
                    continue
                n_sites, n_b = _merge_cyclone(day_idx, arrs)
                n_below += n_b
                if n_sites:
                    n_used += 1

    n_days = len(day_best_hs)
    n_site_day = int(sum(np.isfinite(p["hs"]).sum() for p in day_pack.values()))
    print(
        f"[{scenario}] cyclones contributing: {n_used}  "
        f"unique days: {n_days}  cyclone-site-day HS hits: {n_site_day}  "
        f"below-min site hits: {n_below}"
    )

    written: dict[str, Path] = {}
    for stem, clim_var, _cyc_var, role in var_list:
        src = original_dir / f"{stem}_500m.nc"
        out = out_paths[stem]
        if skip_existing and out.is_file() and out.stat().st_size > 1_000_000:
            print(f"  skip existing {out.name}")
            written[stem] = out
            continue

        print(f"  writing {out.name} …")
        with xr.open_dataset(src) as ds:
            if clim_var not in ds:
                raise KeyError(f"{src.name} missing data var {clim_var!r}")
            data = np.array(ds[clim_var].values, dtype=np.float32, copy=True)
            coords = {k: ds.coords[k] for k in ds.coords}
            other = {k: ds[k] for k in ds.data_vars if k != clim_var}
            attrs = dict(ds.attrs)

        cmap = file_maps[stem].cyc_index  # climate site -> cyclone site
        vmin = var_min_climate.get(stem)
        n_rep = 0
        for day_idx, pack in day_pack.items():
            vals_cyc = pack[stem]
            vals = vals_cyc[cmap]
            m = np.isfinite(vals)
            if vmin is not None:
                m = m & (vals.astype(np.float64) >= vmin)
            if not m.any():
                continue
            data[day_idx, m] = vals[m]
            n_rep += int(m.sum())

        da = xr.DataArray(
            data,
            dims=("time", "site"),
            coords={"time": coords["time"], "site": coords["site"]},
        )
        out_ds = xr.Dataset({clim_var: da, **other})
        for k, v in coords.items():
            out_ds = out_ds.assign_coords({k: v})
        out_ds.attrs.update(attrs)
        out_ds.attrs["spliced"] = "all_sites"
        out_ds.attrs["scenario"] = scenario
        out_ds.attrs["cyclone_dir"] = str(cyclone_dir)
        out_ds.attrs["cyclone_agg"] = "max_of_24h" if role == "peak_hs" else "at_hour_of_max_hs"
        out_ds.attrs["n_cyclones_contributing"] = int(n_used)
        out_ds.attrs["n_unique_days"] = int(n_days)
        out_ds.attrs["n_site_day_replaced"] = int(n_rep)
        out_ds.attrs["history"] = (
            f"Copied from {src.name}; all sites: cyclone values on time_emulation days "
            f"(HS=max 24h; others at hour of max HS; skip if HS peak < site HS min)."
        )

        tmp = out.with_suffix(".nc.tmp")
        tmp.unlink(missing_ok=True)
        out_ds.to_netcdf(tmp)
        tmp.replace(out)
        written[stem] = out
        print(f"    -> {out}  ({n_rep} site-days replaced)")

    return {
        "outputs": written,
        "scenario": scenario,
        "n_cyclones": n_used,
        "n_days": n_days,
        "n_site_day_hs": n_site_day,
        "hs_map_max_dist_deg": hs_map.max_dist_deg,
    }


def validate_against_single_site(
    *,
    scenario: str = "ssp245",
    all_sites_hs: Path | None = None,
    single_site_hs: Path | None = None,
    target_lon: float = -75.7316,
    target_lat: float = 36.1942,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> dict:
    """Compare all-sites HS at the diagnostic point vs legacy single-site file."""
    cfg = SCENARIOS[scenario]
    all_sites_hs = Path(all_sites_hs or (cfg["output_dir"] / "hs_500m.nc"))
    if single_site_hs is None:
        single_site_hs = DEFAULT_PROJECT / "outputs" / (
            "hs_500m_with_cyclones.nc"
            if scenario == "ssp245"
            else "hs_500m_with_cyclones_ssp585.nc"
        )
    single_site_hs = Path(single_site_hs)

    with xr.open_dataset(all_sites_hs) as a, xr.open_dataset(single_site_hs) as b:
        lon = np.asarray(a.lon.values, dtype=np.float64)
        lat = np.asarray(a.lat.values, dtype=np.float64)
        i = int(np.argmin((lon - target_lon) ** 2 + (lat - target_lat) ** 2))
        ya = np.asarray(a.hs.isel(site=i).values, dtype=np.float64)
        yb = np.asarray(b.hs.isel(site=i).values, dtype=np.float64)

    if ya.shape != yb.shape:
        raise ValueError(f"shape mismatch {ya.shape} vs {yb.shape}")
    ok = np.allclose(ya, yb, rtol=rtol, atol=atol, equal_nan=True)
    max_abs = float(np.nanmax(np.abs(ya - yb)))
    n_diff = int(np.nansum(~np.isclose(ya, yb, rtol=rtol, atol=atol, equal_nan=True)))
    print(
        f"validate site {i} ({lon[i]:.5f},{lat[i]:.5f}): "
        f"allclose={ok}  max_abs={max_abs:.6g}  n_diff={n_diff}"
    )
    return {"ok": ok, "site": i, "max_abs": max_abs, "n_diff": n_diff}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Splice cyclones into climate series (all sites)")
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default="ssp245")
    p.add_argument("--both", action="store_true", help="Run ssp245 and ssp585")
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--max-cyclones", type=int, default=None, help="Debug: only first N cyclones")
    p.add_argument(
        "--variables",
        nargs="*",
        default=None,
        help="Optional subset of stems (default: all shared vars)",
    )
    p.add_argument("--validate", action="store_true", help="Compare HS at diagnostic point")
    args = p.parse_args()

    scenarios = ["ssp245", "ssp585"] if args.both else [args.scenario]
    vars_tuple = tuple(args.variables) if args.variables else None
    for sc in scenarios:
        splice_all_sites_with_cyclones(
            scenario=sc,
            variables=vars_tuple,
            skip_existing=args.skip_existing,
            n_jobs=args.n_jobs,
            max_cyclones=args.max_cyclones,
        )
        if args.validate:
            validate_against_single_site(scenario=sc)
