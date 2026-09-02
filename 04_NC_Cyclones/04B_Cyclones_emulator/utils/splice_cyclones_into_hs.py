"""Splice cyclone HS (daily max of 24 h) into SSP245 daily HS — one site only.

Baseline:
  03B/outputs/earth3_veg_ssp245/hs_500m.nc  (daily, 2015→2100)

For the nearest site to a target coordinate, each catalog ``case_num`` with
``time_emulation`` inside the baseline replaces that day's HS with
max(cyclone HS over the 24 hourly steps) from
  outputs/merged_cyclones_compact_ptm4_v1/cyclone_{id}_....nc

Writes a *new* file (baseline is never overwritten).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
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

DEFAULT_BASELINE = earth3_baseline_dir("ssp245") / "hs_500m.nc"
DEFAULT_ORIGINAL_DIR = earth3_baseline_dir("ssp245")
DEFAULT_CATALOG = predicted_emu_tracks("ssp245", mf=True)
DEFAULT_CYCLONE_DIR = DEFAULT_PROJECT / "outputs" / "merged_cyclones_compact_ptm4_v1"
DEFAULT_OUTPUT = DEFAULT_PROJECT / "outputs" / "hs_500m_with_cyclones.nc"
DEFAULT_TARGET_LON = -75.7316
DEFAULT_TARGET_LAT = 36.1942


def nearest_site_index(lon: np.ndarray, lat: np.ndarray, target_lon: float, target_lat: float) -> tuple[int, float]:
    d = (lon.astype(np.float64) - target_lon) ** 2 + (lat.astype(np.float64) - target_lat) ** 2
    i = int(np.argmin(d))
    return i, float(np.sqrt(d[i]))


def splice_bulk_vars_one_site_with_cyclones(
    variables: tuple[str, ...] = ("hs", "tp", "dp"),
    *,
    original_dir: Path = DEFAULT_ORIGINAL_DIR,
    catalog_nc: Path = DEFAULT_CATALOG,
    cyclone_dir: Path = DEFAULT_CYCLONE_DIR,
    output_dir: Path | None = None,
    output_suffix: str = "",
    target_lon: float = DEFAULT_TARGET_LON,
    target_lat: float = DEFAULT_TARGET_LAT,
    skip_existing: bool = False,
) -> dict:
    """Splice hs/tp/dp at one site into new ``{var}_500m_with_cyclones{suffix}.nc`` files.

    Rules (same day = ``time_emulation``):
      - hs: max of 24 h; skip if peak < original hs min
      - tp: value at hour of max hs; skip if that hs peak < hs min, or tp < original tp min
      - dp: value at hour of max hs; skip if that hs peak < hs min

    ``output_suffix`` examples: ``""`` -> ``hs_500m_with_cyclones.nc``,
    ``"_ssp585"`` -> ``hs_500m_with_cyclones_ssp585.nc``.
    """
    original_dir = Path(original_dir)
    catalog_nc = Path(catalog_nc)
    cyclone_dir = Path(cyclone_dir)
    output_dir = Path(output_dir or DEFAULT_PROJECT / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = output_suffix or ""

    # Prepare output copies
    outputs: dict[str, Path] = {}
    arrays: dict[str, np.ndarray] = {}
    coords_by_var: dict[str, dict] = {}
    data_vars_by_var: dict[str, dict] = {}
    attrs_by_var: dict[str, dict] = {}
    mins: dict[str, float] = {}
    site_i = None
    time_index = None

    for var in variables:
        src = original_dir / f"{var}_500m.nc"
        out = output_dir / f"{var}_500m_with_cyclones{suffix}.nc"
        outputs[var] = out
        if skip_existing and out.is_file() and out.stat().st_size > 1_000_000:
            print(f"Skip existing: {out}")
            continue
        print(f"Copying {src.name} -> {out.name}")
        if out.exists():
            out.unlink()
        shutil.copy2(src, out)

        with xr.open_dataset(src) as base:
            if site_i is None:
                site_i, dist = nearest_site_index(
                    np.asarray(base.lon.values), np.asarray(base.lat.values), target_lon, target_lat
                )
                time_index = pd.DatetimeIndex(base.time.values).normalize()
                print(
                    f"Target ({target_lon}, {target_lat}) -> site {site_i} "
                    f"({float(base.lon.values[site_i]):.5f}, {float(base.lat.values[site_i]):.5f}) "
                    f"dist={dist:.2e} deg"
                )
            vname = var if var in base else [x for x in base.data_vars if x.lower() == var][0]
            series = np.asarray(base[vname].isel(site=site_i).values, dtype=np.float64)
            mins[var] = float(np.nanmin(series))
            arrays[var] = np.array(base[vname].values, dtype=np.float32, copy=True)
            coords_by_var[var] = {k: base.coords[k] for k in base.coords}
            data_vars_by_var[var] = {k: base[k] for k in base.data_vars if k != vname}
            attrs_by_var[var] = dict(base.attrs)
            attrs_by_var[var]["_vname"] = vname
            print(f"  {var} original min={mins[var]:.6g}")

    if not arrays:
        return {"outputs": outputs, "skipped_all": True}

    sample = next(cyclone_dir.glob("cyclone_*_merged_allvars_ptm4.nc"), None)
    if sample is None:
        raise FileNotFoundError(f"No cyclone files in {cyclone_dir}")
    with xr.open_dataset(sample) as cyc0:
        cyc_i, _ = nearest_site_index(
            np.asarray(cyc0.lon.values), np.asarray(cyc0.lat.values), target_lon, target_lat
        )
        print(f"Cyclone site index: {cyc_i}")

    with xr.open_dataset(catalog_nc) as cat:
        case_nums = np.asarray(cat.case_num.values)
        time_emul = pd.to_datetime(np.asarray(cat.time_emulation.values)).normalize()

    assert time_index is not None and site_i is not None
    hs_min = mins.get("hs", -np.inf)

    # day -> replacement values
    day_vals: dict[pd.Timestamp, dict[str, float]] = {}
    n_used = 0
    n_below = 0

    for cid, te in tqdm(zip(case_nums, time_emul), total=len(case_nums), desc="Splice hs/tp/dp"):
        day = pd.Timestamp(te).normalize()
        if day not in time_index:
            continue
        cyc_path = cyclone_dir / f"cyclone_{int(cid)}_merged_allvars_ptm4.nc"
        if not cyc_path.is_file():
            continue
        with xr.open_dataset(cyc_path) as cyc:
            hs24 = np.asarray(cyc.hs.isel(site=cyc_i).values, dtype=np.float64)
            if not np.any(np.isfinite(hs24)):
                continue
            ih = int(np.nanargmax(hs24))
            hs_peak = float(hs24[ih])
            if hs_peak < hs_min:
                n_below += 1
                continue
            pack = {"hs": hs_peak}
            if "tp" in arrays and "tp" in cyc:
                tp_v = float(np.asarray(cyc.tp.isel(site=cyc_i).values)[ih])
                if np.isfinite(tp_v) and tp_v >= mins.get("tp", -np.inf):
                    pack["tp"] = tp_v
            if "dp" in arrays and "dp" in cyc:
                dp_v = float(np.asarray(cyc.dp.isel(site=cyc_i).values)[ih])
                if np.isfinite(dp_v):
                    pack["dp"] = dp_v

        prev = day_vals.get(day)
        if prev is None or pack["hs"] > prev["hs"]:
            day_vals[day] = pack
        n_used += 1

    print(f"Cyclones used: {n_used}  below hs min: {n_below}  unique days: {len(day_vals)}")

    for var, arr in arrays.items():
        vname = attrs_by_var[var]["_vname"]
        n_rep = 0
        for day, pack in day_vals.items():
            if var not in pack:
                continue
            pos = time_index.get_indexer([day])[0]
            if pos < 0:
                continue
            arr[pos, site_i] = np.float32(pack[var])
            n_rep += 1

        coords = coords_by_var[var]
        hs_da = xr.DataArray(arr, dims=("time", "site"), coords={"time": coords["time"], "site": coords["site"]})
        out_ds = xr.Dataset({vname: hs_da, **data_vars_by_var[var]})
        for k in coords:
            out_ds = out_ds.assign_coords({k: coords[k]})
        attrs = {k: v for k, v in attrs_by_var[var].items() if not k.startswith("_")}
        out_ds.attrs.update(attrs)
        out_ds.attrs["spliced_site_index"] = int(site_i)
        out_ds.attrs["cyclone_agg"] = "at_hour_of_max_hs" if var != "hs" else "max_of_24h"
        out_ds.attrs["hs_min_filter"] = float(hs_min)
        out_ds.attrs["n_days_replaced"] = int(n_rep)

        out = outputs[var]
        tmp = out.with_suffix(".nc.tmp")
        tmp.unlink(missing_ok=True)
        out_ds.to_netcdf(tmp)
        tmp.replace(out)
        print(f"Wrote {out} ({n_rep} days replaced for {var})")

    return {"outputs": outputs, "site_index": site_i, "n_days": len(day_vals), "n_used": n_used}


def splice_hs_one_site_with_cyclones(
    baseline_hs: Path = DEFAULT_BASELINE,
    catalog_nc: Path = DEFAULT_CATALOG,
    cyclone_dir: Path = DEFAULT_CYCLONE_DIR,
    output_nc: Path = DEFAULT_OUTPUT,
    *,
    target_lon: float = DEFAULT_TARGET_LON,
    target_lat: float = DEFAULT_TARGET_LAT,
    skip_existing: bool = False,
) -> dict:
    """Copy baseline; replace daily HS at one site with max of each cyclone's 24 h."""
    baseline_hs = Path(baseline_hs)
    catalog_nc = Path(catalog_nc)
    cyclone_dir = Path(cyclone_dir)
    output_nc = Path(output_nc)
    output_nc.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and output_nc.is_file() and output_nc.stat().st_size > 1_000_000:
        print(f"Skip existing: {output_nc}")
        return {"output": output_nc}

    print(f"Copying baseline -> {output_nc}")
    if output_nc.exists():
        output_nc.unlink()
    shutil.copy2(baseline_hs, output_nc)

    with xr.open_dataset(baseline_hs) as base:
        base_lon = np.asarray(base.lon.values)
        base_lat = np.asarray(base.lat.values)
        time_index = pd.DatetimeIndex(base.time.values).normalize()
        site_i, dist = nearest_site_index(base_lon, base_lat, target_lon, target_lat)
        hs_site = np.asarray(base.hs.isel(site=site_i).values, dtype=np.float64)
        hs_min = float(np.nanmin(hs_site))
        print(
            f"Target ({target_lon}, {target_lat}) -> baseline site {site_i} "
            f"({base_lon[site_i]:.5f}, {base_lat[site_i]:.5f}) dist={dist:.2e} deg"
        )
        print(f"Original HS min at site (filter threshold): {hs_min:.6g} m")

    # Cyclone site nearest to same coordinate
    sample = next(cyclone_dir.glob("cyclone_*_merged_allvars_ptm4.nc"), None)
    if sample is None:
        raise FileNotFoundError(f"No cyclone files in {cyclone_dir}")
    with xr.open_dataset(sample) as cyc0:
        cyc_i, cyc_dist = nearest_site_index(
            np.asarray(cyc0.lon.values), np.asarray(cyc0.lat.values), target_lon, target_lat
        )
        print(
            f"Cyclone site {cyc_i} "
            f"({float(cyc0.lon.values[cyc_i]):.5f}, {float(cyc0.lat.values[cyc_i]):.5f}) "
            f"dist={cyc_dist:.2e} deg"
        )

    with xr.open_dataset(catalog_nc) as cat:
        case_nums = np.asarray(cat.case_num.values)
        time_emul = pd.to_datetime(np.asarray(cat.time_emulation.values)).normalize()
        storm_ids = np.asarray(cat.original_storm_id.values)

    # day -> max hs if several cyclones share a day
    day_to_hs: dict[pd.Timestamp, float] = {}
    day_to_meta: dict[pd.Timestamp, list] = {}
    n_used = 0
    n_skip = 0
    n_below_min = 0

    for cid, te, sid in tqdm(
        zip(case_nums, time_emul, storm_ids),
        total=len(case_nums),
        desc="Cyclone daily max HS",
    ):
        day = pd.Timestamp(te).normalize()
        if day not in time_index:
            n_skip += 1
            continue
        cyc_path = cyclone_dir / f"cyclone_{int(cid)}_merged_allvars_ptm4.nc"
        if not cyc_path.is_file():
            n_skip += 1
            continue
        with xr.open_dataset(cyc_path) as cyc:
            series = np.asarray(cyc.hs.isel(site=cyc_i).values, dtype=np.float64)
        peak = float(np.nanmax(series))
        if not np.isfinite(peak):
            n_skip += 1
            continue
        # Do not replace with cyclone values below the original series minimum
        if peak < hs_min:
            n_below_min += 1
            continue
        prev = day_to_hs.get(day)
        if prev is None or peak > prev:
            day_to_hs[day] = peak
        day_to_meta.setdefault(day, []).append((int(cid), int(sid), peak))
        n_used += 1

    print(
        f"Cyclones applied: {n_used}  skipped: {n_skip}  "
        f"below original min: {n_below_min}  unique days replaced: {len(day_to_hs)}"
    )

    # Patch only this site in the copied file
    with xr.open_dataset(output_nc) as ds_out:
        hs = np.array(ds_out.hs.values, dtype=np.float32, copy=True)
        coords = {k: ds_out.coords[k] for k in ds_out.coords}
        data_vars = {k: ds_out[k] for k in ds_out.data_vars if k != "hs"}
        attrs = dict(ds_out.attrs)

    replaced_days = []
    for day, peak in day_to_hs.items():
        # exact day match in time_index
        pos = time_index.get_indexer([day])[0]
        if pos < 0:
            continue
        hs[pos, site_i] = np.float32(peak)
        replaced_days.append(day)

    hs_da = xr.DataArray(hs, dims=("time", "site"), coords={"time": coords["time"], "site": coords["site"]})
    out_ds = xr.Dataset({"hs": hs_da, **{k: data_vars[k] for k in data_vars}})
    for k, v in coords.items():
        if k not in out_ds.coords:
            out_ds = out_ds.assign_coords({k: v})
        else:
            out_ds = out_ds.assign_coords({k: coords[k]})
    # ensure lon/lat present
    for k in ("lon", "lat"):
        if k in coords:
            out_ds = out_ds.assign_coords({k: coords[k]})

    out_ds.attrs.update(attrs)
    out_ds.attrs["spliced_site_index"] = int(site_i)
    out_ds.attrs["spliced_target_lon"] = float(target_lon)
    out_ds.attrs["spliced_target_lat"] = float(target_lat)
    out_ds.attrs["cyclone_agg"] = "max_of_24h"
    out_ds.attrs["hs_min_filter"] = float(hs_min)
    out_ds.attrs["n_cyclones_applied"] = int(n_used)
    out_ds.attrs["n_cyclones_below_min"] = int(n_below_min)
    out_ds.attrs["n_days_replaced"] = int(len(replaced_days))
    out_ds.attrs["history"] = (
        f"Copied from {baseline_hs.name}; at site {site_i} replaced daily HS with "
        f"max(24h cyclone HS) when peak >= original min ({hs_min:.6g} m); "
        f"applied={n_used}, below_min={n_below_min}, days={len(replaced_days)}."
    )

    tmp = output_nc.with_suffix(".nc.tmp")
    tmp.unlink(missing_ok=True)
    out_ds.to_netcdf(tmp)
    tmp.replace(output_nc)
    print(f"Wrote {output_nc}")

    return {
        "output": output_nc,
        "site_index": site_i,
        "cyclone_site_index": cyc_i,
        "n_cyclones": n_used,
        "n_days": len(replaced_days),
        "day_to_hs": day_to_hs,
        "day_to_meta": day_to_meta,
    }


def plot_hs_with_cyclones(
    baseline_hs: Path = DEFAULT_BASELINE,
    with_cyclones_hs: Path = DEFAULT_OUTPUT,
    *,
    target_lon: float = DEFAULT_TARGET_LON,
    target_lat: float = DEFAULT_TARGET_LAT,
    time_start: str | None = None,
    time_end: str | None = None,
    output_png: Path | None = None,
) -> Path:
    """Black = original daily HS; orange = days replaced by cyclone max."""
    with xr.open_dataset(baseline_hs) as b:
        lon = np.asarray(b.lon.values)
        lat = np.asarray(b.lat.values)
        site_i, _ = nearest_site_index(lon, lat, target_lon, target_lat)
        times = pd.DatetimeIndex(b.time.values)
        hs_b = np.asarray(b.hs.isel(site=site_i).values)
        lon_s, lat_s = float(lon[site_i]), float(lat[site_i])

    with xr.open_dataset(with_cyclones_hs) as c:
        # same site index (copied file)
        hs_c = np.asarray(c.hs.isel(site=site_i).values)

    if time_start is not None:
        m = times >= pd.Timestamp(time_start)
        times, hs_b, hs_c = times[m], hs_b[m], hs_c[m]
    if time_end is not None:
        m = times <= pd.Timestamp(time_end)
        times, hs_b, hs_c = times[m], hs_b[m], hs_c[m]

    changed = np.isfinite(hs_b) & np.isfinite(hs_c) & (np.abs(hs_c.astype(float) - hs_b.astype(float)) > 1e-6)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(times, hs_b, color="black", lw=0.9, label="original HS (SSP245 daily)")
    # Full spliced series as a continuous orange line (diverges only on cyclone days)
    ax.plot(times, hs_c, color="tab:orange", lw=1.2, alpha=0.9, label="HS with cyclones")
    if np.any(changed):
        # Emphasize replaced fragments as thicker orange segments
        idx = np.flatnonzero(changed)
        starts = [idx[0]]
        for a, b in zip(idx[:-1], idx[1:]):
            if b != a + 1:
                starts.append(b)
        ends = []
        for s in starts:
            e = s
            while e + 1 < len(changed) and changed[e + 1]:
                e += 1
            ends.append(e)

        labeled = False
        for s, e in zip(starts, ends):
            label = "cyclone-replaced days" if not labeled else None
            # include one baseline neighbor on each side when available so the
            # fragment reads as a connected line spike, not a lone point
            s2 = max(0, s - 1)
            e2 = min(len(times) - 1, e + 1)
            ax.plot(
                times[s2 : e2 + 1],
                hs_c[s2 : e2 + 1],
                color="tab:orange",
                lw=2.4,
                label=label,
            )
            labeled = True

    ax.set_ylabel("Hs (m)")
    ax.set_title(f"HS with cyclones @ ({lon_s:.4f}, {lat_s:.4f})  [target ({target_lon}, {target_lat})]")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_png is None:
        output_png = DEFAULT_PROJECT / "outputs" / "hs_with_cyclones_timeseries.png"
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"Saved {output_png}  (replaced days in window: {int(changed.sum())})")
    return output_png


if __name__ == "__main__":
    info = splice_hs_one_site_with_cyclones()
    plot_hs_with_cyclones()
    plot_hs_with_cyclones(
        time_start="2015-07-01",
        time_end="2017-12-31",
        output_png=DEFAULT_PROJECT / "outputs" / "hs_with_cyclones_timeseries_zoom.png",
    )
