#!/usr/bin/env python3
"""Build wave_statistics_dp-style GeoJSON from a single Dp NetCDF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.postprocessing_binwaves_bmus import MONTH_NAMES, SEASON_MONTHS, _circular_mean_deg


def generate_dp_geojson(nc_path: Path, output_path: Path) -> Path:
    nc_path = Path(nc_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(nc_path)
    try:
        var_name = "dp" if "dp" in ds.data_vars else list(ds.data_vars)[0]
        dp = ds[var_name]
        time_dim = "time"
        site_dim = "site" if "site" in dp.dims else [d for d in dp.dims if d != time_dim][0]

        lon = ds["lon"].values if "lon" in ds else ds["coord_x"].values
        lat = ds["lat"].values if "lat" in ds else ds["coord_y"].values
        n_sites = dp.sizes[site_dim]

        dp_mean = _circular_mean_deg(dp, time_dim)

        dp_rad = np.deg2rad(dp)
        dp_sin = np.sin(dp_rad)
        dp_cos = np.cos(dp_rad)
        dp_month_sin = dp_sin.groupby(f"{time_dim}.month").mean(dim=time_dim, skipna=True)
        dp_month_cos = dp_cos.groupby(f"{time_dim}.month").mean(dim=time_dim, skipna=True)
        dp_month = np.rad2deg(np.arctan2(dp_month_sin, dp_month_cos)) % 360

        dp_season: dict[str, xr.DataArray | None] = {}
        for season, months in SEASON_MONTHS.items():
            avail = [m for m in months if int(m) in dp_month_sin["month"].values]
            if avail:
                sin_s = dp_month_sin.sel(month=avail).mean(dim="month", skipna=True)
                cos_s = dp_month_cos.sel(month=avail).mean(dim="month", skipna=True)
                dp_season[season] = np.rad2deg(np.arctan2(sin_s, cos_s)) % 360
            else:
                dp_season[season] = None

        dp_mean_vals = dp_mean.values
        dp_month_vals = {m: dp_month.sel(month=m).values for m in dp_month["month"].values}
        dp_season_vals = {
            s: (da.values if da is not None else None) for s, da in dp_season.items()
        }

        features = []
        for idx in tqdm(range(n_sites), desc=output_path.name):
            props = {"id": f"{idx:07d}", "Dp_mean": float(dp_mean_vals[idx])}
            for m_idx, m_name in enumerate(MONTH_NAMES, start=1):
                props[f"Dp_mean_{m_name}"] = (
                    float(dp_month_vals[m_idx][idx]) if m_idx in dp_month_vals else np.nan
                )
            for season in SEASON_MONTHS:
                arr = dp_season_vals[season]
                props[f"Dp_mean_{season}"] = float(arr[idx]) if arr is not None else np.nan
            features.append(
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(lon[idx]), float(lat[idx])],
                    },
                }
            )

        output_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2))
        return output_path
    finally:
        ds.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nc_path", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output GeoJSON path (default: webpage_binwaves_bmus/wave_statistics_dp_<stem>.geojson)",
    )
    args = parser.parse_args()

    stem = args.nc_path.stem
    output = args.output or PROJECT_ROOT / "webpage_binwaves_bmus" / f"wave_statistics_{stem}.geojson"
    out = generate_dp_geojson(args.nc_path, output)
    size_mb = out.stat().st_size / 1e6
    print(f"Wrote {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
