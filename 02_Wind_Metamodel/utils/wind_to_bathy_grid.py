"""Interpolate WHACS wind onto the SWAN / bathy grid (hourly files, one folder per day)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.path as mpath
import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

FILL_VALUE = np.float32(-9999.0)
INTERP_METHOD = "idw_scattered_k8"
IDW_K = 8
IDW_POWER = 2
IDW_MAX_DIST_DEG = 0.25

# SWAN INPGRID BOTTOM / WIND in inputs/INPUT (same grid as depth.dat; geographic, angle 0)
SWAN_INPGRID = dict(
    x0=-77.31875000000001,
    y0=34.37708333333333,
    alpc=0.0,
    dx=0.004166666666677088,
    dy=0.004166666666677088,
    nx=796,
    ny=749,
)

# SWAN CGRID in inputs/INPUT (computational grid; rotated 26 deg from East)
SWAN_CGRID = dict(
    xpc=-75.53081442660546,
    ypc=34.92560297685041,
    alpc=26.0,
    xlen=1.37,
    ylen=2.05,
    mxc=328,
    myc=491,
)

# ShoreShop NC domain (lon, lat); optional crop_polygon=NC_CROP_POLYGON
NC_CROP_POLYGON = np.array(
    [
        [-75.53081443, 34.92560298],
        [-74.29946658, 35.52617145],
        [-75.19812743, 37.36869924],
        [-76.42947528, 36.76813077],
        [-75.53081443, 34.92560298],
    ],
    dtype=np.float64,
)


def _rotated_grid_lonlat(
    x0: float,
    y0: float,
    alpc_deg: float,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Geographic lon/lat at each grid node; m along x-axis, n along y-axis."""
    alpc = np.deg2rad(alpc_deg)
    m = np.arange(nx + 1, dtype=np.float64)
    n = np.arange(ny + 1, dtype=np.float64)
    mm, nn = np.meshgrid(m, n)
    lon2d = x0 + mm * dx * np.cos(alpc) - nn * dy * np.sin(alpc)
    lat2d = y0 + mm * dx * np.sin(alpc) + nn * dy * np.cos(alpc)
    return m, n, lat2d, lon2d


def swan_inpgrid_coords() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = SWAN_INPGRID
    return _rotated_grid_lonlat(g["x0"], g["y0"], g["alpc"], g["dx"], g["dy"], g["nx"], g["ny"])


def swan_cgrid_coords() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = SWAN_CGRID
    dx = g["xlen"] / g["mxc"]
    dy = g["ylen"] / g["myc"]
    return _rotated_grid_lonlat(g["xpc"], g["ypc"], g["alpc"], dx, dy, g["mxc"], g["myc"])


def _swan_grid_coords() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, _, lat2d, lon2d = swan_inpgrid_coords()
    return lat2d[:, 0], lon2d[0, :], lat2d, lon2d


def _wind_bbox_slices(inside_mask: np.ndarray) -> tuple[slice, slice]:
    rows, cols = np.where(inside_mask)
    if rows.size == 0:
        raise ValueError("No grid points inside the requested domain")
    return slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1)


def _polygon_outside_mask(lon2d: np.ndarray, lat2d: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    verts = np.asarray(polygon, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 3:
        raise ValueError("crop_polygon must be an (N, 2) array of [lon, lat] vertices")
    points = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    inside = mpath.Path(verts).contains_points(points).reshape(lon2d.shape)
    return ~inside


def _magdir_from_uv(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wspd = np.hypot(u, v)
    wdir = (180 + np.rad2deg(np.arctan2(u, v))) % 360
    return wspd, wdir


def _idw_scatter(
    lon_pt: np.ndarray,
    lat_pt: np.ndarray,
    values: np.ndarray,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    *,
    k: int = IDW_K,
    power: float = IDW_POWER,
    max_dist_deg: float | None = IDW_MAX_DIST_DEG,
) -> np.ndarray:
    """Inverse-distance weighting from WHACS seapoints onto target lon/lat."""
    src = np.column_stack([lon_pt, lat_pt])
    tgt = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    tree = cKDTree(src)
    k_use = min(k, len(lon_pt))
    dist, idx = tree.query(tgt, k=k_use)
    dist = np.maximum(dist, 1e-12)
    w = 1.0 / dist**power
    out = (w * values[idx]).sum(axis=1) / w.sum(axis=1)
    out = out.reshape(lon2d.shape)
    if max_dist_deg is not None:
        nearest, _ = tree.query(tgt, k=1)
        out[nearest.reshape(lon2d.shape) > max_dist_deg] = np.nan
    return out


def _interp_to_grid(
    lon_pt: np.ndarray,
    lat_pt: np.ndarray,
    outside_mask: np.ndarray,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    wspd_pt: np.ndarray,
    wdir_pt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    wdir = np.deg2rad(wdir_pt)
    u_pt = (-wspd_pt * np.sin(wdir)).astype(np.float64)
    v_pt = (-wspd_pt * np.cos(wdir)).astype(np.float64)
    u_out = _idw_scatter(lon_pt, lat_pt, u_pt, lon2d, lat2d)
    v_out = _idw_scatter(lon_pt, lat_pt, v_pt, lon2d, lat2d)

    wspd_out = np.hypot(u_out, v_out).astype(np.float32)
    wdir_out = ((np.rad2deg(np.arctan2(-u_out, -v_out)) + 360) % 360).astype(np.float32)
    far = ~np.isfinite(wspd_out)
    wspd_out[far | outside_mask] = FILL_VALUE
    wdir_out[far | outside_mask] = FILL_VALUE
    return wspd_out, wdir_out


def _write_hour_nc(
    out_path: Path,
    time: np.datetime64,
    lat: np.ndarray,
    lon: np.ndarray,
    wspd: np.ndarray,
    wdir: np.ndarray,
    grid_source: str,
    lat_slice: slice,
    lon_slice: slice,
    crop_polygon: np.ndarray | None = None,
    *,
    lat2d: np.ndarray | None = None,
    lon2d: np.ndarray | None = None,
    grid_rotation_deg: float | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".nc.tmp")
    if tmp.exists():
        tmp.unlink()

    ny, nx = wspd.shape
    rotated = lat2d is not None and lon2d is not None
    with nc.Dataset(tmp, "w", format="NETCDF4") as ds:
        if rotated:
            ds.createDimension("y", ny)
            ds.createDimension("x", nx)
            ydim, xdim = "y", "x"
        else:
            ds.createDimension("lat", ny)
            ds.createDimension("lon", nx)
            ydim, xdim = "lat", "lon"

        if rotated:
            lat_var = ds.createVariable("lat", "f8", (ydim, xdim))
            lat_var[:] = lat2d
            lon_var = ds.createVariable("lon", "f8", (ydim, xdim))
            lon_var[:] = lon2d
        else:
            lat_var = ds.createVariable("lat", "f8", (ydim,))
            lat_var[:] = lat
            lon_var = ds.createVariable("lon", "f8", (xdim,))
            lon_var[:] = lon
        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"

        for name, data, attrs in (
            ("wspd", wspd, {"long_name": "Wind speed", "units": "m s-1"}),
            ("wdir", wdir, {"long_name": "Wind direction", "units": "degree"}),
        ):
            var = ds.createVariable(
                name, "f4", (ydim, xdim), zlib=True, complevel=4, fill_value=FILL_VALUE
            )
            var[:] = data
            var.setncattr("missing_value", float(FILL_VALUE))
            for k, v in attrs.items():
                var.setncattr(k, v)

        ds.setncattr("time", np.datetime_as_string(time, unit="s"))
        ds.setncattr("grid_source", grid_source)
        ds.setncattr("interpolation_method", INTERP_METHOD)
        if grid_rotation_deg is not None:
            ds.setncattr("grid_rotation_deg", float(grid_rotation_deg))
        ds.setncattr("bathy_lat_start", int(lat_slice.start or 0))
        ds.setncattr("bathy_lon_start", int(lon_slice.start or 0))
        if crop_polygon is not None:
            verts = np.asarray(crop_polygon, dtype=np.float64)
            ds.setncattr(
                "crop_polygon_vertices",
                ";".join(f"{lon:.8f},{lat:.8f}" for lon, lat in verts),
            )

    tmp.rename(out_path)


def _grid_setup(
    bathy: xr.Dataset,
    lon_pt: np.ndarray,
    lat_pt: np.ndarray,
    *,
    target_grid: str = "swan",
    crop_polygon: np.ndarray | None = None,
):
    if target_grid == "swan":
        lat, lon, lat2d, lon2d = _swan_grid_coords()
        lat_slice = slice(0, lat.size)
        lon_slice = slice(0, lon.size)
        grid_source = "SWAN INPGRID (depth.dat)"
        grid_rotation_deg = SWAN_INPGRID["alpc"]
    elif target_grid == "cgrid":
        _, _, lat2d, lon2d = swan_cgrid_coords()
        lat = lat2d[:, 0]
        lon = lon2d[0, :]
        lat_slice = slice(0, lat2d.shape[0])
        lon_slice = slice(0, lat2d.shape[1])
        grid_source = "SWAN CGRID (26 deg rotated)"
        grid_rotation_deg = SWAN_CGRID["alpc"]
    elif target_grid == "bathy":
        lon_full = bathy.lon.values
        lat_full = bathy.lat.values
        lon2d_full, lat2d_full = np.meshgrid(lon_full, lat_full)
        if crop_polygon is not None:
            inside = ~_polygon_outside_mask(lon2d_full, lat2d_full, crop_polygon)
        else:
            inside = (
                (lat2d_full >= lat_pt.min()) & (lat2d_full <= lat_pt.max())
                & (lon2d_full >= lon_pt.min()) & (lon2d_full <= lon_pt.max())
            )
        lat_slice, lon_slice = _wind_bbox_slices(inside)
        lat = lat_full[lat_slice]
        lon = lon_full[lon_slice]
        lon2d, lat2d = np.meshgrid(lon, lat)
        grid_source = "bathy.nc"
        grid_rotation_deg = 0.0
    else:
        raise ValueError("target_grid must be 'swan', 'cgrid', or 'bathy'")

    outside = np.zeros(lat2d.shape, dtype=bool)
    if crop_polygon is not None:
        outside |= _polygon_outside_mask(lon2d, lat2d, crop_polygon)

    use_rotated_coords = target_grid == "cgrid"
    return (
        outside, lon2d, lat2d, lat, lon, lat_slice, lon_slice, grid_source,
        grid_rotation_deg, use_rotated_coords,
    )


def wind_magdir_to_bathy_hourly(
    bathy_path: str | Path,
    out_dir: str | Path,
    t_start: str,
    t_end: str,
    *,
    uwnd_path: str | Path | None = None,
    vwnd_path: str | Path | None = None,
    magdir_path: str | Path | None = None,
    target_grid: str = "swan",
    crop_polygon: np.ndarray | None = None,
    skip_existing: bool = True,
) -> list[Path]:
    """
    Write one NetCDF per hour under out_dir/yyyy_mm_dd/HH/wind.nc.

    target_grid='swan' (default): same regular grid as depth.dat / SWAN INPGRID.
    target_grid='cgrid': SWAN computational grid (26 deg rotation; matches bathy plot).
    target_grid='bathy': subset of bathy.nc (optionally crop_polygon).

    Wind is IDW-interpolated from WHACS seapoints (k=8). Nodes farther than
    IDW_MAX_DIST_DEG from any seapoint are left as fill value (-9999).
    """

    def _hour_path(ts: pd.Timestamp) -> Path:
        return out_dir / ts.strftime("%Y_%m_%d") / f"{ts.hour:02d}" / "wind.nc"

    bathy_path = Path(bathy_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bathy = xr.open_dataset(bathy_path)

    if magdir_path is not None:
        wind = xr.open_dataset(magdir_path)
        lon_pt = wind.longitude.isel(time=0).values
        lat_pt = wind.latitude.isel(time=0).values
        ctx = _grid_setup(
            bathy, lon_pt, lat_pt, target_grid=target_grid, crop_polygon=crop_polygon
        )
        (
            outside, lon2d, lat2d, lat, lon,
            ls, lns, grid_source, grid_rot, rotated_coords,
        ) = ctx
        times = pd.date_range(t_start, t_end, freq="h")
        times = times[times <= pd.Timestamp(t_end)]
        times = times.intersection(pd.DatetimeIndex(wind.time.values))
        written: list[Path] = []
        for ts in times:
            out_path = _hour_path(ts)
            if skip_existing and out_path.exists():
                written.append(out_path)
                continue
            step = wind.sel(time=ts, method="nearest")
            wspd_g, wdir_g = _interp_to_grid(
                lon_pt, lat_pt, outside, lon2d, lat2d,
                step.wspd.values, step.wdir.values,
            )
            _write_hour_nc(
                out_path, np.datetime64(ts), lat, lon, wspd_g, wdir_g,
                grid_source, ls, lns, crop_polygon=crop_polygon,
                lat2d=lat2d if rotated_coords else None,
                lon2d=lon2d if rotated_coords else None,
                grid_rotation_deg=grid_rot,
            )
            written.append(out_path)
            print(f"wrote {out_path}", flush=True)
        wind.close()
        bathy.close()
        return written

    if uwnd_path is None or vwnd_path is None:
        raise ValueError("Provide magdir_path or both uwnd_path and vwnd_path")

    u_ds = xr.open_dataset(uwnd_path)
    v_ds = xr.open_dataset(vwnd_path)
    lon_pt = u_ds.longitude.isel(time=0).values
    lat_pt = u_ds.latitude.isel(time=0).values
    ctx = _grid_setup(
        bathy, lon_pt, lat_pt, target_grid=target_grid, crop_polygon=crop_polygon
    )
    (
        outside, lon2d, lat2d, lat, lon,
        ls, lns, grid_source, grid_rot, rotated_coords,
    ) = ctx

    times = pd.date_range(t_start, t_end, freq="h")
    times = times[times <= pd.Timestamp(t_end)]
    times = times.intersection(pd.DatetimeIndex(u_ds.time.values))

    written = []
    for ts in times:
        out_path = _hour_path(ts)
        if skip_existing and out_path.exists():
            written.append(out_path)
            continue
        u = u_ds.uwnd.sel(time=ts, method="nearest").values
        v = v_ds.vwnd.sel(time=ts, method="nearest").values
        wspd_pt, wdir_pt = _magdir_from_uv(u, v)
        wspd_g, wdir_g = _interp_to_grid(
            lon_pt, lat_pt, outside, lon2d, lat2d, wspd_pt, wdir_pt,
        )
        _write_hour_nc(
            out_path, np.datetime64(ts), lat, lon, wspd_g, wdir_g,
            grid_source, ls, lns, crop_polygon=crop_polygon,
            lat2d=lat2d if rotated_coords else None,
            lon2d=lon2d if rotated_coords else None,
            grid_rotation_deg=grid_rot,
        )
        written.append(out_path)
        print(f"wrote {out_path}", flush=True)

    u_ds.close()
    v_ds.close()
    bathy.close()
    return written


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1] / "inputs"
    wind_magdir_to_bathy_hourly(
        base / "bathy.nc",
        base,
        "2017-12-20T00:00:00",
        "2018-01-15T23:00:00",
        uwnd_path=base / "north_carolina_04_uwnd_WHACS.nc",
        vwnd_path=base / "north_carolina_04_vwnd_WHACS.nc",
        target_grid="swan",
        skip_existing=False,
    )
