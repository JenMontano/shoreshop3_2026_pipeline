from scipy.io import loadmat
import numpy as np
import xarray as xr
import pandas as pd

def mat_to_xr_dataset(path):
    """
    Convert a MATLAB-like dictionary (from loadmat) into an xarray.Dataset.

    Works with keys like 'Windv_x_20000101_000000' or 'Hsig_20000101_000000'.
    """

    mat = loadmat(path)
    lon = mat.get("Xp").ravel()
    lat = mat.get("Yp").ravel()

    vars_with_time = {}
    for key in mat.keys():
        if key.startswith(("Xp", "Yp", "__")):
            continue
        parts = key.split("_")
        if len(parts) >= 3:
            varname = "_".join(parts[:-2])
            vars_with_time.setdefault(varname, []).append(key)

    ds_vars = {}
    time_index = None

    for varname, keys in vars_with_time.items():
        keys.sort()
        data_list = [mat[k].ravel() for k in keys]
        data_array = np.vstack(data_list)  # (time, points)

        timestamps = ["_".join(k.split("_")[-2:]) for k in keys]
        time_index = pd.to_datetime(timestamps, format="%Y%m%d_%H%M%S")

        ds_vars[varname] = (("time", "points"), data_array.astype(np.float32))

    ds = xr.Dataset(
        data_vars=ds_vars,
        coords={
            "time": time_index,
            "lon": ("points", lon.astype(np.float32)),
            "lat": ("points", lat.astype(np.float32)),
        },
        attrs={k: str(mat[k]) for k in mat.keys() if k.startswith("__")},
    )

    return ds

def read_adcirc_grd(grd_file: str):

    with open(grd_file, "r") as f:
        _header0 = f.readline()
        header1 = f.readline()
        header_nums = list(map(float, header1.split()))
        nelmts = int(header_nums[0])
        nnodes = int(header_nums[1])

        Nodes = np.loadtxt(f, max_rows=nnodes)
        Elmts = np.loadtxt(f, max_rows=nelmts) - 1
        lines = f.readlines()

    return Nodes, Elmts, lines