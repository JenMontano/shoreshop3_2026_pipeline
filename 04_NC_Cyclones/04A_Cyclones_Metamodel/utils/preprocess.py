from tqdm import tqdm
import xarray as xr
import pickle
from scipy.io import loadmat, whosmat
import numpy as np
import pandas as pd
import warnings

df = pd.read_csv("outputs/selected_cases_MDA.csv", header=None)
selected_cases = df.iloc[:, 0].tolist()[:2000]

def mat_to_xr_dataset(path, variables=None, time_range=None):
    """
    Convert a MATLAB file to xarray.Dataset, loading only requested data.
    
    Parameters
    ----------
    path : str
        Path to .mat file
    variables : list of str, optional
        Variable names to load (e.g., ['Hsig', 'Windv_x']). 
        If None, loads all variables.
    time_range : tuple of (start, end), optional
        Only load timesteps within this range.
        e.g., ('2000-01-01', '2000-01-02')
    
    Returns
    -------
    xr.Dataset
    """
    mat_info = whosmat(path)
    all_keys = [name for name, shape, dtype in mat_info]

    vars_with_keys = {}
    
    for key in all_keys:
        if key.startswith(("Xp", "Yp", "__")):
            continue
        parts = key.rsplit("_", 2)
        if len(parts) >= 3:
            varname = "_".join(parts[:-2])
            timestamp_str = f"{parts[-2]}_{parts[-1]}"
            try:
                timestamp = pd.to_datetime(timestamp_str, format="%Y%m%d_%H%M%S")
                vars_with_keys.setdefault(varname, []).append((key, timestamp))
            except ValueError:
                continue

    if variables is not None:
        vars_with_keys = {k: v for k, v in vars_with_keys.items() if k in variables}
    
    if not vars_with_keys:
        raise ValueError(f"No matching variables found. Available: {list(vars_with_keys.keys())}")

    if time_range is not None:
        t_start, t_end = pd.to_datetime(time_range[0]), pd.to_datetime(time_range[1])
        for varname in vars_with_keys:
            vars_with_keys[varname] = [
                (k, t) for k, t in vars_with_keys[varname] 
                if t_start <= t <= t_end
            ]

    keys_to_load = {"Xp", "Yp"}
    for varname, key_times in vars_with_keys.items():
        keys_to_load.update(k for k, t in key_times)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Duplicate variable name")
        mat = loadmat(path, variable_names=list(keys_to_load))
    
    lon = mat["Xp"].ravel().astype(np.float32)
    lat = mat["Yp"].ravel().astype(np.float32)

    ds_vars = {}

    for varname, key_times in vars_with_keys.items():
        key_times.sort(key=lambda x: x[1])  # Sort by timestamp
        
        n_times = len(key_times)
        n_points = lon.size
        data_array = np.empty((n_times, n_points), dtype=np.float32)
        
        for i, (key, _) in enumerate(key_times):
            data_array[i] = mat[key].ravel()
        
        time_index = pd.DatetimeIndex([t for _, t in key_times])
        
        unique_times, time_indices = np.unique(time_index, return_index=True)
        data_array = data_array[time_indices]
        time_index = pd.DatetimeIndex(unique_times)
        
        ds_vars[varname] = (("time", "points"), data_array)
    
    return xr.Dataset(
        data_vars=ds_vars,
        coords={
            "time": time_index,
            "lon": ("points", lon),
            "lat": ("points", lat),
        },
    )


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

ds_list = []

for i, case in enumerate(selected_cases):
    print(f"Processing case {i+1}/{len(selected_cases)}: {case}")
    swan_output = f"cases/metamodel/{case}/output.mat"

    ds = mat_to_xr_dataset(swan_output, variables=["Hsig","Dir", "TDir", "TPsmoo", "Tm01", "Windv_x", "Windv_y"])
    ds = ds.expand_dims(case_num=[case])
    ds_list.append(ds)

print("saving")
with open("outputs/ds_list_complet.pkl", "wb") as f:
    pickle.dump(ds_list, f)