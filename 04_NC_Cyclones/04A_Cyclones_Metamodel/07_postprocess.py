from scipy.io import loadmat, whosmat
import numpy as np
import pandas as pd
import xarray as xr
import warnings
import matplotlib.pyplot as plt
import os

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

df = pd.read_csv("outputs/selected_cases_MDA.csv", header=None)
selected_cases = df.iloc[:, 0].tolist()[:3000]

lons = [-77.766, -76.9845, -75.36, -74.837]
lats = [33.441, 33.8928, 35.10, 36.603]
names_points = ["point_1", "point_2", "point_3", "point_4"]

#variables = ["Hsig", "Dir", "TDir", "TPsmoo", "Tm01", "Dir_wind", "Windv"]
variables = ["Hsig", "Dir", "TPsmoo", "Windv_x", "Windv_y"]
n_var = len(variables)

all_cases = {}  # Dataset par cas (toutes les variables)

for ii, case in enumerate(selected_cases):
    if ii % 100 == 0:
        print(f"Processing case {(ii+1)/len(selected_cases)*100:.2f} % ({ii+1}/{len(selected_cases)})")

    path1 = f"cases/metamodel/{case}/output.mat"
    path2 = f"cases/metamodel_added/{case}/output.mat"

    if os.path.exists(path1):
        swan_output = path1
    elif os.path.exists(path2):
        swan_output = path2
    else:
        print(f"⚠️ output.mat not found for case {case}")
        continue

    ds = mat_to_xr_dataset(swan_output, ["Hsig","Dir", "TDir", "TPsmoo", "Tm01", "Windv_x", "Windv_y"])
    # W = np.sqrt(ds.Windv_x**2 + ds.Windv_y**2)
    # Dir_wind = (270 - np.rad2deg(np.arctan2(ds.Windv_y, ds.Windv_x))) % 360
    # ds["Dir_wind"] = Dir_wind
    # ds["Windv"] = W
    #ds = ds.drop_vars(["Windv_x", "Windv_y"])

    if case == selected_cases[0]:
        lon_model = ds.lon.values
        lat_model = ds.lat.values

        def nearest_index(lon, lat, lon_model, lat_model):
            dist2 = (lon_model - lon) ** 2 + (lat_model - lat) ** 2
            return np.argmin(dist2)

        nearest_idx = [
            nearest_index(lon, lat, lon_model, lat_model)
            for lon, lat in zip(lons, lats)
        ]

    time_peaks = []
    for idx in nearest_idx:
        hsig_point = ds.Hsig[:, idx]
        t_peak = ds.time[hsig_point.argmax(dim="time")].values
        time_peaks.append(pd.Timestamp(t_peak))

    time_mean = pd.to_datetime(np.mean([t.value for t in time_peaks]))
    t_start = time_mean - pd.Timedelta(hours=11)
    t_end = time_mean + pd.Timedelta(hours=12)
    common_time = pd.date_range(t_start, t_end, freq="h")
    time_index = pd.to_datetime(ds.time.values)
    data_start = time_index.min()
    data_end = time_index.max()

    point_names = [f"point_{i+1}" for i in range(len(nearest_idx))]

    # Construire un tableau (time, point) par variable
    data_vars = {}
    for var in variables:
        arr = np.full((len(common_time), len(nearest_idx)), np.nan, dtype=np.float32)
        for j, idx in enumerate(nearest_idx):
            series = pd.Series(ds[var][:, idx].values, index=time_index)
            series = series.reindex(
                common_time,
                method="nearest",
                tolerance=pd.Timedelta("1h"),
            )
            series[(common_time < data_start) | (common_time > data_end)] = np.nan
            first_valid = series.first_valid_index()
            last_valid = series.last_valid_index()
            if first_valid is not None:
                series.loc[:first_valid] = series.loc[:first_valid].bfill()
            if last_valid is not None:
                series.loc[last_valid:] = series.loc[last_valid:].ffill()
            arr[:, j] = series.values
        data_vars[var] = (("time", "point"), arr)
        
    ntimes = np.arange(len(common_time))
    ds_case = xr.Dataset(
        data_vars=data_vars,
        coords={"time": ntimes, "point": point_names},
    )
    all_cases[case] = ds_case

    # Figure : un cadran par variable
    n_rows = n_var

    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2 * n_rows), sharex=True)
    axes = np.atleast_2d(axes)

    for k, var in enumerate(variables):
        ax = axes.flat[k]
        arr = ds_case[var].values
        for j in range(arr.shape[1]):
            ax.plot(common_time, arr[:, j], label=point_names[j])
        ax.axvline(time_mean, color="k", ls="--", lw=1, alpha=0.7)
        ax.set_ylabel(var)
        if k == 0:
            ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    for k in range(n_var, axes.size):
        axes.flat[k].set_visible(False)

    fig.suptitle(f"Case {case} — toutes variables (fenêtre centrée sur pic Hsig)", fontsize=12)
    axes[-1, 0].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(f"figures/08_metamodel_test/all_vars_case_{case}.png", dpi=150)
    plt.close()
    del ds

xds_post = xr.concat(
    list(all_cases.values()),
    dim=pd.Index(list(all_cases.keys()), name="case_num"),
)
xds_post.to_netcdf("outputs/Vars_postprocessed_new.nc")  # ou "postprocessed.nc"