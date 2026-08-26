# Generate GeoJSON with water level statistics (trend, quantile 95, quantile 5)
import json
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

# Input and output paths
input_path = Path("./inputs/waterLevels/WaterLevels_1979_2024_NorthCarolina.nc")
out_path = Path("./outputs/water_level_statistics.geojson")

print(f"Reading water level data from: {input_path}")
print(f"Saving GeoJSON to: {out_path}")

# Load water level dataset
ds = xr.open_dataset(input_path)

# Identify dimensions
time_dim = "time"
node_dim = "node"

print(f"Dimensions: {dict(ds.sizes)}")
print(f"Time dimension: {time_dim}")
print(f"Node dimension: {node_dim}")

# Get coordinates (lon and lat are data variables with dimension 'node')
lon = ds.lon.values
lat = ds.lat.values
depth = ds.depth.values if "depth" in ds.data_vars else None

if lon is None or lat is None:
    raise RuntimeError("Could not find lon/lat in water level dataset")

# Get water level data variable
wl_data = ds.wl  # wl is (time, node)

# Ensure common (time, node) ordering
wl_data = wl_data.transpose(time_dim, node_dim)

# Time range
time_vals = pd.to_datetime(wl_data[time_dim].values)
if time_vals.size == 0:
    raise RuntimeError("No time values found in water level dataset")

time_start = time_vals[0].strftime("%Y-%m-%d")
time_end = time_vals[-1].strftime("%Y-%m-%d")

n_nodes = wl_data.sizes[node_dim]
print(f"Nodes: {n_nodes}")
print(f"Time range: {time_start} to {time_end} (total steps: {wl_data.sizes[time_dim]})")

# Convert time to numeric for trend computation (days since first observation)
# Convert pandas DatetimeIndex to numpy datetime64, then to days
time_vals_np = np.array(time_vals, dtype='datetime64[ns]')
time_num = (time_vals_np - time_vals_np[0]).astype('timedelta64[D]').astype(float)

# -------- Vectorized statistics over time --------
# Quantiles
wl_q95 = wl_data.quantile(0.95, dim=time_dim, skipna=True)
wl_q5 = wl_data.quantile(0.05, dim=time_dim, skipna=True)
wl_mean = wl_data.mean(dim=time_dim, skipna=True)

# -------- Compute trends for each node --------
print("Computing trends for each node...")
trends_mm_per_yr = np.full(n_nodes, np.nan)
trends_r2 = np.full(n_nodes, np.nan)
trends_pvalue = np.full(n_nodes, np.nan)

for idx in tqdm(range(n_nodes), desc="Computing trends"):
    wl_series = wl_data.isel({node_dim: idx}).values
    mask = ~np.isnan(wl_series)
    
    if np.sum(mask) > 1:  # Need at least 2 points for trend
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                time_num[mask], wl_series[mask])
            # Convert to mm/year (slope is in m/day, multiply by 365.25 * 1000)
            trends_mm_per_yr[idx] = slope * 365.25 * 1000
            trends_r2[idx] = r_value ** 2
            trends_pvalue[idx] = p_value
        except Exception as e:
            print(f"Warning: Could not compute trend for node {idx}: {e}")

# -------- Build GeoJSON features --------
features = []

for idx in tqdm(range(n_nodes), desc="Building GeoJSON"):
    # Extract scalar stats for this node
    def _val(da):
        if isinstance(da, (xr.DataArray, xr.Dataset)):
            v = da.isel({node_dim: idx}).values
        else:
            v = da[idx]
        return float(v) if np.isfinite(v) else None
    
    # Node identifier (use node index)
    node_id = f"{idx:07d}"
    
    feature = {
        "type": "Feature",
        "properties": {
            "id": node_id,
            "node_id": node_id,
            "dataset_index": int(idx),
            "time_start": time_start,
            "time_end": time_end,
            "trend_mm_per_yr": _val(trends_mm_per_yr),
            "trend_r2": _val(trends_r2),
            "trend_pvalue": _val(trends_pvalue),
            "quantile_95": _val(wl_q95),
            "quantile_5": _val(wl_q5),
            "mean": _val(wl_mean),
        },
        "geometry": {
            "type": "Point",
            "coordinates": [float(lon[idx]), float(lat[idx])],
        },
    }
    
    # Add depth if available
    if depth is not None:
        feature["properties"]["depth"] = float(depth[idx]) if np.isfinite(depth[idx]) else None
    
    features.append(feature)

geojson = {
    "type": "FeatureCollection",
    "features": features,
}

# Ensure output directory exists
out_path.parent.mkdir(parents=True, exist_ok=True)

# Write GeoJSON
out_path.write_text(json.dumps(geojson, indent=2))
print(f"Saved GeoJSON with {len(features)} features to {out_path}")

# Close dataset
ds.close()
