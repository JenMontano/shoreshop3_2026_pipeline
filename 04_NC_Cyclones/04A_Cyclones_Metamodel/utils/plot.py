import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
from pyproj import Geod
from bluemath_tk.tcs.tracks import track_triming_circle
from utils.tcs_fcts import point_from_center, mean_translation_velocity, travel_time_linear_v, interpolate_geodesic, integrate_distance
geod = Geod(ellps='WGS84')

def plot_available_buoys_two_events(
    event_name: str,
    ds_parametrized: xr.Dataset,
    ds_historic: xr.Dataset,
    csv_path: str,
    nc_path: str,
    data_dir: str = "./",
    lon_site: float = -75.5730,
    lat_site: float = 36.0050,
    radius_deg: float = 7,
    radius_km: float = 777,
):
    """
    Compare buoy observations vs model results for two simulations.
    Figure 1: Track comparison (map + pressure + velocity) from CSV/NetCDF
    Figure 2: Hs comparison at buoys
    """
    
    # =========================================================================
    # LOAD TRACK DATA FROM CSV + NETCDF
    # =========================================================================
    df_params = pd.read_csv(csv_path)
    if event_name not in df_params['name'].values:
        print(f"'{event_name}' not found in CSV.")
        return
    
    idx = df_params[df_params['name'] == event_name].index[0]
    p = df_params.iloc[idx].to_dict()
    p['date_furthest'] = pd.to_datetime(p['date_furthest'])
    
    # Load observed track from NetCDF
    nc = xr.open_dataset(nc_path)
    storm = nc.isel(storm=idx)
    # ref_time = np.datetime64(storm.date.item())
    # storm["time"] = ref_time + storm.time.values * np.timedelta64(1, "h")
    df_obs = pd.DataFrame({
        'lat': storm['lat'].values, 
        'lon': ((storm['lon'].values + 180) % 360) - 180,
        'pmin': storm['wmo_pres'].values
    }, index=pd.to_datetime(storm['time'].values)).dropna()
    df_obs = df_obs[~df_obs.index.duplicated()].resample('H').interpolate()
    df_obs = track_triming_circle(df_obs, lon_site, lat_site, radius_deg)
    
    # Reconstruct synthetic track
    v_e, v_f, v_x = p['vmean_entrance_kmh'], p['vmean_furthest_kmh'], p['vmean_exit_kmh']
    df_synth = None
    
    if v_e > 0 and v_f > 0 and v_x > 0:
        lat_f, lon_f = point_from_center(lat_site, lon_site, p['azimuth_furthest'], p['distance_furthest_km'])
        lat_e, lon_e = point_from_center(lat_site, lon_site, p['azimuth_entrance'], p['distance_entrance'])
        lat_x, lon_x = point_from_center(lat_site, lon_site, p['azimuth_exit'], p['distance_exit'])
        
        d1 = geod.inv(lon_f, lat_f, lon_e, lat_e)[2] / 1000
        d2 = geod.inv(lon_f, lat_f, lon_x, lat_x)[2] / 1000
        dt1, dt2 = travel_time_linear_v(d1, v_f, v_e), travel_time_linear_v(d2, v_f, v_x)
        
        if not (np.isnan(dt1) or np.isnan(dt2)):
            t1, v1, dd1 = integrate_distance(v_f, v_e, dt1)
            t1, v1, dd1 = t1[::-1], v1[::-1], dd1[-1] - dd1[::-1]
            lat1, lon1 = interpolate_geodesic(lat_e, lon_e, lat_f, lon_f, dd1)
            
            t2, v2, dd2 = integrate_distance(v_f, v_x, dt2)
            lat2, lon2 = interpolate_geodesic(lat_f, lon_f, lat_x, lon_x, dd2)
            
            n1, n2 = len(t1), len(t2)
            df_synth = pd.DataFrame({
                'lat': np.concatenate([lat1, lat2[1:]]),
                'lon': np.concatenate([lon1, lon2[1:]]),
                'pmin': np.concatenate([np.linspace(p['p_entrance_hPa'], p['p_furthest_hPa'], n1),
                                        np.linspace(p['p_furthest_hPa'], p['p_exit_hPa'], n2)[1:]]),
                'vmean': np.concatenate([v1[:-1], [v_f], v2[1:]])
            }, index=pd.date_range(p['date_furthest'] - pd.Timedelta(hours=n1-1), periods=n1+n2-1, freq='H'))
    
    # =========================================================================
    # LOAD BUOY DATA
    # =========================================================================
    Hsig_param = ds_parametrized["Hsig"].values
    Hsig_hist = ds_historic["Hsig"].values
    time_param = pd.to_datetime(ds_parametrized["time"].values)
    time_hist = pd.to_datetime(ds_historic["time"].values)
    Xp = ds_parametrized["lon"].values
    Yp = ds_parametrized["lat"].values
    
    files = sorted(glob.glob(os.path.join(data_dir, "buoy_*_bulk_parameters.nc")))
    cmap = cm.get_cmap("tab10", len(files))
    
    buoys_valid = []
    for f in files:
        ds = xr.open_dataset(f)
        if "Hs_Buoy" not in ds:
            ds.close()
            continue
        time_min = min(time_param.min(), time_hist.min())
        time_max = max(time_param.max(), time_hist.max())
        ds_sel = ds.sel(time=slice(time_min, time_max))
        if ds_sel.time.size == 0:
            ds.close()
            continue
        hs = ds_sel["Hs_Buoy"].values
        if not np.any(np.isfinite(hs) & (hs > 0) & (hs < 20)):
            ds.close()
            continue
        lon_buoy = float(ds.longitude.values)
        lat_buoy = float(ds.latitude.values)
        buoy_id = os.path.basename(f).split("_")[1]
        station = np.argmin(np.sqrt((Xp - lon_buoy)**2 + (Yp - lat_buoy)**2))
        buoys_valid.append((buoy_id, f, lon_buoy, lat_buoy, station))
        ds.close()
    
    n_valid = len(buoys_valid)
    
    # =========================================================================
    # FIGURE 1: TRACK COMPARISON
    # =========================================================================
    fig1 = plt.figure(figsize=(16, 5))
    fig1.suptitle(f"{event_name}", fontsize=14, fontweight='bold')
    
    ax1 = fig1.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
    ax1.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax1.plot(df_obs['lon'], df_obs['lat'], 'r-', lw=2, label='Observed')
    if df_synth is not None:
        ax1.plot(df_synth['lon'], df_synth['lat'], 'b--', marker='o', ms=2, lw=1, label='Parametrized')
    angles = np.linspace(0, 2*np.pi, 361)
    ax1.plot(lon_site + radius_deg*np.cos(angles)/np.cos(np.radians(lat_site)),
             lat_site + radius_deg*np.sin(angles), 'k--', lw=1)
    ax1.scatter(lon_site, lat_site, c='red', s=80, marker='x', zorder=10)
    for i, (buoy_id, _, lon_b, lat_b, _) in enumerate(buoys_valid):
        ax1.scatter(lon_b, lat_b, color=cmap(i/max(1,n_valid-1)), edgecolor='k', s=60, zorder=5)
        ax1.text(lon_b+0.1, lat_b+0.1, buoy_id, fontsize=7)
    ax1.set_extent([lon_site-10, lon_site+10, lat_site-8, lat_site+8])
    ax1.legend(fontsize=8, loc='lower left')
    ax1.set_title('Track + Buoys')
    
    ax2 = fig1.add_subplot(1, 3, 2)
    ax2.plot(df_obs.index, df_obs['pmin'], 'r-', lw=2, label='Observed')
    if df_synth is not None:
        ax2.plot(df_synth.index, df_synth['pmin'], 'b--', lw=2, label='Parametrized')
    ax2.axvline(p['date_furthest'], color='g', ls=':', lw=1)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3); ax2.set_ylabel('Pressure (hPa)')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax3 = fig1.add_subplot(1, 3, 3)
    v_obs = [mean_translation_velocity(df_obs, j, j+1) for j in range(len(df_obs)-1)]
    ax3.plot(df_obs.index[:-1], v_obs, 'r-', lw=2, label='Observed')
    if df_synth is not None:
        ax3.plot(df_synth.index, df_synth['vmean'], 'b--', lw=2, label='Parametrized')
    ax3.axvline(p['date_furthest'], color='g', ls=':', lw=1)
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3); ax3.set_ylabel('Velocity (km/h)')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # =========================================================================
    # FIGURE 2: Hs COMPARISON
    # =========================================================================
    if n_valid == 0:
        print("⚠️ No valid buoys found.")
        return
    
    ncols = 2
    nrows = int(np.ceil(n_valid / ncols))
    fig2, axes = plt.subplots(nrows, ncols, figsize=(14, nrows*3), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, (buoy_id, f, _, _, station) in enumerate(buoys_valid):
        ds = xr.open_dataset(f)
        ds_sel = ds.sel(time=slice(min(time_param.min(),time_hist.min()), max(time_param.max(),time_hist.max())))
        hs = ds_sel["Hs_Buoy"].values
        mask = np.isfinite(hs) & (hs > 0) & (hs < 20)
        ds.close()
        
        ax = axes[i]
        ax.plot(ds_sel.time.values[mask], hs[mask], lw=1.5, color=cmap(i/max(1,n_valid-1)), label=f"Buoy {buoy_id}")
        ax.plot(time_hist, Hsig_hist[:, station], 'r-', lw=1.2, label="Historic")
        ax.plot(time_param, Hsig_param[:, station], 'b--', lw=1.2, label="Parametrized")
        ax.set_title(f"Buoy {buoy_id}"); ax.set_ylabel("Hs [m]"); ax.legend(fontsize=7)
    
    for ax in axes[n_valid:]:
        ax.set_visible(False)
    
    fig2.suptitle(f"{event_name} — Hs Comparison", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
