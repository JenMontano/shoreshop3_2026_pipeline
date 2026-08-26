from pyproj import Geod
from scipy.optimize import root_scalar
import numpy as np
geod = Geod(ellps='WGS84')

def point_from_center(lat_c, lon_c, az_deg, dist_km):
    """Calcule lat/lon depuis un centre, azimut et distance."""
    lon2, lat2, _ = geod.fwd(lon_c, lat_c, az_deg, dist_km * 1000)
    return lat2, lon2


def compute_entrance_exit_azimuth_prolong(df, lat_s, lon_s, radius_km=777):
    """Calcule les azimuts d'entrée/sortie avec projection sur cercle."""
    def dist_from_site(lat, lon):
        return geod.inv(lon_s, lat_s, lon, lat)[2] / 1000
    
    lat_e, lon_e = df.iloc[0][['lat', 'lon']]
    d1 = dist_from_site(lat_e, lon_e)

    lat_x, lon_x = df.iloc[-1][['lat', 'lon']]
    dn = dist_from_site(lat_x, lon_x)
    
    return geod.inv(lon_s, lat_s, lon_e, lat_e)[0] % 360, d1, geod.inv(lon_s, lat_s, lon_x, lat_x)[0] % 360, dn


def mean_translation_velocity(df, i1, i2):
    """Vitesse de translation (km/h)."""
    lat1, lon1 = df.iloc[i1][['lat', 'lon']]
    lat2, lon2 = df.iloc[i2][['lat', 'lon']]
    dt_h = (df.index[i2] - df.index[i1]).total_seconds() / 3600
    return geod.inv(lon1, lat1, lon2, lat2)[2] / 1000 / dt_h if dt_h else np.nan


def furthest_point_from_entry_exit(df, lat_s, lon_s):
    """Point le plus éloigné de la ligne entrée-sortie."""
    if df is None or len(df) < 3:
        return None, None, None
    lon_e, lat_e = df.iloc[0][['lon', 'lat']]
    lon_x, lat_x = df.iloc[-1][['lon', 'lat']]
    dists = [geod.inv(lon_e, lat_e, r['lon'], r['lat'])[2] + 
             geod.inv(lon_x, lat_x, r['lon'], r['lat'])[2] for _, r in df.iterrows()]
    idx = df.index[dists.index(max(dists))]
    az, _, d = geod.inv(lon_s, lat_s, df.loc[idx, 'lon'], df.loc[idx, 'lat'])
    return idx, d / 1000, az % 360


def travel_time_linear_v(d_km, v1, v2):
    """Temps de parcours avec vitesse linéaire."""
    return d_km / v1 if abs(v2 - v1) < 1e-6 else (d_km / (v2 - v1)) * np.log(v2 / v1)


def solve_velocity(d, dt, v2):
    """Résout v1 sachant distance, temps et v2."""
    if dt <= 0 or d <= 0:
        return d / max(dt, 1e-6)
    def eq(v1):
        if v1 <= 0: return dt
        if abs(v2 - v1) < 1e-6: return v1 - d/dt
        return (d / (v2 - v1)) * np.log(v2 / v1) - dt
    try:
        sol = root_scalar(eq, bracket=[1e-3, 10 * v2], method='brentq')
        return sol.root if sol.converged else d / dt
    except:
        return d / dt


def interpolate_geodesic(lat1, lon1, lat2, lon2, dists_km):
    """Interpole le long d'une géodésique."""
    total_m = geod.inv(lon1, lat1, lon2, lat2)[2]
    az = geod.inv(lon1, lat1, lon2, lat2)[0]
    frac = np.clip(dists_km / (total_m / 1000), 0, 1)
    pts = [geod.fwd(lon1, lat1, az, f * total_m) for f in frac]
    return np.array([p[1] for p in pts]), np.array([p[0] for p in pts])


def integrate_distance(v1, v2, total_h, dt=1.0):
    """Intègre distance avec vitesse linéaire."""
    t = np.arange(0, total_h + dt, dt)
    v = np.linspace(v1, v2, len(t))
    return t, v, np.cumsum(v * dt) - v[0] * dt
