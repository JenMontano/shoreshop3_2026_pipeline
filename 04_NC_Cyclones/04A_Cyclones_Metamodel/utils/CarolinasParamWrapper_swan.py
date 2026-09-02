import os.path as op

from datetime import datetime, timedelta
import numpy as np
import xarray as xr
from bluemath_tk.wrappers.swan.swan_wrapper import SwanModelWrapper

from bluemath_tk.tcs.vortex import vortex_model_grid
from bluemath_tk.tcs.tracks import get_vmean, ibtracs_fit_pmin_wmax, wind2rmw
import pandas as pd
from bluemath_tk.core.operations import nautical_to_mathematical


class CarolinasParaMWrapper(SwanModelWrapper):
    """
    Wrapper example for the BinWaves model.
    """

    def build_case(self, case_dir: str, case_context: dict) -> None:

        Tracks = case_context.get("Tracks")
        index = case_context.get("index")
        lon_grid = case_context.get("lon_grid")
        lat_grid = case_context.get("lat_grid")
        time_step = case_context.get("time_step")/3600
        additional_time_init = case_context.get("additional_time_init")
        additional_time_end = case_context.get("additional_time_end")

        time_rteference, time_end, time_end_calc = self.generate_wind_forcing_wind(Tracks, index,lon_grid ,lat_grid, time_step, f"{case_dir}/wind_forcing.wind", additional_time_end, additional_time_init)
        case_context["time_rteference"] = time_rteference
        case_context["time_end"] = time_end
        case_context["time_end_calc"] = time_end_calc

    def generate_wind_forcing_wind(self, Tracks, index,lon_grid ,lat_grid, time_step, output_file, additional_time_end, additional_time_init):
        """
        Generate a SWAN wind forcing file (structured grid) from a NetCDF dataset.

        Parameters
        ----------
        ds_path : str
            Path to the NetCDF file containing u10, v10, lon, lat, and time.
        output_file : str
            Output text file name (e.g., 'wind_forcing.wind').
        """

        storm_sel = Tracks.isel(storm=index)
        pmin = storm_sel.pmin.values
        mask = ~np.isnan(pmin)
        storm_sel = storm_sel.isel(time=np.where(mask)[0])

        center = 'WMO'
        basin = 'SP'

        xds_coef = ibtracs_fit_pmin_wmax()

        st_lon = storm_sel.lon.values
        st_lat = storm_sel.lat.values
        pmin = storm_sel.pmin.values

        self.logger.info(
                f"Creating case {index} st_lon {st_lon.mean()} st_lat {st_lat.mean()} with pmin = {pmin.mean()}"
            )

        st_vmean = np.zeros(len(storm_sel.time) - 1)
        st_move = np.zeros(len(storm_sel.time) - 1)

        for i in range(0, len(storm_sel.time) - 1):
            # consecutive storm coordinates
            lon1, lon2 = st_lon[i], st_lon[i + 1]
            lat1, lat2 = st_lat[i], st_lat[i + 1]

            # translation speed
            gamma_h, vel_mean, _, _ = get_vmean(lat1, lon1, lat2, lon2, time_step)
            st_vmean[i] = vel_mean / 1.852   # translation speed [km/h to kt]
            st_move[i] = gamma_h             # forward direction [º]

        vfx = st_vmean * np.sin((st_move - 180) * np.pi / 180) # [kt]
        vfy = st_vmean * np.cos((st_move - 180) * np.pi / 180)  # [kt]

        p1, p2, p3, p4 = xds_coef.sel(center=center.encode(), basin=basin.encode()).coef.values[:]
        #p1, p2, p3, p4 = xds_coef.sel(center=center, basin=basin).coef.values[:]
        wind_estimate = (
            p1 * np.power(pmin, 3) + p2 * np.power(pmin, 2) + p3 * np.power(pmin, 1) + p4
        )

        # radii of maximum winds is filled with Knaff (2016) estimate
        rmw_estimate = wind2rmw(
            np.full(st_lat[:-1].size, wind_estimate[:-1]), np.full(st_lat[:-1].size, st_vmean), st_lat[:-1]
        )

        start = np.datetime64(storm_sel["date"].values.item())
        hours = storm_sel["time"].values
        time_vec = start + hours.astype('timedelta64[h]')


        st = pd.DataFrame(
            index=time_vec[:-1],
            columns=[
                "lon",
                "lat",
                "vf",
                "vfx",
                "vfy",
                "pn",
                "p0",
                "vmax",
                "rmw",
            ],
        )

        st["lon"] = st_lon[:-1]  # longitude coordinate
        st["lat"] = st_lat[:-1]  # latitude coordinate
        st["vf"] = st_vmean  # translational speed [kt]
        st["vfx"] = vfx  # x-component
        st["vfy"] = vfy  # y-component
        st["pn"] = 1013  # average pressure at the surface [mbar]
        st["p0"] = pmin[:-1]  # minimum central pressure [mbar]
        st["vmax"] = wind_estimate[:-1]  # maximum winds [kt, 1-min avg]
        st["rmw"] = rmw_estimate  # radii of maximum winds [nmile]

        st.attrs = {
            "dist": "km",
            "vf": "kt",
            "p0": "mbar",
            "vmax": "kt, 1-min avg",
            "rmw": "nmile",
            "R": 4,
        }

        xds_vortex_GS = vortex_model_grid(
            storm_track=st,
            cg_lon=lon_grid,
            cg_lat=lat_grid,
            coords_mode="SPHERICAL",
        )
        xds_vortex_GS = xds_vortex_GS.transpose("time", "lon", "lat")

        dt = (xds_vortex_GS.time[1] - xds_vortex_GS.time[0]).values
        dt_s = pd.to_timedelta(dt).total_seconds()

        n_add_end = int(additional_time_end / dt_s)

        new_times_end = pd.date_range(
            start=xds_vortex_GS.time[-1].values + np.timedelta64(int(dt_s), "s"),
            periods=n_add_end,
            freq=f"{int(dt_s)}S",
        )
        n_add_init = int(additional_time_init / dt_s)

        new_times_init = pd.date_range(
            end=xds_vortex_GS.time[0].values - np.timedelta64(int(dt_s), "s"),
            periods=n_add_init,
            freq=f"{int(dt_s)}S",
        )

        template_end = xr.zeros_like(xds_vortex_GS.isel(time=0)).expand_dims(time=new_times_end)
        template_init = xr.zeros_like(xds_vortex_GS.isel(time=0)).expand_dims(time=new_times_init)

        xds_vortex_GS = xr.concat([template_init, xds_vortex_GS, template_end], dim="time")

        W = xds_vortex_GS['W']
        Dir = xds_vortex_GS['Dir']

        u10 = -np.cos(nautical_to_mathematical(Dir) * np.pi / 180) * W
        v10 = -np.sin(nautical_to_mathematical(Dir) * np.pi / 180) * W

        dims = W.dims
        xds_vortex_GS["u10"] = (dims, u10.data)
        xds_vortex_GS["v10"] = (dims, v10.data)
        u10 = xds_vortex_GS.u10
        v10 = xds_vortex_GS.v10

        nt, nlon, nlat = u10.shape

        with open(output_file, "w") as f:
            for t in range(nt):
                time_str = np.datetime_as_string(u10.time[t].values, unit="s")
                time_str = time_str.replace("T", ".").replace("-", "").replace(":", "")
                f.write(f"{time_str}\n")
                f.write("wind x component\n")
                u2d = np.flipud(u10[t].values.T)
                np.savetxt(f, u2d, fmt="%.3f")
                f.write("wind y component\n")
                v2d = np.flipud(v10[t].values.T)
                np.savetxt(f, v2d, fmt="%.3f")

        t_np_init = xds_vortex_GS.time.values[0]
        time_reference = t_np_init.astype('M8[ms]').astype(datetime).strftime("%Y%m%d.%H%M%S")

        t_np_emd = xds_vortex_GS.time.values[-1]
        time_emd = t_np_emd.astype('M8[ms]').astype(datetime)

        time_end_calc = (time_emd - timedelta(hours=1)).strftime("%Y%m%d.%H%M%S")

        return time_reference, time_emd.strftime("%Y%m%d.%H%M%S"), time_end_calc