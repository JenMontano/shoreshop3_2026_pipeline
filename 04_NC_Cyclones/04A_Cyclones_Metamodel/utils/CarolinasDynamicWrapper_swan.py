import os.path as op

from datetime import datetime, timedelta
import numpy as np
import xarray as xr
from bluemath_tk.wrappers.swan.swan_wrapper import SwanModelWrapper

from bluemath_tk.tcs.vortex import vortex_model_grid
from bluemath_tk.tcs.tracks import get_vmean, ibtracs_fit_pmin_wmax, wind2rmw
import pandas as pd
from bluemath_tk.core.operations import nautical_to_mathematical


class CarolinasDynamicMWrapper(SwanModelWrapper):
    """
    Wrapper example for the BinWaves model.
    """

    def build_case(self, case_dir: str, case_context: dict) -> None:

        xds_vortex_GS = case_context.get("xds_vortex_GS")
        additional_time = case_context.get("additional_time")

        time_rteference, time_end, time_end_calc = self.generate_wind_forcing_wind(xds_vortex_GS, f"{case_dir}/wind_forcing.wind", additional_time)
        case_context["time_rteference"] = time_rteference
        case_context["time_end"] = time_end
        case_context["time_end_calc"] = time_end_calc

    def generate_wind_forcing_wind(self, xds_vortex_GS, output_file, additional_time):
        """
        Generate a SWAN wind forcing file (structured grid) from a NetCDF dataset.

        Parameters
        ----------
        ds_path : str
            Path to the NetCDF file containing u10, v10, lon, lat, and time.
        output_file : str
            Output text file name (e.g., 'wind_forcing.wind').
        """

        dt = (xds_vortex_GS.time[1] - xds_vortex_GS.time[0]).values
        dt_s = pd.to_timedelta(dt).total_seconds()

        n_add = int(additional_time / dt_s)

        new_times = pd.date_range(
            start=xds_vortex_GS.time[-1].values + np.timedelta64(int(dt_s), "s"),
            periods=n_add,
            freq=f"{int(dt_s)}S",
        )

        template = xr.zeros_like(xds_vortex_GS.isel(time=0)).expand_dims(time=new_times)

        xds_vortex_GS = xr.concat([xds_vortex_GS, template], dim="time")

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