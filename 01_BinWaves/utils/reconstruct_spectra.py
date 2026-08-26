#!/usr/bin/env python
"""
Fast reconstruction of onshore spectra from offshore spectra and kp coefficients.

The sum over case_num:
    onshore(time, dir, site, freq) = sum_c offshore(c, time) * kp(c, dir, site, freq)
is a matrix multiplication, so it is computed with a single multithreaded BLAS
matmul (all cores, float32) instead of dask.

The offshore input can be either:
- already case-binned: efth(case_num, time), e.g. offshore_spectra_case_gridN.nc
- a raw spectrum: efth(time, freq, dir), e.g. a corrected buoy spectrum. It is
  then projected onto the SWAN cases using the cases CSV (swan_cases_averaged.csv),
  same as transform_Offshore_spectrum with fixed_direction=True.

Usage:
    python reconstruct_spectra.py 1982 1983 1984 --grid 1     # individual years
    python reconstruct_spectra.py 1982-1990 --grid 2          # year range
    python reconstruct_spectra.py 1997-02 --grid 4            # single month
    python reconstruct_spectra.py 1997-01-1997-06 --grid 4    # month range
    python reconstruct_spectra.py 1982-1990 1997-02 --grid 3  # mix of the above

    # custom offshore spectrum (raw buoy spectrum) and output tag:
    python reconstruct_spectra.py 1997-01 --grid 4 \\
        --offshore grid4/inputs/44014_spec_WHACS_buoy_correted_15D_Jan97.nc \\
        --tag WHACS_44014
"""

import argparse
import os
import time as time_mod

import numpy as np
import pandas as pd
import xarray as xr


def parse_periods(period_args):
    """
    Parse period arguments into a sorted list of unique period labels.

    Supported token formats:
    - "1997"              -> whole year
    - "1982-1990"         -> year range (inclusive)
    - "1997-02"           -> single month (second number <= 12)
    - "1997-01-1997-06"   -> month range (inclusive)

    Returned labels ("1997", "1997-02") are used directly in
    xarray's time selection and in output filenames.
    """
    periods = set()
    for arg in period_args:
        parts = arg.split("-")
        if len(parts) == 1:
            periods.add(f"{int(parts[0])}")
        elif len(parts) == 2:
            first, second = int(parts[0]), int(parts[1])
            if second <= 12:
                # Year-month, e.g. "1997-02"
                periods.add(f"{first}-{second:02d}")
            else:
                # Year range, e.g. "1982-1990"
                periods.update(str(y) for y in range(first, second + 1))
        elif len(parts) == 4:
            # Month range, e.g. "1997-01-1997-06"
            y1, m1, y2, m2 = (int(p) for p in parts)
            year, month = y1, m1
            while (year, month) <= (y2, m2):
                periods.add(f"{year}-{month:02d}")
                month += 1
                if month > 12:
                    year, month = year + 1, 1
        else:
            raise ValueError(f"Cannot parse period argument: {arg!r}")
    return sorted(periods)


def open_efth(path):
    """Open a spectra file and return its efth DataArray (whatever it is named)."""
    if not os.path.exists(path) and os.path.exists(path + ".nc"):
        path = path + ".nc"
    ds = xr.open_dataset(path)
    if "efth" in ds:
        return ds["efth"]
    if "__xarray_dataarray_variable__" in ds:
        return ds["__xarray_dataarray_variable__"].rename("efth")
    data_vars = list(ds.data_vars)
    if len(data_vars) == 1:
        return ds[data_vars[0]].rename("efth")
    raise ValueError(
        f"Could not identify the spectra variable in {path}. "
        f"Available variables: {data_vars}"
    )


def case_projection_indices(
    spectrum, cases_csv_path, case_nums, freq_tol=0.001, dir_tol=2.0
):
    """
    Map each SWAN case to the nearest (freq, dir) bin of a raw spectrum.

    Equivalent to transform_Offshore_spectrum with fixed_direction=True:
    nearest-neighbour selection with tolerances, missing cases get zeros.

    Returns (freq_idx, dir_idx, valid_mask), one entry per case in case_nums.
    """
    cases = pd.read_csv(cases_csv_path).set_index("case_num").loc[case_nums]
    case_freq = cases["freq"].to_numpy()
    case_dir = cases["dir"].to_numpy()

    spec_freq = spectrum["freq"].values
    spec_dir = spectrum["dir"].values

    freq_dist = np.abs(spec_freq[None, :] - case_freq[:, None])
    freq_idx = freq_dist.argmin(axis=1)

    # Circular distance in degrees
    dir_dist = np.abs(((spec_dir[None, :] - case_dir[:, None] + 180) % 360) - 180)
    dir_idx = dir_dist.argmin(axis=1)

    valid = (freq_dist[np.arange(len(case_nums)), freq_idx] <= freq_tol) & (
        dir_dist[np.arange(len(case_nums)), dir_idx] <= dir_tol
    )
    if not valid.all():
        print(
            f"Warning: {(~valid).sum()}/{len(valid)} cases have no matching "
            f"(freq, dir) bin in the offshore spectrum; they are set to zero."
        )
    return freq_idx, dir_idx, valid


def load_kp_matrix(kp_coeffs):
    """
    Load kp coefficients as a contiguous float32 matrix (case_num, dir*site*freq).

    Returns the matrix and the (dir, site, freq) template DataArray used to
    rebuild coordinates on the reconstructed output.
    """
    kp = kp_coeffs.transpose("case_num", "dir", "site", "freq")
    n_cases = kp.sizes["case_num"]
    kp_matrix = kp.values.reshape(n_cases, -1).astype(np.float32)
    return kp_matrix, kp


def offshore_to_case_matrix(offshore_spectra, projection=None):
    """
    Return the offshore spectra as a float32 (time, case_num) matrix.

    If projection is None, offshore_spectra must already have (case_num, time)
    dims. Otherwise it is a raw efth(time, freq, dir) spectrum and projection
    is the (freq_idx, dir_idx, valid) tuple from case_projection_indices.
    """
    if projection is None:
        return offshore_spectra.transpose("time", "case_num").values.astype(
            np.float32
        )
    freq_idx, dir_idx, valid = projection
    values = offshore_spectra.transpose("time", "freq", "dir").values
    matrix = values[:, freq_idx, dir_idx].astype(np.float32)
    matrix[:, ~valid] = 0.0
    return matrix


def reconstruct_period(offshore_spectra, kp_matrix, kp_template, period, projection=None):
    """
    Reconstruct onshore spectra for one period (year or month) with one matmul.
    """
    offshore = offshore_spectra.sel(time=period)
    offshore_matrix = offshore_to_case_matrix(offshore, projection)

    result = offshore_matrix @ kp_matrix  # (time, dir*site*freq)

    n_time = offshore.sizes["time"]
    shape = (
        n_time,
        kp_template.sizes["dir"],
        kp_template.sizes["site"],
        kp_template.sizes["freq"],
    )
    reconstructed = xr.DataArray(
        result.reshape(shape),
        dims=("time", "dir", "site", "freq"),
        coords={
            "time": offshore["time"],
            "dir": kp_template["dir"],
            "site": kp_template["site"],
            "freq": kp_template["freq"],
        },
        name="efth",
    )
    # Carry over auxiliary site coordinates (coord_x, coord_y, lat, lon, ...)
    extra_coords = {
        name: coord
        for name, coord in kp_template.coords.items()
        if name not in reconstructed.coords
        and set(coord.dims) <= set(reconstructed.dims)
    }
    return reconstructed.assign_coords(extra_coords)


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct spectra for specified years or months."
    )
    parser.add_argument(
        "periods",
        nargs="+",
        help=(
            "Periods to process: years (1982 1983), year ranges (1982-1990), "
            "months (1997-02) or month ranges (1997-01-1997-06)"
        ),
    )
    parser.add_argument(
        "--grid",
        type=int,
        required=True,
        choices=[1, 2, 3, 4],
        help="Grid number (1, 2, 3, or 4)",
    )
    parser.add_argument(
        "--offshore",
        type=str,
        default=None,
        help=(
            "Path to the offshore spectra file. Either case-binned "
            "efth(case_num, time) or raw efth(time, freq, dir). Default: "
            "gridN/offshore_spectra_case_gridN.nc"
        ),
    )
    parser.add_argument(
        "--kps",
        type=str,
        default=None,
        help=(
            "Path to the kp coefficients file. Two variants exist: "
            "kp_coeffs_filtered_gridN.nc (500m mesh sites only) and "
            "kp_coefficients_averaged.nc (ALL output locations: grid + buoys + "
            "isobath + 3km + WHACS points). Default: first one found of "
            "gridN/outputs/kp_coeffs_filtered_gridN.nc, "
            "gridN/kp_coeffs_filtered_gridN.nc, "
            "gridN/outputs/kp_coefficients_averaged.nc"
        ),
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help=(
            "Path to the SWAN cases CSV (needed only for raw offshore spectra). "
            "Default: gridN/CASES/swan_cases_averaged.csv"
        ),
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag appended to output filenames (e.g. WHACS_44014).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to write reconstructed spectra to. "
            "Default: reconstructed_spectra/gridN next to this script."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="(Ignored, kept for backwards compatibility. BLAS uses all cores.)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="(Ignored, kept for backwards compatibility.)",
    )

    args = parser.parse_args()
    periods = parse_periods(args.periods)
    grid_num = args.grid

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    grid_dir = os.path.join(script_dir, f"grid{grid_num}")

    def first_existing(*candidates):
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]

    offshore_spectra_path = args.offshore or first_existing(
        os.path.join(grid_dir, f"offshore_spectra_case_grid{grid_num}.nc"),
        os.path.join(grid_dir, "outputs", f"offshore_spectra_case_grid{grid_num}.nc"),
    )
    if offshore_spectra_path and not os.path.exists(offshore_spectra_path) and os.path.exists(offshore_spectra_path + ".nc"):
        offshore_spectra_path = offshore_spectra_path + ".nc"
    kp_coeffs_path = args.kps or first_existing(
        os.path.join(grid_dir, "outputs", f"kp_coeffs_filtered_grid{grid_num}.nc"),
        os.path.join(grid_dir, f"kp_coeffs_filtered_grid{grid_num}.nc"),
        os.path.join(grid_dir, "outputs", "kp_coefficients_averaged.nc"),
    )
    cases_csv_path = args.cases or os.path.join(
        grid_dir, "CASES", "swan_cases_averaged.csv"
    )

    # Output directory: custom via --output-dir, or reconstructed_spectra/gridN
    output_dir = args.output_dir or os.path.join(
        script_dir, "reconstructed_spectra", f"grid{grid_num}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading offshore spectra and kp coefficients for grid {grid_num}...")
    print(f"  Offshore spectra: {offshore_spectra_path}")
    print(f"  KP coefficients: {kp_coeffs_path}")
    offshore_spectra = open_efth(offshore_spectra_path)
    kp_coeffs = xr.open_dataset(kp_coeffs_path).kps

    if "case_num" in offshore_spectra.dims:
        # Already case-binned: guarantee case ordering matches
        kp_coeffs = kp_coeffs.sel(case_num=offshore_spectra["case_num"])
        projection = None
    else:
        # Raw efth(time, freq, dir) spectrum: project onto the SWAN cases
        print(f"  Raw spectrum detected, projecting onto cases: {cases_csv_path}")
        projection = case_projection_indices(
            offshore_spectra, cases_csv_path, kp_coeffs["case_num"].values
        )

    t0 = time_mod.perf_counter()
    kp_matrix, kp_template = load_kp_matrix(kp_coeffs)
    print(
        f"KP matrix loaded: {kp_matrix.shape} float32 "
        f"({kp_matrix.nbytes / 1e9:.1f} GB) in {time_mod.perf_counter() - t0:.1f} s"
    )

    print(f"Periods to process: {periods}")

    tag = f"_{args.tag}" if args.tag else ""
    for period in periods:
        print(f"Processing period: {period}")
        t0 = time_mod.perf_counter()
        reconstructed_spectra = reconstruct_period(
            offshore_spectra, kp_matrix, kp_template, period, projection
        )
        t_compute = time_mod.perf_counter() - t0

        output_path = os.path.join(
            output_dir, f"reconstructed_spectra_grid{grid_num}{tag}_{period}.nc"
        )
        t0 = time_mod.perf_counter()
        reconstructed_spectra.to_netcdf(
            output_path,
            encoding={"efth": {"zlib": True, "complevel": 1}},
        )
        t_write = time_mod.perf_counter() - t0
        print(
            f"Saved: {output_path} "
            f"(compute: {t_compute:.1f} s, write: {t_write:.1f} s)"
        )

    print("Done!")


if __name__ == "__main__":
    main()
