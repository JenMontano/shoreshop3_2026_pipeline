#!/usr/bin/env python
"""
Fast reconstruction of onshore spectra for cyclone cases (BLAS matmul).

    onshore(time, dir, site, freq) = sum_c offshore(c, time) * kp(c, dir, site, freq)

Offshore input is a raw spectrum with cyclone IDs:
    efth(cyclone_id, time, freq, dir)
projected onto SWAN cases via swan_cases_averaged.csv (nearest freq/dir bins).

Per-cyclone checkpointing: one NetCDF per cyclone; already-written files are skipped.

Example (run from 04B_Cyclones_emulator/):
    conda run -n bluemath-dev python utils/reconstruct_spectra_cyclones_checkpointed.py \\
        --grid 1 --batch-size 8 --cyclone-start-index 0 --cyclone-end-index 100
"""

import argparse
import os
import time as time_mod
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

UTILS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UTILS_DIR.parent


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

    Returns (freq_idx, dir_idx, valid_mask), one entry per case in case_nums.
    """
    cases = pd.read_csv(cases_csv_path).set_index("case_num").loc[case_nums]
    case_freq = cases["freq"].to_numpy()
    case_dir = cases["dir"].to_numpy()

    spec_freq = spectrum["freq"].values
    spec_dir = spectrum["dir"].values

    freq_dist = np.abs(spec_freq[None, :] - case_freq[:, None])
    freq_idx = freq_dist.argmin(axis=1)

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


def first_existing(*candidates):
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else None


def load_offshore_cyclone_spectra(path):
    """
    Load offshore spectra with dims (cyclone_id, time, freq, dir).

    Accepts cyclone_id or case_num as the cyclone axis name, and sums over
    'part' if present.
    """
    spectra = open_efth(path)

    if "part" in spectra.dims:
        spectra = spectra.sum(dim="part")

    if "cyclone_id" not in spectra.dims and "case_num" in spectra.dims:
        if "freq" in spectra.dims and "dir" in spectra.dims:
            spectra = spectra.rename(case_num="cyclone_id")

    if "time" not in spectra.dims:
        raise ValueError("Expected a 'time' dimension in offshore spectra.")
    if "cyclone_id" not in spectra.dims:
        raise ValueError(
            "Expected a 'cyclone_id' (or 'case_num' with freq/dir) dimension "
            "in offshore spectra."
        )
    if "freq" not in spectra.dims or "dir" not in spectra.dims:
        raise ValueError(
            "Expected raw spectra with 'freq' and 'dir' dims "
            f"(got {spectra.dims})."
        )

    return spectra


def offshore_batch_to_case_matrix(offshore_batch, projection):
    """
    Project efth(cyclone_id, time, freq, dir) -> float32 (n_cyclones*n_time, case_num).

    NaNs (invalid constructed spectra) are treated as zeros.
    """
    freq_idx, dir_idx, valid = projection
    values = offshore_batch.transpose("cyclone_id", "time", "freq", "dir").values
    n_cyclones, n_time = values.shape[:2]
    matrix = values[:, :, freq_idx, dir_idx].astype(np.float32, copy=False)
    matrix[:, :, ~valid] = 0.0
    np.nan_to_num(matrix, copy=False, nan=0.0)
    return matrix.reshape(n_cyclones * n_time, -1), n_cyclones, n_time


def reconstruct_cyclone_batch(offshore_batch, kp_matrix, kp_template, projection):
    """
    Reconstruct onshore spectra for a batch of cyclones with one matmul.

    Returns DataArray efth(cyclone_id, time, dir, site, freq).
    """
    offshore_matrix, n_cyclones, n_time = offshore_batch_to_case_matrix(
        offshore_batch, projection
    )
    result = offshore_matrix @ kp_matrix

    shape = (
        n_cyclones,
        n_time,
        kp_template.sizes["dir"],
        kp_template.sizes["site"],
        kp_template.sizes["freq"],
    )
    reconstructed = xr.DataArray(
        result.reshape(shape),
        dims=("cyclone_id", "time", "dir", "site", "freq"),
        coords={
            "cyclone_id": offshore_batch["cyclone_id"],
            "time": offshore_batch["time"],
            "dir": kp_template["dir"],
            "site": kp_template["site"],
            "freq": kp_template["freq"],
        },
        name="efth",
    )
    extra_coords = {
        name: coord
        for name, coord in kp_template.coords.items()
        if name not in reconstructed.coords
        and set(coord.dims) <= set(reconstructed.dims)
    }
    return reconstructed.assign_coords(extra_coords)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fast (BLAS) reconstruction of cyclone spectra with per-cyclone "
            "checkpointing."
        )
    )
    parser.add_argument(
        "--grid",
        type=int,
        required=True,
        choices=[1, 2, 3, 4],
        help="Grid number (1, 2, 3, or 4)",
    )
    parser.add_argument(
        "--input-spectra",
        type=str,
        default=None,
        help=(
            "Path to input cyclone spectra "
            "(default: grid{N}/inputs/spectra_point_{N}.nc)"
        ),
    )
    parser.add_argument(
        "--kp-coeffs",
        "--kps",
        dest="kp_coeffs",
        type=str,
        default=None,
        help=(
            "Path to kp coefficients NetCDF. Default: first existing of "
            "grid{N}/outputs/kp_coeffs_filtered_grid{N}.nc, "
            "grid{N}/kp_coeffs_filtered_grid{N}.nc, "
            "grid{N}/outputs/kp_coefficients_averaged.nc"
        ),
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help=(
            "Path to SWAN cases CSV (case_num, freq, dir). "
            "Default: grid{N}/CASES/swan_cases_averaged.csv"
        ),
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="Base path for grid{N} dirs (default: 04B project root)",
    )
    parser.add_argument(
        "--kp-var",
        type=str,
        default="kps",
        help="Variable name in kp coefficients NetCDF (default: kps)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=(
            "Number of cyclones per matmul batch (default: 8). "
            "Increase for throughput; decrease if you hit MemoryError."
        ),
    )
    parser.add_argument(
        "--cyclone-start-index",
        type=int,
        default=None,
        help="Start index along cyclone_id (0-based, inclusive). Default: 0.",
    )
    parser.add_argument(
        "--cyclone-end-index",
        type=int,
        default=None,
        help=(
            "End index along cyclone_id (0-based, exclusive). "
            "Default: total number of cyclones."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help=(
            "Prefix for per-cyclone files in grid{N}/outputs. "
            "Default: reconstructed_spectra_grid{N}_cyclone"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: grid{N}/outputs)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="(Ignored; BLAS uses all available cores.)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="(Ignored; kept for backwards compatibility.)",
    )

    args = parser.parse_args()
    grid_num = args.grid

    base_path = args.base_path if args.base_path else str(PROJECT_ROOT)
    grid_dir = os.path.join(base_path, f"grid{grid_num}")
    inputs_dir = os.path.join(grid_dir, "inputs")
    outputs_dir = args.output_dir or os.path.join(grid_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    offshore_path = args.input_spectra or os.path.join(
        inputs_dir, f"spectra_point_{grid_num}.nc"
    )
    kp_path = args.kp_coeffs or first_existing(
        os.path.join(outputs_dir, f"kp_coeffs_filtered_grid{grid_num}.nc"),
        os.path.join(grid_dir, f"kp_coeffs_filtered_grid{grid_num}.nc"),
        os.path.join(outputs_dir, "kp_coefficients_averaged.nc"),
        os.path.join(grid_dir, "outputs", "kp_coefficients_averaged.nc"),
    )
    cases_csv_path = args.cases or first_existing(
        os.path.join(grid_dir, "CASES", "swan_cases_averaged.csv"),
        os.path.join(
            os.path.expanduser("~"),
            "BinWaves_missed_feb",
            f"grid{grid_num}",
            "CASES",
            "swan_cases_averaged.csv",
        ),
    )

    print(f"Grid {grid_num} (BLAS matmul)")
    print(f"  Offshore cyclone spectra: {offshore_path}")
    print(f"  KP coefficients:         {kp_path}")
    print(f"  SWAN cases CSV:          {cases_csv_path}")

    if not os.path.exists(offshore_path):
        raise FileNotFoundError(f"Offshore spectra not found: {offshore_path}")
    if not os.path.exists(kp_path):
        raise FileNotFoundError(f"KP coefficients not found: {kp_path}")
    if not os.path.exists(cases_csv_path):
        raise FileNotFoundError(
            f"SWAN cases CSV not found: {cases_csv_path}. "
            "Pass --cases path/to/swan_cases_averaged.csv"
        )

    offshore_spectra = load_offshore_cyclone_spectra(offshore_path)
    kp_ds = xr.open_dataset(kp_path)
    kp_var = args.kp_var if args.kp_var in kp_ds.data_vars else list(kp_ds.data_vars)[0]
    kp_coeffs = kp_ds[kp_var]

    print("  Projecting offshore (freq, dir) onto SWAN cases...")
    projection = case_projection_indices(
        offshore_spectra, cases_csv_path, kp_coeffs["case_num"].values
    )

    t0 = time_mod.perf_counter()
    kp_matrix, kp_template = load_kp_matrix(kp_coeffs)
    print(
        f"  KP matrix loaded: {kp_matrix.shape} float32 "
        f"({kp_matrix.nbytes / 1e9:.2f} GB) in {time_mod.perf_counter() - t0:.1f} s"
    )

    all_cyclone_ids = offshore_spectra["cyclone_id"].values
    total_cyclones = len(all_cyclone_ids)
    start_idx = args.cyclone_start_index if args.cyclone_start_index is not None else 0
    end_idx = (
        args.cyclone_end_index if args.cyclone_end_index is not None else total_cyclones
    )
    if start_idx < 0 or start_idx > total_cyclones:
        raise ValueError(
            f"cyclone-start-index={start_idx} is out of bounds for "
            f"{total_cyclones} cyclones"
        )
    if end_idx < start_idx or end_idx > total_cyclones:
        raise ValueError(
            f"cyclone-end-index={end_idx} must be between start ({start_idx}) "
            f"and total cyclones ({total_cyclones})"
        )

    cyclone_ids = all_cyclone_ids[start_idx:end_idx]
    n_cyclones = len(cyclone_ids)
    n_time = offshore_spectra.sizes["time"]
    print(
        f"  Cyclones to process: {n_cyclones} (indices {start_idx}..{end_idx - 1}, "
        f"total {total_cyclones}; each with {n_time} hours)"
    )

    output_prefix = args.output_prefix or (
        f"reconstructed_spectra_grid{grid_num}_cyclone"
    )
    batch_size = max(1, int(args.batch_size))

    for batch_start in range(0, n_cyclones, batch_size):
        batch_end = min(batch_start + batch_size, n_cyclones)
        batch_to_process = []
        for local_idx in range(batch_start, batch_end):
            cid = cyclone_ids[local_idx]
            global_idx = start_idx + local_idx
            out_name = f"{output_prefix}_{int(global_idx)}.nc"
            output_path = os.path.join(outputs_dir, out_name)
            if os.path.exists(output_path):
                print(
                    f"  Skipping cyclone_id={cid} "
                    f"(already exists: {out_name}, global index {global_idx})"
                )
                continue
            batch_to_process.append((cid, global_idx, out_name, output_path))

        if not batch_to_process:
            continue

        batch_cids = [cid for (cid, _, _, _) in batch_to_process]
        print(
            f"Processing batch of {len(batch_cids)} cyclones: "
            f"{batch_cids[0]}..{batch_cids[-1]}"
        )

        t0 = time_mod.perf_counter()
        off_batch = offshore_spectra.sel(cyclone_id=batch_cids)
        off_batch = off_batch.load()
        rec_batch = reconstruct_cyclone_batch(
            off_batch, kp_matrix, kp_template, projection
        )
        t_compute = time_mod.perf_counter() - t0

        for cid, global_idx, out_name, output_path in batch_to_process:
            print(
                f"  Writing cyclone_id={cid} "
                f"(global index {global_idx}/{total_cyclones - 1}) "
                f"to {out_name}"
            )
            t0 = time_mod.perf_counter()
            rec_one = rec_batch.sel(cyclone_id=cid)
            rec_one.to_netcdf(
                output_path,
                encoding={"efth": {"zlib": True, "complevel": 1}},
            )
            t_write = time_mod.perf_counter() - t0
            print(
                f"    Saved: {output_path} "
                f"(batch compute: {t_compute:.1f} s, write: {t_write:.1f} s)"
            )

    print("Done (checkpointed per cyclone, BLAS matmul).")


if __name__ == "__main__":
    main()
