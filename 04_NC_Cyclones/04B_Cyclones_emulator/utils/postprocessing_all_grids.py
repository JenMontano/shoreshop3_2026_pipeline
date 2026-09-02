import os
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import display
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*'M' is deprecated and will be removed in a future version.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*The specified chunks separate the stored chunks.*",
    category=UserWarning,
)

# ---------------- CONFIG ----------------
# Base directory: parent of utils/ = Cyclones_NC project root
_BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = str(_BASE_DIR / "outputs" / "partitions_cyclones")
VAR_NAME = "hs"      # change if your variable has a different name
TIME_NAME = "time"   # change if your time coordinate is named differently

# size (in samples) of the window used to detect "fixed" data
ROLLING_WINDOW = 30       # e.g. 30 consecutive samples
ROLLING_STD_EPS = 1e-6    # std below this -> considered "nearly constant"

# monthly variance threshold: months with var below this are flagged
MONTHLY_VAR_EPS = 1e-4
# ----------------------------------------


def longest_consecutive_true(mask: np.ndarray) -> int:
    """Return the longest run of True in a 1D boolean mask."""
    if mask.size == 0:
        return 0
    # Ensure 1D
    mask = mask.ravel()
    # Differences between consecutive values
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]

    if starts.size == 0:
        return 0
    lengths = ends - starts
    return int(lengths.max())


def analyze_hs_file(
    path: str,
    var_name: str,
    time_name: str,
    rolling_window: int,
    rolling_std_eps: float,
    monthly_var_eps: float,
):
    ds = xr.open_dataset(path)
    if var_name not in ds.data_vars:
        # fallback: first data var
        var_name = list(ds.data_vars)[0]
        print(f"  WARNING: requested variable not found in {os.path.basename(path)}, using {var_name}")

    da = ds[var_name]
    if time_name not in ds.coords:
        # best effort to find time coordinate
        for cand in ["time", "Time", "t"]:
            if cand in ds.coords:
                time_name = cand
                break
        else:
            raise ValueError(f"No time coordinate named '{time_name}' (or common aliases) in {path}")

    da = da.load()  # load into memory for simplicity
    n_total = da.size

    # --- NaNs ---
    nan_mask = np.isnan(da.values)
    n_nan = int(nan_mask.sum())
    frac_nan = n_nan / n_total if n_total > 0 else np.nan
    max_consec_nan = longest_consecutive_true(nan_mask)

    # --- Zeros ---
    zero_mask = np.isfinite(da.values) & (da.values == 0)
    n_zero = int(zero_mask.sum())
    frac_zero = n_zero / n_total if n_total > 0 else np.nan
    max_consec_zero = longest_consecutive_true(zero_mask)

    # --- Fixed / nearly-constant periods (rolling std) ---
    # work on 1D along time (if more dims, flatten them except time)
    if da.ndim > 1:
        # move time to axis 0 and flatten others
        da_1d = da.stack(_space=[d for d in da.dims if d != time_name]).mean("_space")
    else:
        da_1d = da

    # rolling std along time
    roll_std = da_1d.rolling({time_name: rolling_window}, center=True).std()
    const_mask = np.isfinite(roll_std.values) & (roll_std.values < rolling_std_eps)
    max_consec_const = longest_consecutive_true(const_mask)

    # --- Monthly variance ---
    # resample to calendar months (month-end, avoids 'M' deprecation warning)
    monthly = da_1d.resample({time_name: "1ME"})
    monthly_var = monthly.var(dim=time_name)
    low_var_months = monthly_var.where(monthly_var < monthly_var_eps, drop=True)
    n_low_var_months = low_var_months[time_name].size

    # summarize "weirdness"
    issues = []
    if n_nan > 0:
        issues.append(f"NaNs (total={n_nan}, frac={frac_nan:.4f}, max_consec={max_consec_nan})")
    if n_zero > 0:
        issues.append(f"Zeros (total={n_zero}, frac={frac_zero:.4f}, max_consec={max_consec_zero})")
    if max_consec_const > 0:
        issues.append(f"Long nearly-constant periods (max_consec={max_consec_const})")
    if n_low_var_months > 0:
        first_low = pd.to_datetime(str(low_var_months[time_name].values[0])).strftime("%Y-%m")
        last_low = pd.to_datetime(str(low_var_months[time_name].values[-1])).strftime("%Y-%m")
        issues.append(
            f"Low monthly variance in {n_low_var_months} month(s) "
            f"(from {first_low} to {last_low}, var<{monthly_var_eps})"
        )

    ds.close()

    return {
        "file": os.path.basename(path),
        "n_total": n_total,
        "n_nan": n_nan,
        "frac_nan": frac_nan,
        "max_consec_nan": max_consec_nan,
        "n_zero": n_zero,
        "frac_zero": frac_zero,
        "max_consec_zero": max_consec_zero,
        "max_consec_const": max_consec_const,
        "n_low_var_months": n_low_var_months,
        "issues": "; ".join(issues) if issues else "OK",
    }


def run_hs_postprocessing(
    data_dir: str = DATA_DIR,
    var_name: str = VAR_NAME,
    time_name: str = TIME_NAME,
    rolling_window: int = ROLLING_WINDOW,
    rolling_std_eps: float = ROLLING_STD_EPS,
    monthly_var_eps: float = MONTHLY_VAR_EPS,
    save_csv_path: str | None = None,
    verbose: bool = True,
):
    """
    Run quality checks on all yearly hs NetCDF files in a directory.

    Returns a pandas DataFrame with one row per file/year.
    """
    # --------- Run over all yearly files and build a yearly summary ----------
    all_files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".nc")
    )

    results = []
    for path in all_files:
        if verbose:
            print(f"Analyzing {os.path.basename(path)} ...")
        res = analyze_hs_file(
            path,
            var_name=var_name,
            time_name=time_name,
            rolling_window=rolling_window,
            rolling_std_eps=rolling_std_eps,
            monthly_var_eps=monthly_var_eps,
        )
        results.append(res)

    summary_df = pd.DataFrame(results)

    # extract year from filenames like "nc_hs_grid1_1980.nc"
    summary_df["year"] = summary_df["file"].str.extract(r"(\d{4})").astype(int)

    # order by year
    summary_df = summary_df.sort_values("year").reset_index(drop=True)

    # Show compact yearly summary with "weirdness" description
    summary_cols = [
        "year",
        "n_total",
        "n_nan",
        "frac_nan",
        "max_consec_nan",
        "n_zero",
        "frac_zero",
        "max_consec_zero",
        "max_consec_const",
        "n_low_var_months",
        "issues",
    ]

    if verbose:
        display(summary_df[summary_cols])
        print("Done.")

    if save_csv_path is not None:
        summary_df.to_csv(save_csv_path, index=False)

    return summary_df


if __name__ == "__main__":
    run_hs_postprocessing()


def convert_output_variables_to_float32(
    base_dir: str | None = None,
    out_root: str | None = None,
    overwrite: bool = False,
    compress: bool = True,
    complevel: int = 4,
    year: int | None = None,
    variable: str | list[str] | None = None,
):
    """
    Convert all NetCDF files under `base_dir` to float32 and save into
    `out_root/<variable_subfolder>/file.nc`.

    - Assumes structure: base_dir / <var_name> / *.nc
    - Skips the float_32 output directory if it exists inside base_dir.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing variable subdirectories
    out_root : str | None
        Output root directory (default: base_dir/float_32)
    overwrite : bool
        Whether to overwrite existing files
    compress : bool
        Whether to compress output files
    complevel : int
        Compression level (1-9)
    year : int | None
        If specified, only process files containing this year in filename
    variable : str | list[str] | None
        If specified, only process these variable(s). 
        - None: process all variables
        - str: process single variable (e.g., "hs")
        - list: process multiple variables (e.g., ["hs", "tp", "dm"])
    """
    if base_dir is None:
        base_dir = str(_BASE_DIR / "grid1" / "outputs" / "output_variables")
    if out_root is None:
        out_root = os.path.join(base_dir, "float_32")

    os.makedirs(out_root, exist_ok=True)

    # Normalize variable parameter to a set for easy checking
    if variable is None:
        variables_to_process = None  # Process all
    elif isinstance(variable, str):
        variables_to_process = {variable}
    elif isinstance(variable, list):
        variables_to_process = set(variable)
    else:
        raise ValueError(f"variable parameter must be str, list[str], or None, got {type(variable)}")

    float32_dtype = np.float32
    skipped_existing = 0
    converted = 0
    tasks: list[tuple[str, str, str, str]] = []

    # First, collect all files to process
    for sub in sorted(os.listdir(base_dir)):
        in_var_dir = os.path.join(base_dir, sub)

        # skip non-directories and the output root itself
        if not os.path.isdir(in_var_dir):
            continue
        if os.path.abspath(in_var_dir) == os.path.abspath(out_root):
            continue
        
        # Filter by variable if specified
        if variables_to_process is not None and sub not in variables_to_process:
            continue

        out_var_dir = os.path.join(out_root, sub)
        os.makedirs(out_var_dir, exist_ok=True)

        for fname in sorted(os.listdir(in_var_dir)):
            if not fname.endswith(".nc"):
                continue

            # If a specific year is requested, only process files whose name contains that year
            if year is not None and str(year) not in fname:
                continue

            in_path = os.path.join(in_var_dir, fname)
            out_path = os.path.join(out_var_dir, fname)
            tasks.append((sub, fname, in_path, out_path))

    # Progress bar over all files
    for sub, fname, in_path, out_path in tqdm(tasks, desc="Converting NetCDFs to float32"):
        if (not overwrite) and os.path.exists(out_path):
            skipped_existing += 1
            continue
        print(f"Converting: {in_path}")
        ds = xr.open_dataset(in_path)
        ds.load()  # ensure data in memory before writing new file

        # Copy dataset and cast all floating-point data variables to float32
        new_ds = ds.copy()
        encoding: dict[str, dict] = {}

        # Allowed encoding keys for netCDF4 backend (subset of what xarray documents)
        valid_encoding_keys = {
            "contiguous",
            "blosc_shuffle",
            "significant_digits",
            "dtype",
            "quantize_mode",
            "zlib",
            "complevel",
            "least_significant_digit",
            "chunksizes",
            "compression",
            "endian",
            "szip_pixels_per_block",
            "fletcher32",
            "_FillValue",
            "shuffle",
        }

        for var in ds.data_vars:
            da = ds[var]
            if np.issubdtype(da.dtype, np.floating):
                new_ds[var] = da.astype(float32_dtype)
                var_enc = da.encoding.copy()
                var_enc["dtype"] = "float32"
                if compress:
                    var_enc.setdefault("zlib", True)
                    var_enc.setdefault("complevel", complevel)
                # drop keys that netCDF4 backend doesn't accept
                cleaned_enc = {k: v for k, v in var_enc.items() if k in valid_encoding_keys}
                encoding[var] = cleaned_enc

        # preserve coordinates encodings as far as possible
        for coord in ds.coords:
            da = ds[coord]
            coord_enc = da.encoding.copy()
            cleaned_coord_enc = {k: v for k, v in coord_enc.items() if k in valid_encoding_keys}
            # usually coords remain same dtype
            encoding.setdefault(coord, cleaned_coord_enc)

        new_ds.to_netcdf(out_path, encoding=encoding)
        ds.close()
        new_ds.close()
        converted += 1

    # Print summary
    print(f"Finished converting NetCDFs to float32 under {base_dir}")
    if variables_to_process is not None:
        var_list = sorted(variables_to_process)
        if len(var_list) == 1:
            print(f"  Variable: {var_list[0]}")
        else:
            print(f"  Variables: {', '.join(var_list)}")
    else:
        print(f"  Variables: all")
    if year is not None:
        print(f"  Year: {year}")
    print(f"  Converted files: {converted}")
    print(f"  Skipped existing (overwrite={overwrite}): {skipped_existing}")


def concatenate_float32_grids_by_variable(
    base_path,
    grids,
    output_dir,
    reference_grid="grid1",
    verbose=True,
    variable=None,
):
    """
    Concatenate all yearly float32 files for each variable of each grid into single files.
    
    This function works with files in the float_32 subdirectory, checking for gaps
    and resuming from last merged year.
    
    Parameters
    ----------
    base_path : str or Path
        Base path to the grid directories (e.g., "/path/to/ShoreShop2026")
    grids : dict
        Dictionary mapping grid names to their configurations
    output_dir : str or Path
        Output directory for concatenated files
    reference_grid : str, optional
        Grid name to use for discovering variable names (default: "grid1")
    verbose : bool, optional
        Whether to print progress messages (default: True)
    variable : str | list[str] | None, optional
        If specified, only process these variable(s).
        - None: process all variables (default)
        - str: process single variable (e.g., "hs")
        - list: process multiple variables (e.g., ["hs", "tp", "dm"])
    
    Returns
    -------
    None
        Files are saved to output_dir/concatenated_grids_float32/
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import xarray as xr
    from tqdm import tqdm
    
    # Convert to Path objects
    base_path = Path(base_path)
    output_dir = Path(output_dir)
    
    # Output directory for concatenated files
    concatenated_dir = output_dir / "concatenated_grids_float32"
    concatenated_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"{'='*80}")
        print(f"CONCATENATING FLOAT32 GRID FILES BY VARIABLE")
        print(f"{'='*80}")
        print(f"Output directory: {concatenated_dir}\n")
    
    # Discover all variables from reference grid
    ref_grid_var_dir = base_path / reference_grid / "outputs" / "output_variables" / "float_32"
    if not ref_grid_var_dir.exists():
        raise ValueError(f"Reference grid float_32 directory not found: {ref_grid_var_dir}")
    
    # Get all variable directories
    all_variable_names = [d.name for d in ref_grid_var_dir.iterdir() if d.is_dir()]
    all_variable_names = sorted(all_variable_names)
    
    # Filter by variable parameter if specified
    if variable is None:
        variable_names = all_variable_names
    elif isinstance(variable, str):
        if variable not in all_variable_names:
            raise ValueError(f"Variable '{variable}' not found in {ref_grid_var_dir}. Available: {', '.join(all_variable_names)}")
        variable_names = [variable]
    elif isinstance(variable, list):
        variable_names = []
        for v in variable:
            if v not in all_variable_names:
                if verbose:
                    print(f"  ⚠ Warning: Variable '{v}' not found, skipping")
                continue
            variable_names.append(v)
        if len(variable_names) == 0:
            raise ValueError(f"None of the specified variables found. Available: {', '.join(all_variable_names)}")
        variable_names = sorted(variable_names)
    else:
        raise ValueError(f"variable parameter must be str, list[str], or None, got {type(variable)}")
    
    if verbose:
        if variable is None:
            print(f"Found {len(variable_names)} variables: {', '.join(variable_names)}\n")
        elif len(variable_names) == 1:
            print(f"Processing variable: {variable_names[0]}\n")
        else:
            print(f"Processing {len(variable_names)} variables: {', '.join(variable_names)}\n")
    
    # Process each grid
    for grid_name in grids.keys():
        if verbose:
            print(f"{'='*80}")
            print(f"Processing grid: {grid_name}")
            print(f"{'='*80}")
        
        grid_var_dir = base_path / grid_name / "outputs" / "output_variables" / "float_32"
        
        if not grid_var_dir.exists():
            if verbose:
                print(f"  ⚠ Grid float_32 directory not found: {grid_var_dir}")
            continue
        
        # Process each variable
        for var_name in variable_names:
            var_dir = grid_var_dir / var_name
            
            if not var_dir.exists():
                if verbose:
                    print(f"  ⚠ Variable directory not found: {var_dir}")
                continue
            
            # Find all yearly files for this variable
            pattern = f"nc_{var_name}_{grid_name}_*.nc"
            yearly_files = sorted(var_dir.glob(pattern))
            
            if len(yearly_files) == 0:
                if verbose:
                    print(f"  ⚠ No files found for {var_name} in {grid_name}")
                continue
            
            # Extract years from filenames
            year_files = {}
            for nc_file in yearly_files:
                try:
                    # Extract year from filename: nc_{var_name}_{grid_name}_{year}.nc
                    year_str = nc_file.stem.split('_')[-1]
                    if year_str.isdigit():
                        year = int(year_str)
                        year_files[year] = nc_file
                except Exception as e:
                    if verbose:
                        print(f"    ⚠ Could not parse year from {nc_file.name}: {e}")
                    continue
            
            if len(year_files) == 0:
                if verbose:
                    print(f"  ⚠ No valid yearly files found for {var_name} in {grid_name}")
                continue
            
            years = sorted(year_files.keys())
            if verbose:
                print(f"\n  Variable: {var_name}")
                print(f"    Found {len(years)} years: {years[0]} to {years[-1]}")
            
            # Check for gaps in years
            expected_years = set(range(min(years), max(years) + 1))
            missing_years = sorted(expected_years - set(years))
            if missing_years:
                if verbose:
                    print(f"    ⚠ WARNING: Missing years: {missing_years}")
            else:
                if verbose:
                    print(f"    ✓ No gaps detected (continuous from {years[0]} to {years[-1]})")
            
            # Output file name
            output_file = concatenated_dir / f"{var_name}_{grid_name}.nc"
            
            # Check if concatenated file already exists
            start_year = None
            if output_file.exists():
                if verbose:
                    print(f"    Existing concatenated file found: {output_file}")
                try:
                    # Open existing file and check last year
                    ds_existing = xr.open_dataset(output_file)
                    if 'time' in ds_existing.coords:
                        last_time = pd.Timestamp(ds_existing.time.values[-1])
                        last_year = last_time.year
                        if verbose:
                            print(f"    Last year in existing file: {last_year}")
                        
                        # Find the next year to start from
                        if last_year in years:
                            # Find index of last_year in sorted years
                            last_year_idx = years.index(last_year)
                            if last_year_idx < len(years) - 1:
                                start_year = years[last_year_idx + 1]
                                if verbose:
                                    print(f"    Resuming from year: {start_year}")
                            else:
                                if verbose:
                                    print(f"    ✓ File already contains all available years. Skipping.")
                                ds_existing.close()
                                continue
                        else:
                            # Last year not in current files, start from beginning
                            if verbose:
                                print(f"    ⚠ Last year ({last_year}) not found in current files. Rebuilding from start.")
                            start_year = None
                    else:
                        if verbose:
                            print(f"    ⚠ No time coordinate found. Rebuilding from start.")
                        start_year = None
                    
                    ds_existing.close()
                except Exception as e:
                    if verbose:
                        print(f"    ⚠ Error reading existing file: {e}")
                        print(f"    Rebuilding from start.")
                    start_year = None
            
            # Filter years to process
            if start_year is not None:
                years_to_process = [y for y in years if y >= start_year]
                if len(years_to_process) == 0:
                    if verbose:
                        print(f"    ✓ No new years to add. Skipping.")
                    continue
                if verbose:
                    print(f"    Processing {len(years_to_process)} new year(s): {years_to_process}")
                
                # Load existing dataset and append new years
                if verbose:
                    print(f"    Loading existing file...")
                ds_combined = xr.open_dataset(output_file)
                existing_times = pd.to_datetime(ds_combined.time.values)
                last_existing_time = existing_times[-1]
                
                # Load and concatenate new years
                datasets_to_append = []
                for year in tqdm(years_to_process, desc=f"    Loading {var_name} files", disable=not verbose):
                    nc_file = year_files[year]
                    try:
                        ds_year = xr.open_dataset(nc_file)
                        
                        # Check if time coordinate exists (some files might have no data)
                        if 'time' not in ds_year.coords and 'time' not in ds_year.dims:
                            if verbose:
                                print(f"      ⚠ Skipping {nc_file.name}: no time coordinate (file may be empty)")
                            ds_year.close()
                            continue
                        
                        if 'time' in ds_year.coords and len(ds_year.coords['time']) == 0:
                            if verbose:
                                print(f"      ⚠ Skipping {nc_file.name}: empty time coordinate (no data)")
                            ds_year.close()
                            continue
                        
                        year_times = pd.to_datetime(ds_year.time.values)
                        
                        # Check for overlap with existing data
                        if len(year_times) > 0 and year_times[0] <= last_existing_time:
                            if verbose:
                                print(f"      ⚠ Skipping {nc_file.name}: overlaps with existing data (starts at {year_times[0]}, existing ends at {last_existing_time})")
                            ds_year.close()
                            continue
                        
                        datasets_to_append.append(ds_year)
                    except Exception as e:
                        if verbose:
                            print(f"      ⚠ Error loading {nc_file.name}: {e}")
                        continue
                
                if len(datasets_to_append) == 0:
                    if verbose:
                        print(f"    ⚠ No valid datasets to append. Skipping.")
                    ds_combined.close()
                    continue
                
                # Concatenate new datasets
                ds_new = xr.concat(datasets_to_append, dim='time')
                
                # Concatenate with existing
                ds_combined = xr.concat([ds_combined, ds_new], dim='time')
                
                # Sort by time
                ds_combined = ds_combined.sortby('time')
                
                # Close individual datasets
                for ds in datasets_to_append:
                    ds.close()
                ds_new.close()
                
            else:
                # Build from scratch
                if verbose:
                    print(f"    Building concatenated file from {len(years)} year(s)...")
                
                # Load all datasets and validate they have time coordinate
                datasets = []
                invalid_files = []
                for year in tqdm(years, desc=f"    Loading {var_name} files", disable=not verbose):
                    nc_file = year_files[year]
                    try:
                        ds = xr.open_dataset(nc_file)
                        # Validate that dataset has 'time' coordinate
                        if 'time' not in ds.coords and 'time' not in ds.dims:
                            if verbose:
                                print(f"      ⚠ Skipping {nc_file.name}: missing 'time' coordinate")
                            invalid_files.append((year, nc_file.name, "missing time coordinate"))
                            ds.close()
                            continue
                        # Check if time coordinate has data
                        if 'time' in ds.coords and len(ds.coords['time']) == 0:
                            if verbose:
                                print(f"      ⚠ Skipping {nc_file.name}: empty 'time' coordinate")
                            invalid_files.append((year, nc_file.name, "empty time coordinate"))
                            ds.close()
                            continue
                        datasets.append(ds)
                    except Exception as e:
                        if verbose:
                            print(f"      ⚠ Error loading {nc_file.name}: {e}")
                        invalid_files.append((year, nc_file.name, str(e)))
                        continue
                
                if len(datasets) == 0:
                    if verbose:
                        print(f"    ⚠ No valid datasets found. Skipping.")
                        if invalid_files:
                            print(f"    Invalid files ({len(invalid_files)}):")
                            for year, fname, reason in invalid_files[:10]:  # Show first 10
                                print(f"      {year}: {fname} - {reason}")
                            if len(invalid_files) > 10:
                                print(f"      ... and {len(invalid_files) - 10} more")
                    continue
                
                if invalid_files and verbose:
                    print(f"    ⚠ Warning: {len(invalid_files)} file(s) skipped due to missing/invalid time coordinate")
                
                # Validate all datasets have time coordinate before concatenation
                datasets_with_time = []
                for i, ds in enumerate(datasets):
                    if 'time' in ds.coords or 'time' in ds.dims:
                        datasets_with_time.append(ds)
                    else:
                        if verbose:
                            year = years[i] if i < len(years) else "unknown"
                            print(f"      ⚠ Skipping dataset {i} (year {year}): missing time coordinate")
                        ds.close()
                
                if len(datasets_with_time) == 0:
                    if verbose:
                        print(f"    ⚠ No datasets with valid time coordinate found. Skipping.")
                    continue
                
                if len(datasets_with_time) < len(datasets):
                    if verbose:
                        print(f"    ⚠ Warning: {len(datasets) - len(datasets_with_time)} dataset(s) excluded due to missing time coordinate")
                    datasets = datasets_with_time
                
                # Concatenate along time dimension
                if verbose:
                    print(f"    Concatenating {len(datasets)} datasets...")
                ds_combined = xr.concat(datasets, dim='time')
                
                # Sort by time
                ds_combined = ds_combined.sortby('time')
                
                # Close individual datasets
                for ds in datasets:
                    ds.close()
            
            # Verify continuity
            time_values = pd.to_datetime(ds_combined.time.values)
            time_diffs = np.diff(time_values)
            
            # Auto-detect expected time interval (use median of first 1000 differences)
            sample_size = min(1000, len(time_diffs))
            if sample_size > 0:
                expected_diff = pd.Timedelta(np.median(time_diffs[:sample_size]))
                if verbose:
                    print(f"    Detected time interval: {expected_diff}")
            else:
                expected_diff = pd.Timedelta(hours=1)  # Fallback to hourly
            
            # Check for large gaps (more than 2x expected interval)
            large_gaps = np.where(time_diffs > 2 * expected_diff)[0]
            if len(large_gaps) > 0:
                if verbose:
                    print(f"    ⚠ WARNING: Found {len(large_gaps)} large gaps in time series:")
                    for gap_idx in large_gaps[:5]:  # Show first 5 gaps
                        gap_start = time_values[gap_idx]
                        gap_end = time_values[gap_idx + 1]
                        gap_duration = time_diffs[gap_idx]
                        print(f"      Gap: {gap_start} to {gap_end} (duration: {gap_duration})")
                    if len(large_gaps) > 5:
                        print(f"      ... and {len(large_gaps) - 5} more gaps")
            else:
                if verbose:
                    print(f"    ✓ Time series is continuous")
            
            # Save concatenated file with float32 encoding and compression
            if verbose:
                print(f"    Saving concatenated file (float32, compressed)...")
            try:
                # Remove existing file if rebuilding from scratch
                if start_year is None and output_file.exists():
                    output_file.unlink()
                
                # Prepare encoding to ensure float32 and compression
                encoding = {}
                valid_encoding_keys = {
                    "zlib", "complevel", "shuffle", "fletcher32", "contiguous",
                    "chunksizes", "dtype", "_FillValue"
                }
                
                # Set encoding for all data variables (ensure float32 and compression)
                for var_name in ds_combined.data_vars:
                    var = ds_combined[var_name]
                    # Ensure float32 dtype
                    if np.issubdtype(var.dtype, np.floating) and var.dtype != np.float32:
                        ds_combined[var_name] = var.astype(np.float32)
                    
                    # Set encoding with compression
                    var_enc = var.encoding.copy() if hasattr(var, 'encoding') and var.encoding else {}
                    var_enc["dtype"] = "float32"
                    var_enc["zlib"] = True
                    var_enc["complevel"] = 4  # Compression level (1-9, 4 is a good balance)
                    var_enc["shuffle"] = True  # Enable shuffle filter for better compression
                    
                    # Clean encoding to only include valid keys
                    cleaned_enc = {k: v for k, v in var_enc.items() if k in valid_encoding_keys}
                    encoding[var_name] = cleaned_enc
                
                # Preserve coordinate encodings
                for coord_name in ds_combined.coords:
                    coord = ds_combined[coord_name]
                    coord_enc = coord.encoding.copy() if hasattr(coord, 'encoding') and coord.encoding else {}
                    cleaned_coord_enc = {k: v for k, v in coord_enc.items() if k in valid_encoding_keys}
                    encoding.setdefault(coord_name, cleaned_coord_enc)
                
                # Save with encoding
                ds_combined.to_netcdf(output_file, encoding=encoding)
                file_size_mb = output_file.stat().st_size / (1024 * 1024)
                if verbose:
                    print(f"    ✓ Saved: {output_file.name} ({file_size_mb:.1f} MB)")
                    print(f"      Time range: {pd.Timestamp(time_values[0])} to {pd.Timestamp(time_values[-1])}")
                    print(f"      Total timesteps: {len(time_values)}")
                    print(f"      Format: float32, compressed (zlib, level 4)")
            except Exception as e:
                if verbose:
                    print(f"    ✗ Error saving file: {e}")
                    import traceback
                    traceback.print_exc()
            finally:
                ds_combined.close()
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"CONCATENATION COMPLETE")
        print(f"{'='*80}")
        if variable is not None:
            if isinstance(variable, str):
                print(f"  Variable processed: {variable}")
            else:
                print(f"  Variables processed: {', '.join(sorted(variable_names))}")
        else:
            print(f"  Variables processed: all ({len(variable_names)} variables)")
        print(f"\nConcatenated float32 files saved to: {concatenated_dir}")
        print(f"You can now use these files for merging grids.")
        print(f"\nNote: Check the warnings above for:")
        print(f"  - Missing years in file sequences")
        print(f"  - Files skipped due to missing/invalid time coordinates")
        print(f"  - Gaps in time series (missing dates)")



def crop_concatenated_files_by_spatial_mask(
    input_dir=None,
    output_dir=None,
    shoreline_geojson_file=None,
    outer_boundary_line_coords=None,
    exclusion_polygon_coords=None,
    exclude_coordinates=None,
    buoy_coordinates=None,
    variable=None,
    grid=None,
    verbose=True,
):
    """
    Apply spatial masking to concatenated float32 files and save cropped results.
    
    This function is based on the notebook code and:
    - Reads coordinates from coord_x/coord_y if lon/lat don't exist
    - Renames coord_x/coord_y to lon/lat before processing
    - Applies spatial mask (seaward of boundary, outside shoreline, etc.)
    - Drops coord_x, coord_y, and weight variables
    - Saves cropped files with float32 compression
    
    Parameters
    ----------
    input_dir : str or Path
        Directory containing concatenated float32 files (default: outputs/concatenated_grids_float32)
    output_dir : str or Path
        Output directory for cropped files (default: outputs/cropped_variables)
    shoreline_geojson_file : str or Path
        Path to shoreline GeoJSON file
    outer_boundary_line_coords : list of [lon, lat] pairs, optional
        Coordinates defining the outer boundary line. If None, uses default.
    exclusion_polygon_coords : list of [lon, lat] pairs, optional
        Coordinates defining exclusion polygon. If None, uses default.
    exclude_coordinates : list of [lon, lat] pairs, optional
        Specific coordinates to exclude. If None, uses default.
    buoy_coordinates : dict or list, optional
        Buoy coordinates to preserve (never remove). Can be:
        - dict: {buoy_id: (lon, lat), ...} (e.g., {'SSBN7': (-78.484, 33.838), ...})
        - list: [(lon, lat), ...] (list of coordinate tuples)
        Points within 0.1 degrees (~11km) of any buoy will be kept even if removed by mask.
    variable : str | list[str] | None, optional
        If specified, only process these variable(s). None processes all.
    grid : str | list[str] | None, optional
        If specified, only process files for these grid(s). 
        - None: process all grids (default)
        - str: process single grid (e.g., "grid1")
        - list: process multiple grids (e.g., ["grid1", "grid2"])
    verbose : bool, optional
        Whether to print progress messages (default: True)
    
    Returns
    -------
    None
        Cropped files are saved to output_dir with float32 and compression
    """
    from pathlib import Path
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, LineString
    from tqdm.auto import tqdm
    try:
        import cartopy.feature as cfeature
        from shapely.geometry import shape
        HAS_CARTOPY = True
    except ImportError:
        HAS_CARTOPY = False
    
    # Defaults for Cyclones_NC project
    if input_dir is None:
        input_dir = _BASE_DIR / "outputs" / "partitions_cyclones"
    if output_dir is None:
        output_dir = _BASE_DIR / "outputs" / "cropped_variables"
    if shoreline_geojson_file is None:
        shoreline_geojson_file = _BASE_DIR / "inputs" / "CoastSat_shoreline_NC_merged.geojson"

    # Convert to Path objects
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    shoreline_geojson_file = Path(shoreline_geojson_file)
    
    # Default coordinates (from the notebook)
    if outer_boundary_line_coords is None:
        outer_boundary_line_coords = [
            [-78.511,   33.3236],
            [-77.7062,  33.6054],
            [-77.4069,  33.8567],
            [-76.4318,  34.4091],
            [-75.7424,  34.8431],
            [-75.3662,  35.095],
            [-74.8125,  35.6875],
            [-74.812987,36.5790],
        ]
    
    # Additional exclusion polygon (closed polygon)
    outer_boundary_line_coords_2 = [
        [-76.1606, 36.6122],
        [-75.9285, 36.7097],
        [-75.8592, 36.4886],
        [-75.7273, 36.1212],
        [-75.8509, 36.0735],
        [-76.1606, 36.6122],
    ]
    
    if exclusion_polygon_coords is None:
        exclusion_polygon_coords = [
            [-76.407166, 34.791250],
            [-75.872955, 35.159214],
            [-75.537872, 35.247862],
            [-75.456848, 35.599925],
            [-75.644989, 35.960223],
            [-76.810913, 35.362176],
            [-76.407166, 34.791250],  # Close polygon
        ]
    
    if exclude_coordinates is None:
        exclude_coordinates = [
            [-77.3429, 34.5594],
            [-77.1875, 34.6250],
            [-76.5137, 34.6452],
            [-76.5,    34.6875],
            [-76.3750, 34.8125],
            [-76.3125, 34.8750],
            [-75.75,   35.1875],
            [-78.6875, 33.8125],
            [-78.500,  33.8750],
            [-74.8750, 35.5],
        ]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"{'='*80}")
        print(f"CROPPING CONCATENATED FILES BY SPATIAL MASK")
        print(f"{'='*80}")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}\n")
    
    # Find all NetCDF files
    nc_files = sorted(input_dir.glob("*.nc"))
    
    if len(nc_files) == 0:
        raise ValueError(f"No NetCDF files found in {input_dir}")
    
    # Filter by variable if specified
    if variable is not None:
        if isinstance(variable, str):
            variable_names = [variable]
        elif isinstance(variable, list):
            variable_names = variable
        else:
            raise ValueError(f"variable parameter must be str, list[str], or None")
        
        # Filter files by variable name (e.g., "hs_grid1.nc" contains "hs")
        filtered_files = []
        for nc_file in nc_files:
            file_var = nc_file.stem.split('_')[0]  # Extract variable from filename
            if file_var in variable_names:
                filtered_files.append(nc_file)
        nc_files = filtered_files
        
        if len(nc_files) == 0:
            raise ValueError(f"No files found for specified variable(s): {variable_names}")
    
    # Filter by grid if specified
    if grid is not None:
        if isinstance(grid, str):
            grid_names = [grid]
        elif isinstance(grid, list):
            grid_names = grid
        else:
            raise ValueError(f"grid parameter must be str, list[str], or None")
        
        # Filter files by grid name
        # Robustly supports patterns like:
        #   {var}_{grid}.nc
        #   {var}_{grid}_sea_states.nc
        #   {var}_{grid}_sea_states_masked.nc
        #   {var}_{grid}_cyclones.nc
        filtered_files = []
        for nc_file in nc_files:
            parts = nc_file.stem.split('_')
            if len(parts) >= 2:
                # Prefer any token that looks like "grid1", "grid2", etc.
                file_grid = next((p for p in parts if p.startswith("grid")), None)
                if file_grid is None:
                    # Fallback: use last token (old behaviour)
                    file_grid = parts[-1]
                if file_grid in grid_names:
                    filtered_files.append(nc_file)
        
        nc_files = filtered_files
        
        if len(nc_files) == 0:
            raise ValueError(f"No files found for specified grid(s): {grid_names}")
    
    if verbose:
        filters = []
        if variable is not None:
            filters.append(f"variable(s): {variable_names if isinstance(variable, list) else variable}")
        if grid is not None:
            filters.append(f"grid(s): {grid_names if isinstance(grid, list) else grid}")
        
        if filters:
            print(f"Processing {len(nc_files)} file(s) for {', '.join(filters)}")
        else:
            print(f"Found {len(nc_files)} file(s) to process")
        print()
    
    # Build geometry objects (once, reused for all files)
    if verbose:
        print("Loading geometry objects...")
    
    # Shoreline: union of all shoreline geometries (simple approach matching notebook)
    if not shoreline_geojson_file.exists():
        raise FileNotFoundError(f"Shoreline file not found: {shoreline_geojson_file}")
    shoreline_gdf = gpd.read_file(shoreline_geojson_file)
    # Use union_all() instead of deprecated unary_union
    if hasattr(shoreline_gdf.geometry, 'union_all'):
        shoreline_geom = shoreline_gdf.geometry.union_all()
    else:
        shoreline_geom = shoreline_gdf.geometry.unary_union
    
    # Outer boundary as a line
    outer_line = LineString(outer_boundary_line_coords)
    
    # Exclusion polygon
    excl_poly = Polygon(exclusion_polygon_coords)
    
    # Additional exclusion polygon (from outer_boundary_line_coords_2)
    excl_poly_2 = Polygon(outer_boundary_line_coords_2)
    
    # Specific coordinates to exclude (exact match on lon/lat)
    exclude_points = {(float(lo), float(la)) for lo, la in exclude_coordinates}
    
    # Land mask: create land polygons from cartopy to catch points on land
    # This is needed because shoreline might be a LineString, not a Polygon
    land_geoms = []
    if HAS_CARTOPY:
        try:
            land_feature = cfeature.NaturalEarthFeature(
                'physical', 'land', '10m',
                edgecolor='none', facecolor='none'
            )
            for geom in land_feature.geometries():
                try:
                    if hasattr(geom, '__geo_interface__'):
                        shp_geom = shape(geom.__geo_interface__)
                        land_geoms.append(shp_geom)
                except:
                    continue
            if verbose and land_geoms:
                print(f"  ✓ Land mask loaded ({len(land_geoms)} land polygons)")
        except Exception as e:
            if verbose:
                print(f"  ⚠ Could not load land mask: {e}")
    
    if verbose:
        print("  ✓ Geometry objects loaded\n")
    
    # Helper: which side of outer boundary line (seaward) - matching notebook exactly
    def is_seaward_of_outer_line(point: Point) -> bool:
        """
        Returns True if 'point' lies on the seaward side of the outer boundary line.
        Assumes the seaward side is to the "left" of the line direction defined
        by OUTER_BOUNDARY_LINE_COORDS. If you see the opposite behavior, flip the sign below.
        """
        # Project point onto line
        s = outer_line.project(point)
        pt_on_line = outer_line.interpolate(s)

        # Local tangent along the line
        ds_dist = max(1e-6 * outer_line.length, 1e-6)
        s1 = max(0.0, s - ds_dist)
        s2 = min(outer_line.length, s + ds_dist)
        p1 = outer_line.interpolate(s1)
        p2 = outer_line.interpolate(s2)
        tx, ty = (p2.x - p1.x, p2.y - p1.y)

        # Left normal
        nx, ny = (-ty, tx)

        # Vector from line to point
        vx, vy = (point.x - pt_on_line.x, point.y - pt_on_line.y)

        # > 0 means point is to the left of the line
        dot = vx * nx + vy * ny
        return dot >= 0.0
    
    # Main point-filter function (applied per site) - matching notebook exactly
    def keep_point(lon_i: float, lat_i: float) -> bool:
        p = Point(lon_i, lat_i)
        
        # 6. Latitude filter: exclude points with latitude > 36.75
        if lat_i > 36.75:
            return False
        
        # 5. Land mask check - exclude points on land (catches points missed by shoreline check)
        if land_geoms:
            for land_poly in land_geoms:
                try:
                    if land_poly.contains(p):
                        return False
                except:
                    continue
        
        # 4. NOT matching excluded coordinates
        if (float(lon_i), float(lat_i)) in exclude_points:
            return False

        # 3. NOT inside exclusion polygons
        if excl_poly.contains(p):
            return False
        
        # 3b. NOT inside additional exclusion polygon
        if excl_poly_2.contains(p):
            return False

        # 2. Seaward (outside) of the shoreline
        if shoreline_geom.contains(p):
            return False

        # 1. On the seaward side of the outer boundary line
        if not is_seaward_of_outer_line(p):
            return False

        return True
    
    # Process each file
    for nc_file in tqdm(nc_files, desc="Processing files", disable=not verbose, unit="file"):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {nc_file.name}")
            print(f"{'='*60}")
        
        try:
            # Open dataset (lazy loading for large files, chunk only time dimension)
            # Match notebook: ds = xr.open_dataset(src, chunks={"time": 1000})
            ds = xr.open_dataset(nc_file, chunks={"time": 1000})
            
            # Get coordinates - use coord_x/coord_y directly (they contain the actual coordinates)
            # lon/lat may exist but be empty or have wrong size
            lon = None
            lat = None
            needs_rename = False
            
            # Use coord_x/coord_y directly
            if "coord_x" in ds.coords or "coord_x" in ds.data_vars:
                try:
                    lon = ds["coord_x"].values
                    needs_rename = True
                except:
                    pass
            
            if "coord_y" in ds.coords or "coord_y" in ds.data_vars:
                try:
                    lat = ds["coord_y"].values
                    needs_rename = True
                except:
                    pass
            
            # If coord_x/coord_y not found, fallback to lon/lat
            if lon is None:
                if "lon" in ds.coords or "lon" in ds.data_vars:
                    try:
                        lon = ds["lon"].values
                    except:
                        pass
            
            if lat is None:
                if "lat" in ds.coords or "lat" in ds.data_vars:
                    try:
                        lat = ds["lat"].values
                    except:
                        pass
            
            if lon is None or lat is None:
                if verbose:
                    print(f"  ⚠ Could not find coord_x/coord_y or lon/lat coordinates.")
                    print(f"    Available coords: {list(ds.coords.keys())}")
                    print(f"    Available data_vars: {list(ds.data_vars.keys())}")
                ds.close()
                continue
            
            # Convert to numpy arrays and flatten
            lon = np.asarray(lon).flatten()
            lat = np.asarray(lat).flatten()
            
            if lon.size == 0 or lat.size == 0:
                if verbose:
                    print(f"  ⚠ Coordinates are empty arrays. Skipping.")
                ds.close()
                continue
            
            if len(lon) != len(lat):
                if verbose:
                    print(f"  ⚠ lon and lat have different lengths ({len(lon)} vs {len(lat)}). Skipping.")
                ds.close()
                continue
            
            # Rename coord_x/coord_y to lon/lat BEFORE processing (as requested)
            # First drop empty lon/lat if they exist, then rename coord_x/coord_y
            if needs_rename:
            
                # Drop empty lon/lat first to avoid conflicts
                vars_to_drop_before_rename = []
                if "lon" in ds.coords or "lon" in ds.data_vars:
                    vars_to_drop_before_rename.append("lon")
                if "lat" in ds.coords or "lat" in ds.data_vars:
                    vars_to_drop_before_rename.append("lat")
                
                if vars_to_drop_before_rename:
                    ds = ds.drop_vars(vars_to_drop_before_rename, errors="ignore")
                    if verbose:
                        print(f"    Dropped empty lon/lat variables")
                
                # Now rename coord_x/coord_y to lon/lat
                rename_dict = {}
                if "coord_x" in ds.coords:
                    rename_dict["coord_x"] = "lon"
                elif "coord_x" in ds.data_vars:
                    rename_dict["coord_x"] = "lon"
                if "coord_y" in ds.coords:
                    rename_dict["coord_y"] = "lat"
                elif "coord_y" in ds.data_vars:
                    rename_dict["coord_y"] = "lat"
                
                if rename_dict:
                    ds = ds.rename(rename_dict)
                    # Re-extract after renaming to ensure we have the renamed version
                    lon = ds["lon"].values
                    lat = ds["lat"].values
            
            
            # Build boolean mask over 'site' dimension
            # Use efficient np.fromiter (matching notebook) - much faster than loop
            # The loop-based approach was too slow for large datasets
            if verbose:
                print(f"    Building spatial mask for {len(lon)} sites...")
            site_mask = np.fromiter(
                (keep_point(lo, la) for lo, la in zip(lon, lat)),
                dtype=bool,
                count=len(lon)
            )
            
            # Preserve points near buoys (if provided)
            # Only preserve points that are already very close to buoys (< 1km)
            if buoy_coordinates is not None:
                # Convert buoy_coordinates to list of (lon, lat) tuples
                if isinstance(buoy_coordinates, dict):
                    buoy_coords_list = list(buoy_coordinates.values())
                elif isinstance(buoy_coordinates, list):
                    buoy_coords_list = buoy_coordinates
                else:
                    buoy_coords_list = []
                
                if len(buoy_coords_list) > 0:
                    # Check each point to see if it's very close to any buoy
                    # This is O(n_points * n_buoys) which is very fast since n_buoys is small
                    buoy_tolerance = 0.01  # degrees (~1.1km) - much stricter
                    points_preserved = 0
                    
                    for i, (point_lon, point_lat) in enumerate(zip(lon, lat)):
                        # Check if this point is very close to any buoy
                        for buoy_lon, buoy_lat in buoy_coords_list:
                            distance_deg = np.sqrt((point_lon - buoy_lon)**2 + (point_lat - buoy_lat)**2)
                            if distance_deg < buoy_tolerance:
                                # Force this point to be kept (even if mask says False)
                                if not site_mask[i]:
                                    site_mask[i] = True
                                    points_preserved += 1
                                    if verbose:
                                        distance_km = distance_deg * 111.0
                                        print(f"    Preserved point at ({point_lon:.4f}, {point_lat:.4f}) - {distance_km:.2f} km from buoy ({buoy_lon:.4f}, {buoy_lat:.4f})")
                                break  # Only need to match one buoy
                    
                    if points_preserved > 0 and verbose:
                        print(f"  Total: Preserved {points_preserved} point(s) near buoy locations (< {buoy_tolerance*111:.1f} km)")
                    elif verbose:
                        print(f"  No points found within {buoy_tolerance*111:.1f} km of buoy locations")
            
            n_kept = site_mask.sum()
            n_total = len(site_mask)
            if verbose:
                print(f"  Keeping {n_kept} of {n_total} sites ({100*n_kept/n_total:.1f}%)")
            
            if n_kept == 0:
                if verbose:
                    print(f"  ⚠ No sites kept after masking. Skipping file.")
                ds.close()
                continue
            
            # Apply mask
            subset = ds.isel(site=site_mask)
            
            # Drop unused variables/coordinates (coord_x, coord_y, weights)
            # Match notebook: subset.drop_vars(["coord_x", "coord_y", "weight_grid1", "weight_grid2"], errors="ignore")
            vars_to_drop = ["coord_x", "coord_y", "weight_grid1", "weight_grid2", "weight_grid3", "weight_grid4"]
            subset = subset.drop_vars([v for v in vars_to_drop if v in subset.variables], errors="ignore")
            
            # Remove all attributes (global + per variable/coord)
            subset.attrs = {}
            for v in subset.data_vars:
                subset[v].attrs = {}
            for c in subset.coords:
                subset[c].attrs = {}
            
            # Output file
            output_file = output_dir / nc_file.name.replace('.nc', '_masked.nc')
            
            # Save file - match notebook: subset.to_netcdf(dst)
            # But add encoding for float32 and compression
            if verbose:
                print(f"  Saving cropped file...")
            
            # Prepare encoding to ensure float32 and compression
            encoding = {}
            valid_encoding_keys = {
                "zlib", "complevel", "shuffle", "fletcher32", "contiguous",
                "chunksizes", "dtype", "_FillValue"
            }
            
            # Set encoding for all data variables (ensure float32 and compression)
            for var_name in subset.data_vars:
                var = subset[var_name]
                # Ensure float32 dtype
                if np.issubdtype(var.dtype, np.floating) and var.dtype != np.float32:
                    subset[var_name] = var.astype(np.float32)
                
                # Set encoding with compression
                var_enc = var.encoding.copy() if hasattr(var, 'encoding') and var.encoding else {}
                var_enc["dtype"] = "float32"
                var_enc["zlib"] = True
                var_enc["complevel"] = 4
                var_enc["shuffle"] = True
                
                # Clean encoding: exclude chunksizes (can cause None/int errors with netCDF4 for cyclones)
                cleaned_enc = {k: v for k, v in var_enc.items() if k in valid_encoding_keys and k != "chunksizes"}
                
                encoding[var_name] = cleaned_enc
            
            # Preserve coordinate encodings (also check chunksizes)
            for coord_name in subset.coords:
                coord = subset[coord_name]
                coord_enc = coord.encoding.copy() if hasattr(coord, 'encoding') and coord.encoding else {}
                cleaned_coord_enc = {k: v for k, v in coord_enc.items() if k in valid_encoding_keys and k != "chunksizes"}
                
                encoding.setdefault(coord_name, cleaned_coord_enc)
            
            # Save with encoding (matching notebook's simple approach but with compression)
            subset.to_netcdf(output_file, encoding=encoding)
            
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            if verbose:
                print(f"  ✓ Saved: {output_file.name} ({file_size_mb:.1f} MB)")
                print(f"    Format: float32, compressed (zlib, level 4)")
            
            # Close datasets
            ds.close()
            subset.close()
            
        except Exception as e:
            if verbose:
                print(f"  ✗ Error processing {nc_file.name}: {e}")
                import traceback
                traceback.print_exc()
            continue
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"CROPPING COMPLETE")
        print(f"{'='*80}")
        print(f"\nCropped files saved to: {output_dir}")
        filters = []
        if variable is not None:
            filters.append(f"Variables: {variable_names if isinstance(variable, list) else variable}")
        if grid is not None:
            filters.append(f"Grids: {grid_names if isinstance(grid, list) else grid}")
        if filters:
            print(f"{', '.join(filters)}")
        print(f"Files processed: {len(nc_files)}")

from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull
from collections import defaultdict
from utils.outputs_grids import compute_distance_to_border, compute_blending_weights


def detect_data_quality_issues(data, time, min_consecutive_days=5, zero_threshold=1e-6, constant_tolerance=1e-6):
    """
    Detect data quality issues in a time series.
    
    Parameters:
    -----------
    data : array-like
        Time series data values
    time : array-like
        Time stamps (pandas DatetimeIndex or similar)
    min_consecutive_days : int
        Minimum number of consecutive days to flag as issue (default: 5)
    zero_threshold : float
        Threshold below which values are considered zero (default: 1e-6)
    constant_tolerance : float
        Tolerance for detecting constant values (default: 1e-6)
        
    Returns:
    --------
    dict : Dictionary with quality flags
        - 'has_nans': bool, True if any NaNs present
        - 'has_zeros': bool, True if consecutive zeros detected
        - 'has_constant': bool, True if constant values for long period
        - 'quality_mask': array, boolean mask where True = good data
        - 'issue_periods': list of tuples, (start_idx, end_idx) for each issue period
    """
    data = np.asarray(data)
    time = pd.to_datetime(time)
    
    # Initialize quality mask (True = good data)
    quality_mask = np.ones(len(data), dtype=bool)
    issue_periods = []
    
    # Check for NaNs
    nan_mask = np.isnan(data)
    has_nans = np.any(nan_mask)
    if has_nans:
        quality_mask[nan_mask] = False
        # Find consecutive NaN periods
        nan_diff = np.diff(np.concatenate(([False], nan_mask, [False])).astype(int))
        nan_starts = np.where(nan_diff == 1)[0]
        nan_ends = np.where(nan_diff == -1)[0]
        for start, end in zip(nan_starts, nan_ends):
            issue_periods.append((start, end))
    
    # Check for consecutive zeros
    zero_mask = np.abs(data) < zero_threshold
    has_zeros = False
    
    if np.any(zero_mask):
        # Find consecutive zero periods
        zero_diff = np.diff(np.concatenate(([False], zero_mask, [False])).astype(int))
        zero_starts = np.where(zero_diff == 1)[0]
        zero_ends = np.where(zero_diff == -1)[0]
        
        for start, end in zip(zero_starts, zero_ends):
            # Calculate duration in days
            if end < len(time):
                duration = (time[end] - time[start]).total_seconds() / 86400.0
                if duration >= min_consecutive_days:
                    has_zeros = True
                    quality_mask[start:end] = False
                    issue_periods.append((start, end))
    
    # Check for constant values (fixed values for long period)
    has_constant = False
    if len(data) > min_consecutive_days:
        # Use a more robust approach: scan through data and find constant periods
        i = 0
        while i < len(data) - min_consecutive_days:
            # Skip if current value is NaN
            if np.isnan(data[i]):
                i += 1
                continue
            
            ref_value = data[i]
            start_idx = i
            
            # Find how long this value persists
            j = i + 1
            while j < len(data):
                if np.isnan(data[j]):
                    break
                if np.abs(data[j] - ref_value) >= constant_tolerance:
                    break
                j += 1
            
            # Check if this constant period is long enough
            if j - start_idx >= min_consecutive_days:
                # Calculate duration in days
                end_idx = min(j, len(data))
                if end_idx <= len(time):
                    duration = (time[min(end_idx-1, len(time)-1)] - time[start_idx]).total_seconds() / 86400.0
                    if duration >= min_consecutive_days:
                        has_constant = True
                        quality_mask[start_idx:end_idx] = False
                        issue_periods.append((start_idx, end_idx))
                        i = end_idx
                        continue
            
            i += 1
    
    return {
        'has_nans': has_nans,
        'has_zeros': has_zeros,
        'has_constant': has_constant,
        'quality_mask': quality_mask,
        'issue_periods': issue_periods
    }


def compute_quality_adjusted_weights(data1, data2, time, alpha_spatial, beta_spatial, 
                                     min_consecutive_days=5, zero_threshold=1e-6, 
                                     constant_tolerance=1e-6):
    """
    Compute blending weights adjusted for data quality.
    
    When one grid has quality issues (NaNs, zeros, constant values),
    increase the weight of the other grid.
    
    Parameters:
    -----------
    data1 : array-like
        Time series from grid 1
    data2 : array-like
        Time series from grid 2
    time : array-like
        Time stamps
    alpha_spatial : float
        Spatial blending weight for grid 1 (from polygon-based computation)
    beta_spatial : float
        Spatial blending weight for grid 2 (from polygon-based computation)
    min_consecutive_days : int
        Minimum consecutive days to flag as issue
    zero_threshold : float
        Threshold for zero detection
    constant_tolerance : float
        Tolerance for constant value detection
        
    Returns:
    --------
    alpha_adjusted : array
        Time-varying adjusted alpha weights
    beta_adjusted : array
        Time-varying adjusted beta weights
    quality_info : dict
        Quality information for both grids
    """
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    time = pd.to_datetime(time)
    
    # Detect quality issues for both grids
    quality1 = detect_data_quality_issues(data1, time, min_consecutive_days, 
                                          zero_threshold, constant_tolerance)
    quality2 = detect_data_quality_issues(data2, time, min_consecutive_days, 
                                          zero_threshold, constant_tolerance)
    
    # Initialize weights with spatial values
    alpha_adjusted = np.full(len(time), alpha_spatial)
    beta_adjusted = np.full(len(time), beta_spatial)
    
    # Adjust weights based on quality
    for i in range(len(time)):
        # Get quality status at this time step
        good1 = quality1['quality_mask'][i] if i < len(quality1['quality_mask']) else True
        good2 = quality2['quality_mask'][i] if i < len(quality2['quality_mask']) else True
        
        # If both are good, use spatial weights
        if good1 and good2:
            alpha_adjusted[i] = alpha_spatial
            beta_adjusted[i] = beta_spatial
        # If grid1 is bad, favor grid2
        elif not good1 and good2:
            alpha_adjusted[i] = 0.0
            beta_adjusted[i] = 1.0
        # If grid2 is bad, favor grid1
        elif good1 and not good2:
            alpha_adjusted[i] = 1.0
            beta_adjusted[i] = 0.0
        # If both are bad, use spatial weights (fallback)
        else:
            alpha_adjusted[i] = alpha_spatial
            beta_adjusted[i] = beta_spatial
    
    # Normalize to ensure alpha + beta = 1
    total = alpha_adjusted + beta_adjusted
    mask = total > 0
    alpha_adjusted[mask] = alpha_adjusted[mask] / total[mask]
    beta_adjusted[mask] = beta_adjusted[mask] / total[mask]
    
    quality_info = {
        'grid1': quality1,
        'grid2': quality2
    }
    
    return alpha_adjusted, beta_adjusted, quality_info





