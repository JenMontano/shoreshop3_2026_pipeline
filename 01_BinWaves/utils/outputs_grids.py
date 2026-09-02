"""
Functions for processing and blending BinWaves grid outputs.

This module provides functions for:
- Reading and filtering grid locations
- Processing grids and computing wave statistics
- Saving time series to NetCDF files with polygon information
- Blending overlapping grids using distance-based weights
"""

import json
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull
from tqdm import tqdm


def compute_distance_to_border(point, polygon):
    """
    Compute the perpendicular distance from a point to the nearest border of a polygon.
    
    The distance is computed as the shortest distance to the polygon boundary, which
    is always perpendicular to the boundary at the closest point.
    
    Parameters:
    -----------
    point : Point or tuple (lon, lat)
        Point to compute distance for
    polygon : shapely.Polygon
        Polygon to compute distance to border
        
    Returns:
    --------
    float : Perpendicular distance in degrees (multiply by 111.0 to get km)
    """
    from shapely.geometry import Point as ShapelyPoint
    
    if isinstance(point, (tuple, list, np.ndarray)):
        point = ShapelyPoint(point[0], point[1])
    elif not isinstance(point, ShapelyPoint):
        point = ShapelyPoint(point)
    
    boundary = polygon.boundary
    distance = point.distance(boundary)
    return distance


def compute_blending_weights(point, polygon1, polygon2, steepness=10.0):
    """
    Compute complementary blending weights (alpha and beta) based on point location and distance to borders.
    
    Logic:
    - If point is ONLY in grid1 (not in grid2): alpha=1, beta=0
    - If point is ONLY in grid2 (not in grid1): alpha=0, beta=1
    - If point is in BOTH grids (overlap): use distance-based weights
      - Closer to grid1 border: alpha decreases, beta increases
      - Closer to grid2 border: alpha increases, beta decreases
      - Middle of overlap: alpha=0.5, beta=0.5
    
    Blending formula: hs_mean = hs_grid1 * alpha + hs_grid2 * beta
    
    Parameters:
    -----------
    point : Point or tuple (lon, lat)
        Point to compute weights for
    polygon1 : shapely.Polygon
        Polygon for grid1
    polygon2 : shapely.Polygon
        Polygon for grid2
    steepness : float
        Steepness parameter for sigmoid function (default: 10.0)
        Higher values = sharper transition in overlap region
        
    Returns:
    --------
    tuple : (alpha, beta) - normalized blending weights that sum to 1
        - alpha: weight for grid1 (0 to 1)
        - beta: weight for grid2 (0 to 1)
        - alpha + beta = 1
    """
    from shapely.geometry import Point as ShapelyPoint
    
    if isinstance(point, (tuple, list, np.ndarray)):
        point = ShapelyPoint(point[0], point[1])
    elif not isinstance(point, ShapelyPoint):
        point = ShapelyPoint(point)
    
    # Check if point is in each polygon
    in_grid1 = polygon1.contains(point) or polygon1.touches(point)
    in_grid2 = polygon2.contains(point) or polygon2.touches(point)
    
    # Case 1: Point is ONLY in grid1 (not overlapping)
    if in_grid1 and not in_grid2:
        return 1.0, 0.0
    
    # Case 2: Point is ONLY in grid2 (not overlapping)
    if in_grid2 and not in_grid1:
        return 0.0, 1.0
    
    # Case 3: Point is in BOTH grids (overlap region)
    # Use distance-based weights
    dist1 = compute_distance_to_border(point, polygon1)
    dist2 = compute_distance_to_border(point, polygon2)
    
    # Convert distances to km for sigmoid computation
    dist1_km = dist1 * 111.0
    dist2_km = dist2 * 111.0
    
    # Compute raw weights using sigmoid
    alpha_raw = 1.0 / (1.0 + np.exp(-steepness * dist1_km))
    beta_raw = 1.0 / (1.0 + np.exp(-steepness * dist2_km))
    
    # Normalize so that alpha + beta = 1
    total_weight = alpha_raw + beta_raw
    
    if total_weight > 0:
        alpha = alpha_raw / total_weight
        beta = beta_raw / total_weight
    else:
        alpha = 0.5
        beta = 0.5
    
    return alpha, beta


def read_locations(filepath):
    """Read a .loc file and return coordinates."""
    coords = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    lon, lat = float(parts[0]), float(parts[1])
                    coords.append([lon, lat])
    return np.array(coords)


def load_grid_polygon(mesh_file, grid_name):
    """Load the polygon for a specific grid from the mesh GeoJSON file."""
    with open(mesh_file, 'r') as f:
        data = json.load(f)
    
    for feature in data['features']:
        if feature['properties']['name'] == grid_name:
            coords = feature['geometry']['coordinates'][0]
            return Polygon(coords)
    
    raise ValueError(f"Grid {grid_name} not found in {mesh_file}")


def filter_points_by_polygon(coords, polygon):
    """Filter points that are within the polygon (no buffer)."""
    valid_indices = []
    filtered_coords = []
    
    for i, coord in enumerate(coords):
        point = Point(coord[0], coord[1])
        if polygon.contains(point) or polygon.touches(point):
            valid_indices.append(i)
            filtered_coords.append(coord)
    
    return np.array(filtered_coords), valid_indices


def filter_border_points(coords, polygon, buffer_km=1.0):
    """Filter points that are within buffer_km of the polygon border."""
    buffer_deg = buffer_km / 111.0
    inner_polygon = polygon.buffer(-buffer_deg)
    
    valid_indices = []
    filtered_coords = []
    
    for i, coord in enumerate(coords):
        point = Point(coord[0], coord[1])
        if inner_polygon.contains(point):
            valid_indices.append(i)
            filtered_coords.append(coord)
    
    return np.array(filtered_coords), valid_indices


def compute_wave_statistics(reconstructed_spectra, time_start, time_end):
    """
    Compute wave statistics from reconstructed spectra.
    
    Parameters:
    -----------
    reconstructed_spectra : xarray.DataArray
        Reconstructed wave spectra
    time_start : str
        Start time for filtering
    time_end : str
        End time for filtering
    """
    # Filter to specified time period
    if 'time' in reconstructed_spectra.dims:
        spectra_filtered = reconstructed_spectra.sel(time=slice(time_start, time_end))
    else:
        spectra_filtered = reconstructed_spectra
    
    # Convert to wavespectra format
    spec_data = spectra_filtered.rename({"kps": "efth"}).squeeze().spec
    
    # Compute bulk parameters
    hs = spec_data.hs().values
    tp = spec_data.tp().values
    dir_pm = spec_data.dpm().values
    
    # Circular mean for directional data
    dir_pm_rad = np.deg2rad(dir_pm)
    dir_sin = np.nanmean(np.sin(dir_pm_rad))
    dir_cos = np.nanmean(np.cos(dir_pm_rad))
    dir_circular_mean = np.rad2deg(np.arctan2(dir_sin, dir_cos)) % 360
    
    stats = {
        'Hs_mean': float(np.nanmean(hs)),
        'Hs_min': float(np.nanmin(hs)),
        'Hs_max': float(np.nanmax(hs)),
        'Tp_mean': float(np.nanmean(tp)),
        'Tp_min': float(np.nanmin(tp)),
        'Tp_max': float(np.nanmax(tp)),
        'Dp_mean': float(dir_circular_mean),
        'Dp_min': float(np.nanmin(dir_pm)),
        'Dp_max': float(np.nanmax(dir_pm)),
    }
    
    return stats


def process_grid(
    grid_name,
    grid_id,
    global_index_counter,
    grids_path,
    mesh_file,
    time_start,
    time_end,
    filter_by_polygon=True,
    border_buffer_km=0,
    num_workers=15,
    time_chunk_size=24
):
    """
    Process a single grid: reconstruct wave spectra and compute statistics.
    
    Parameters:
    -----------
    grid_name : str
        Name of the grid
    grid_id : str
        ID of the grid (for finding WHACS spectrum file)
    global_index_counter : int
        Global counter for site IDs
    grids_path : Path
        Base path where grid directories are located
    mesh_file : Path
        Path to mesh GeoJSON file
    time_start : str
        Start time for processing (e.g., "2020-01-01")
    time_end : str
        End time for processing (e.g., "2020-07-01")
    filter_by_polygon : bool
        If True, filter locations by grid polygon
    border_buffer_km : float
        Buffer distance in km to exclude border points
    num_workers : int
        Number of workers for parallel processing
    time_chunk_size : int
        Size of time chunks for processing (default: 24 hours)
        
    Returns:
    --------
    tuple: (features, grid_data, updated global_index_counter)
        features: list of GeoJSON features
        grid_data: dict with processed data
        global_index_counter: updated counter
    """
    from utils.operations import transform_Offshore_spectrum
    from bluemath_tk.waves.binwaves import reconstruc_spectra
    
    print(f"\n{'='*80}")
    print(f"Processing {grid_name}")
    print(f"{'='*80}")
    
    grid_path = grids_path / grid_name
    locations_file = grid_path / "CASES" / "0000" / "locations.loc"
    # Check for averaged kp coefficients first, fall back to regular if not found
    kp_file = grid_path / "outputs" / "kp_coefficients_averaged.nc"
    cases_file = grid_path / "CASES" / "swan_cases_averaged.csv"
    
    # Find WHACS spectrum file
    inputs_dir = grid_path / "inputs"
    whacs_spectrum_files = list(inputs_dir.glob(f"{grid_id}_*"))
    
    if not whacs_spectrum_files:
        print(f"  ✗ WHACS spectrum not found matching pattern: {inputs_dir / f'{grid_id}_*'}")
        return [], {}, global_index_counter
    
    whacs_spectrum_file = whacs_spectrum_files[0]
    if len(whacs_spectrum_files) > 1:
        print(f"  ⚠ Multiple files found matching pattern, using: {whacs_spectrum_file.name}")
    
    if not locations_file.exists():
        print(f"  ✗ Locations file not found: {locations_file}")
        return [], {}, global_index_counter
    if not kp_file.exists():
        print(f"  ✗ Kp coefficients not found: {kp_file}")
        return [], {}, global_index_counter
    
    # Load locations
    print(f"  Loading locations...")
    all_coords = read_locations(locations_file)
    print(f"    Total locations: {len(all_coords)}")
    
    # Always load grid polygon (needed for blending, even if not filtering)
    grid_polygon = None
    try:
        grid_polygon = load_grid_polygon(mesh_file, grid_name)
        print(f"  ✓ Loaded grid polygon for {grid_name}")
    except Exception as e:
        print(f"  ⚠ Could not load polygon: {e}")
        # Create convex hull from actual point coordinates to follow the dot pattern
        points_unique = np.unique(all_coords, axis=0)
        if len(points_unique) >= 3:
            hull = ConvexHull(points_unique)
            hull_vertices = points_unique[hull.vertices]
            polygon_coords = [[float(v[0]), float(v[1])] for v in hull_vertices]
            polygon_coords.append(polygon_coords[0])  # Close the polygon
            grid_polygon = Polygon(polygon_coords)
            print(f"    ✓ Created convex hull polygon from coordinates ({len(hull_vertices)} vertices)")
        else:
            # Fallback to bounding box if not enough points
            print(f"    Creating bounding box from coordinates...")
            min_lon, min_lat = all_coords.min(axis=0)
            max_lon, max_lat = all_coords.max(axis=0)
            buffer = 0.01
            grid_polygon = Polygon([
                [min_lon - buffer, min_lat - buffer],
                [max_lon + buffer, min_lat - buffer],
                [max_lon + buffer, max_lat + buffer],
                [min_lon - buffer, max_lat + buffer],
                [min_lon - buffer, min_lat - buffer]
            ])
            print(f"    ⚠ Created bounding box (insufficient points for convex hull)")
    
    # Conditionally filter by polygon or use all locations
    if filter_by_polygon:
        print(f"  Filtering locations by grid polygon...")
        filtered_coords_by_polygon, polygon_valid_indices = filter_points_by_polygon(all_coords, grid_polygon)
        print(f"    Locations within {grid_name} polygon: {len(filtered_coords_by_polygon)}/{len(all_coords)} ({len(filtered_coords_by_polygon)/len(all_coords)*100:.1f}%)")
        
        if len(filtered_coords_by_polygon) == 0:
            print(f"  ✗ No locations found within {grid_name} polygon")
            return [], {}, global_index_counter
    else:
        print(f"  Using all locations without polygon filtering...")
        filtered_coords_by_polygon = all_coords
        polygon_valid_indices = list(range(len(all_coords)))
        print(f"    Using all {len(filtered_coords_by_polygon)} locations")
    
    # Load data files
    print(f"  Loading WHACS spectrum and Kp coefficients...")
    cawcr_spectrum = xr.open_dataset(whacs_spectrum_file)
    
    # Rename '__xarray_dataarray_variable__' to 'efth' if it exists, otherwise check if 'efth' already exists
    if '__xarray_dataarray_variable__' in cawcr_spectrum.data_vars:
        cawcr_spectrum = cawcr_spectrum.rename({'__xarray_dataarray_variable__': 'efth'})
    elif 'efth' not in cawcr_spectrum.data_vars:
        # If neither exists, check what variables are available
        available_vars = list(cawcr_spectrum.data_vars.keys())
        if len(available_vars) == 1:
            # If there's only one data variable, rename it to 'efth'
            cawcr_spectrum = cawcr_spectrum.rename({available_vars[0]: 'efth'})
        else:
            raise ValueError(f"Could not find 'efth' or '__xarray_dataarray_variable__' in spectrum file. Available variables: {available_vars}")
    
    kp_coeffs = xr.open_dataset(kp_file)
    
    # Match coordinates
    if 'coord_x' in kp_coeffs.coords and 'coord_y' in kp_coeffs.coords:
        kp_lon = kp_coeffs.coord_x.values
        kp_lat = kp_coeffs.coord_y.values
    elif 'lon' in kp_coeffs.coords and 'lat' in kp_coeffs.coords:
        if kp_coeffs.lon.dims == ('site',):
            kp_lon = kp_coeffs.lon.values
            kp_lat = kp_coeffs.lat.values
        else:
            raise ValueError("lon/lat in kp_coefficients.nc are not per-site")
    else:
        raise ValueError("Could not find coordinates in kp_coefficients.nc")
    
    # Match coordinates with tolerance
    tolerance = 1e-5
    matched_indices = []
    matched_coords = []
    matched_original_indices = []
    
    for filtered_idx, (loc_lon, loc_lat) in enumerate(filtered_coords_by_polygon):
        distances = np.sqrt((kp_lon - loc_lon)**2 + (kp_lat - loc_lat)**2)
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]
        
        if min_dist < tolerance:
            matched_indices.append(min_dist_idx)
            matched_coords.append([loc_lon, loc_lat])
            matched_original_indices.append(polygon_valid_indices[filtered_idx])
    
    if len(matched_indices) == 0:
        filter_desc = "filtered " if filter_by_polygon else ""
        print(f"  ✗ No matching coordinates found between {filter_desc}locations.loc and kp_coefficients_averaged.nc")
        return [], {}, global_index_counter
    
    filter_desc = "filtered " if filter_by_polygon else ""
    print(f"    Matched {len(matched_indices)}/{len(filtered_coords_by_polygon)} {filter_desc}locations ({len(matched_indices)/len(filtered_coords_by_polygon)*100:.1f}%)")
    
    matched_indices = np.array(matched_indices)
    kp_coeffs_matched = kp_coeffs.isel(site=matched_indices)
    matched_coords = np.array(matched_coords)
    matched_original_indices = np.array(matched_original_indices)
    
    # Apply border buffer if needed
    if border_buffer_km > 0 and filter_by_polygon:
        print(f"  Filtering border points ({border_buffer_km} km buffer)...")
        filtered_coords, valid_indices = filter_border_points(
            matched_coords, grid_polygon, border_buffer_km
        )
        matched_indices = matched_indices[valid_indices]
        matched_original_indices = matched_original_indices[valid_indices]
    else:
        filtered_coords = matched_coords
        valid_indices = np.arange(len(matched_coords))
    
    if len(filtered_coords) == 0:
        print(f"  ✗ No valid points after filtering")
        return [], {}, global_index_counter

    kp_coeffs_filtered = kp_coeffs_matched.isel(site=valid_indices)
    original_valid_indices = matched_original_indices
    
    # Save kp_coeffs_filtered to netCDF
    kp_output_dir = Path(grids_path) / grid_name / "outputs" 
    kp_output_dir.mkdir(parents=True, exist_ok=True)
    kp_output_file = kp_output_dir / f"kp_coeffs_filtered_{grid_name}.nc"
    kp_coeffs_filtered.to_netcdf(kp_output_file)
    print(f"  ✓ Saved kp_coeffs_filtered to: {kp_output_file}")
    
    # Transform and reconstruct spectra
    print(f"  Loading model parameters...")
    # Use regular model parameters
    model_parameters = pd.read_csv(cases_file).to_dict(orient="list")
    
    print(f"  Transforming offshore spectrum...")
    offshore_spectra, offshore_spectra_case = transform_Offshore_spectrum(
        CAWCR_spectrum=cawcr_spectrum.sel(time=slice(time_start, time_end)),
        subset_parameters=model_parameters,
        available_case_num=kp_coeffs_filtered.case_num.values,
        fixed_direction=True
    )
    
    # Save offshore_spectra_case to netCDF
    spectra_output_dir = Path(grids_path) / grid_name / "outputs" 
    spectra_output_dir.mkdir(parents=True, exist_ok=True)
    spectra_output_file = spectra_output_dir / f"offshore_spectra_case_{grid_name}.nc"
    offshore_spectra_case.to_netcdf(spectra_output_file)
    print(f"  ✓ Saved offshore_spectra_case to: {spectra_output_file}")
    
    print(f"  Selecting time period: {time_start} to {time_end}...")
    offshore_spectra_time = offshore_spectra_case.sel(time=slice(time_start, time_end))
    
    print(f"  Reconstructing wave spectra for {len(filtered_coords)} sites...")
    reconstructed_onshore_spectra = reconstruc_spectra(
        offshore_spectra=offshore_spectra_time,
        kp_coeffs=kp_coeffs_filtered,
        chunk_sizes={"time": time_chunk_size},
        num_workers=num_workers,
    )
    
    # # Extract bulk parameters
    # print(f"  Extracting bulk parameters...")
    # spec_data = reconstructed_onshore_spectra.rename({"kps": "efth"}).spec
    # hs_array = spec_data.hs().values
    # tp_array = spec_data.tp().values
    # dir_array = spec_data.dpm().values
    # times = reconstructed_onshore_spectra.time.values
    
    # Extract polygon coordinates for saving (needed for blending)
    polygon_coords = None
    if grid_polygon is not None:
        polygon_coords = np.array(grid_polygon.exterior.coords)
    
    # Prepare grid data
    grid_data = {
        'hs': hs_array,
        'tp': tp_array,
        'dir': dir_array,
        'lon': filtered_coords[:, 0],
        'lat': filtered_coords[:, 1],
        'times': times,
        'grid': grid_name,
        'dataset_indices': list(range(len(filtered_coords))),
        'original_indices': original_valid_indices.tolist(),
        'polygon_coords': polygon_coords,
    }
    
    # Compute statistics for GeoJSON
    print(f"  Computing wave statistics...")
    features = []
    
    for site_idx, coord in enumerate(tqdm(filtered_coords, desc=f"  {grid_name} sites")):
        try:
            site_spectra = reconstructed_onshore_spectra.isel(site=site_idx)
            stats = compute_wave_statistics(site_spectra, time_start=time_start, time_end=time_end)
            
            feature = {
                "type": "Feature",
                "properties": {
                    "id": f"BinWaves_{global_index_counter:07d}",
                    "grid": grid_name,
                    "dataset_index": int(site_idx),
                    "original_index": int(original_valid_indices[site_idx]),
                    **stats
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(coord[0]), float(coord[1])]
                }
            }
            features.append(feature)
            global_index_counter += 1
            
        except Exception as e:
            print(f"    ✗ Error processing site {site_idx}: {e}")
            continue
    
    print(f"  ✓ Successfully processed {len(features)} sites")
    
    return features, grid_data, global_index_counter


def save_timeseries_netcdf_with_polygon(all_grid_data, output_file):
    """
    Save each grid's time series to separate NetCDF files, one per variable.
    Files are named: {variable}_{TIME_START}_{TIME_END}_{grid}.nc
    This version also saves polygon coordinates for blending.
    
    Parameters:
    -----------
    all_grid_data : list of dict
        Each dict contains: {
            'hs': array,
            'tp': array, 
            'dir': array,
            'lon': array,
            'lat': array,
            'times': array,
            'grid': str,
            'polygon_coords': array (optional) - polygon coordinates for blending
        }
    output_file : Path
        Base output directory path (files will be saved here)
    """
    from pandas import Timestamp
    
    print(f"\n{'='*80}")
    print(f"Saving NetCDF files with polygon info")
    print(f"{'='*80}")
    
    output_dir = output_file.parent if output_file.suffix else output_file
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for grid_data in all_grid_data:
        grid_name = grid_data['grid']
        times = grid_data['times']
        n_sites = len(grid_data['lon'])
        polygon_coords = grid_data.get('polygon_coords', None)
        
        time_start = Timestamp(times[0]).strftime('%Y%m%d')
        time_end = Timestamp(times[-1]).strftime('%Y%m%d')
        
        print(f"\n  Processing {grid_name}: {n_sites} sites, {len(times)} timesteps")
        
        site_ids = [f"BinWaves_{i:07d}" for i in range(n_sites)]
        
        variables_config = {
            'hs': {'data': grid_data['hs'], 'long_name': 'Significant wave height', 'units': 'm'},
            'tp': {'data': grid_data['tp'], 'long_name': 'Peak wave period', 'units': 's'},
            'dir': {'data': grid_data['dir'], 'long_name': 'Peak wave direction', 'units': 'degrees'},
        }
        
        for var_name, var_config in variables_config.items():
            filename = f"{var_name}_{time_start}_{time_end}_{grid_name}.nc"
            filepath = output_dir / filename
            
            data_vars = {
                var_name: (['time', 'site'], var_config['data'], {
                    'long_name': var_config['long_name'],
                    'units': var_config['units']
                }),
                'lon': (['site'], grid_data['lon']),
                'lat': (['site'], grid_data['lat']),
                'site_id': (['site'], site_ids),
            }
            
            # Add polygon coordinates if available
            if polygon_coords is not None:
                data_vars['polygon_lon'] = (['polygon_point'], polygon_coords[:, 0])
                data_vars['polygon_lat'] = (['polygon_point'], polygon_coords[:, 1])
            
            coords = {
                'time': times,
                'site': np.arange(n_sites),
            }
            if polygon_coords is not None:
                coords['polygon_point'] = np.arange(len(polygon_coords))
            
            ds = xr.Dataset(
                data_vars,
                coords=coords,
                attrs={
                    'grid': grid_name,
                    'n_sites': n_sites,
                    'n_timesteps': len(times),
                    'time_start': str(times[0]),
                    'time_end': str(times[-1]),
                }
            )
            
            encoding = {
                var_name: {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
                'lon': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
                'lat': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
            }
            if polygon_coords is not None:
                encoding['polygon_lon'] = {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
                encoding['polygon_lat'] = {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
            
            ds.to_netcdf(filepath, encoding=encoding)
            file_size = filepath.stat().st_size / (1024**2)
            print(f"    ✓ Saved {var_name}: {filename} ({file_size:.1f} MB)")
            saved_files.append(filepath)
    
    print(f"\n{'='*80}")
    print(f"✓ Saved {len(saved_files)} NetCDF files")
    print(f"{'='*80}")
    
    return saved_files


def blend_overlapping_grids(grid1_file, grid2_file, var_name='hs', steepness=10.0, output_file=None):
    """
    Blend two overlapping grids based on distance to their borders.
    
    Parameters:
    -----------
    grid1_file : str or Path
        Path to first grid NetCDF file
    grid2_file : str or Path
        Path to second grid NetCDF file
    var_name : str
        Variable name (default: 'hs')
    steepness : float
        Steepness parameter for sigmoid blending (default: 10.0)
    output_file : str or Path, optional
        Output file path
        
    Returns:
    --------
    xarray.Dataset : Blended dataset
    """
    from shapely.geometry import Point, Polygon
    from pathlib import Path
    
    print(f"\n{'='*80}")
    print(f"Blending overlapping grids")
    print(f"{'='*80}")
    print(f"Grid 1: {Path(grid1_file).name}")
    print(f"Grid 2: {Path(grid2_file).name}")
    print(f"Variable: {var_name}, Steepness: {steepness}")
    
    # Load datasets
    ds1 = xr.open_dataset(grid1_file)
    ds2 = xr.open_dataset(grid2_file)
    
    grid1_name = ds1.attrs.get('grid', 'grid1')
    grid2_name = ds2.attrs.get('grid', 'grid2')
    
    print(f"\n  Grid 1 ({grid1_name}): {ds1.dims['site']} sites, {ds1.dims['time']} timesteps")
    print(f"  Grid 2 ({grid2_name}): {ds2.dims['site']} sites, {ds2.dims['time']} timesteps")
    
    # Reconstruct polygons
    if 'polygon_lon' in ds1.data_vars and 'polygon_lat' in ds1.data_vars:
        poly1_coords = list(zip(ds1.polygon_lon.values, ds1.polygon_lat.values))
        polygon1 = Polygon(poly1_coords)
        print(f"  ✓ Loaded polygon for {grid1_name}")
    else:
        # Create convex hull from actual point coordinates to follow the dot pattern
        if 'coord_x' in ds1.coords:
            lon1 = ds1.coord_x.values
            lat1 = ds1.coord_y.values
        else:
            lon1 = ds1.lon.values
            lat1 = ds1.lat.values
        
        points1 = np.column_stack((lon1, lat1))
        points1_unique = np.unique(points1, axis=0)
        if len(points1_unique) >= 3:
            hull1 = ConvexHull(points1_unique)
            hull_vertices1 = points1_unique[hull1.vertices]
            polygon1_coords = [[float(v[0]), float(v[1])] for v in hull_vertices1]
            polygon1_coords.append(polygon1_coords[0])  # Close the polygon
            polygon1 = Polygon(polygon1_coords)
            print(f"  ✓ Created convex hull polygon for {grid1_name} ({len(hull_vertices1)} vertices)")
        else:
            # Fallback to bounding box
            lon1_min, lon1_max = float(np.min(lon1)), float(np.max(lon1))
            lat1_min, lat1_max = float(np.min(lat1)), float(np.max(lat1))
            buffer = 0.01
            polygon1 = Polygon([
                [lon1_min - buffer, lat1_min - buffer],
                [lon1_max + buffer, lat1_min - buffer],
                [lon1_max + buffer, lat1_max + buffer],
                [lon1_min - buffer, lat1_max + buffer],
                [lon1_min - buffer, lat1_min - buffer]
            ])
            print(f"  ⚠ Created bounding box for {grid1_name} (insufficient points for convex hull)")
    
    if 'polygon_lon' in ds2.data_vars and 'polygon_lat' in ds2.data_vars:
        poly2_coords = list(zip(ds2.polygon_lon.values, ds2.polygon_lat.values))
        polygon2 = Polygon(poly2_coords)
        print(f"  ✓ Loaded polygon for {grid2_name}")
    else:
        # Create convex hull from actual point coordinates to follow the dot pattern
        if 'coord_x' in ds2.coords:
            lon2 = ds2.coord_x.values
            lat2 = ds2.coord_y.values
        else:
            lon2 = ds2.lon.values
            lat2 = ds2.lat.values
        
        points2 = np.column_stack((lon2, lat2))
        points2_unique = np.unique(points2, axis=0)
        if len(points2_unique) >= 3:
            hull2 = ConvexHull(points2_unique)
            hull_vertices2 = points2_unique[hull2.vertices]
            polygon2_coords = [[float(v[0]), float(v[1])] for v in hull_vertices2]
            polygon2_coords.append(polygon2_coords[0])  # Close the polygon
            polygon2 = Polygon(polygon2_coords)
            print(f"  ✓ Created convex hull polygon for {grid2_name} ({len(hull_vertices2)} vertices)")
        else:
            # Fallback to bounding box
            lon2_min, lon2_max = float(np.min(lon2)), float(np.max(lon2))
            lat2_min, lat2_max = float(np.min(lat2)), float(np.max(lat2))
            buffer = 0.01
            polygon2 = Polygon([
                [lon2_min - buffer, lat2_min - buffer],
                [lon2_max + buffer, lat2_min - buffer],
                [lon2_max + buffer, lat2_max + buffer],
                [lon2_min - buffer, lat2_max + buffer],
                [lon2_min - buffer, lat2_min - buffer]
            ])
            print(f"  ⚠ Created bounding box for {grid2_name} (insufficient points for convex hull)")
    
    # Find overlapping region
    overlap = polygon1.intersection(polygon2)
    if overlap.is_empty:
        print(f"\n  ⚠ No overlap found between grids!")
        return None
    
    # Get coordinates
    coords1 = np.array([ds1.lon.values, ds1.lat.values]).T
    coords2 = np.array([ds2.lon.values, ds2.lat.values]).T
    
    # Find points in overlap
    overlap_indices1 = []
    overlap_indices2 = []
    
    for i, (lon, lat) in enumerate(coords1):
        point = Point(lon, lat)
        if overlap.contains(point) or overlap.touches(point):
            overlap_indices1.append(i)
    
    for i, (lon, lat) in enumerate(coords2):
        point = Point(lon, lat)
        if overlap.contains(point) or overlap.touches(point):
            overlap_indices2.append(i)
    
    print(f"\n  Grid 1 points in overlap: {len(overlap_indices1)}")
    print(f"  Grid 2 points in overlap: {len(overlap_indices2)}")
    
    if len(overlap_indices1) == 0 or len(overlap_indices2) == 0:
        print(f"  ⚠ Not enough overlapping points!")
        return None
    
    # Match points between grids
    tolerance_deg = 0.001
    matched_pairs = []
    
    for idx1 in overlap_indices1:
        lon1, lat1 = coords1[idx1]
        distances = np.sqrt((coords2[:, 0] - lon1)**2 + (coords2[:, 1] - lat1)**2)
        min_idx2 = np.argmin(distances)
        min_dist = distances[min_idx2]
        
        if min_dist < tolerance_deg and min_idx2 in overlap_indices2:
            matched_pairs.append((idx1, min_idx2, [lon1, lat1]))
    
    print(f"  Matched {len(matched_pairs)} point pairs")
    
    if len(matched_pairs) == 0:
        print(f"  ⚠ No matching points found!")
        return None
    
    # Compute blending weights and blend data
    print(f"\n  Computing blending weights...")
    blended_data = []
    blended_coords = []
    blended_weights1 = []
    blended_weights2 = []
    
    for idx1, idx2, coord in matched_pairs:
        point = Point(coord[0], coord[1])
        alpha, beta = compute_blending_weights(point, polygon1, polygon2, steepness=steepness)
        
        data1 = ds1[var_name].isel(site=idx1).values
        data2 = ds2[var_name].isel(site=idx2).values
        blended = alpha * data1 + beta * data2
        
        blended_data.append(blended)
        blended_coords.append(coord)
        blended_weights1.append(alpha)
        blended_weights2.append(beta)
    
    blended_data = np.array(blended_data)
    blended_coords = np.array(blended_coords)
    
    print(f"  Blended {len(blended_data)} points")
    print(f"  Mean weight grid1: {np.mean(blended_weights1):.3f}, grid2: {np.mean(blended_weights2):.3f}")
    
    # Create blended dataset
    times = ds1.time.values
    site_ids = [f"Blended_{i:07d}" for i in range(len(blended_data))]
    
    ds_blended = xr.Dataset(
        {
            var_name: (['time', 'site'], blended_data.T, {
                'long_name': ds1[var_name].attrs.get('long_name', var_name),
                'units': ds1[var_name].attrs.get('units', ''),
            }),
            'lon': (['site'], blended_coords[:, 0]),
            'lat': (['site'], blended_coords[:, 1]),
            'site_id': (['site'], site_ids),
            'blend_weight_grid1': (['site'], blended_weights1),
            'blend_weight_grid2': (['site'], blended_weights2),
        },
        coords={'time': times, 'site': np.arange(len(blended_data))},
        attrs={
            'grid1': grid1_name,
            'grid2': grid2_name,
            'variable': var_name,
            'steepness': steepness,
        }
    )
    
    # Save if output file specified
    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        encoding = {
            var_name: {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
            'lon': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
            'lat': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
            'blend_weight_grid1': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
            'blend_weight_grid2': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
        }
        
        ds_blended.to_netcdf(output_file, encoding=encoding)
        file_size = output_file.stat().st_size / (1024**2)
        print(f"\n✓ Saved: {output_file.name} ({file_size:.1f} MB)")
    
    print(f"{'='*80}\n")
    return ds_blended


def _check_data_quality_simple(data, time, min_consecutive_days=5, zero_threshold=1e-6, constant_tolerance=1e-6):
    """
    Simple quality check for time series data.
    Returns fraction of good data (0.0 to 1.0) and boolean indicating if data has major issues.
    
    This is a simplified version optimized for speed in merging operations.
    """
    data = np.asarray(data)
    time = pd.to_datetime(time)
    
    if len(data) == 0:
        return 0.0, True
    
    # Check for NaNs
    nan_mask = np.isnan(data)
    n_nans = np.sum(nan_mask)
    
    # Check for consecutive zeros (simplified - check for long zero periods)
    zero_mask = np.abs(data) < zero_threshold
    n_zeros = np.sum(zero_mask)
    
    # Check for consecutive zeros of sufficient duration
    has_long_zeros = False
    if np.any(zero_mask):
        zero_diff = np.diff(np.concatenate(([False], zero_mask, [False])).astype(int))
        zero_starts = np.where(zero_diff == 1)[0]
        zero_ends = np.where(zero_diff == -1)[0]
        for start, end in zip(zero_starts, zero_ends):
            if end < len(time):
                duration = (time[end] - time[start]).total_seconds() / 86400.0
                if duration >= min_consecutive_days:
                    has_long_zeros = True
                    break
    
    # Check for constant values (simplified - check if std is very small over long periods)
    has_constant = False
    if len(data) > min_consecutive_days:
        valid_data = data[~nan_mask]
        if len(valid_data) > min_consecutive_days:
            # Check for constant periods
            i = 0
            while i < len(data) - min_consecutive_days:
                if np.isnan(data[i]):
                    i += 1
                    continue
                ref_value = data[i]
                start_idx = i
                j = i + 1
                while j < len(data):
                    if np.isnan(data[j]) or np.abs(data[j] - ref_value) >= constant_tolerance:
                        break
                    j += 1
                if j - start_idx >= min_consecutive_days:
                    duration = (time[min(j-1, len(time)-1)] - time[start_idx]).total_seconds() / 86400.0
                    if duration >= min_consecutive_days:
                        has_constant = True
                        break
                i += 1
    
    # Calculate fraction of good data
    n_total = len(data)
    n_bad = n_nans
    good_fraction = 1.0 - (n_bad / n_total) if n_total > 0 else 0.0
    
    # Major issue if: >50% bad data, or long zeros, or constant values detected
    has_major_issue = (good_fraction < 0.5) or has_long_zeros or has_constant
    
    return good_fraction, has_major_issue


def merge_multiple_grids(grid_files, var_name='hs', steepness=10.0, tolerance_deg=0.001, output_file=None,
                         use_quality_checks=True, min_consecutive_days=5, zero_threshold=1e-6, constant_tolerance=1e-6):
    """
    Merge multiple grids by blending overlapping regions and combining non-overlapping regions.
    
    For each point location:
    - If point is in multiple grids: blend using distance-based weights (adjusted by quality if enabled)
    - If point is in only one grid: use that grid's value
    - Assign new sequential IDs to all merged points
    
    Quality checks: When enabled, if one grid has NaNs, consecutive zeros, or fixed values,
    its weight is reduced and other grids' weights are increased accordingly.
    
    Parameters:
    -----------
    grid_files : list of str or Path
        List of NetCDF file paths, one per grid (e.g., [grid1_file, grid2_file, grid3_file, grid4_file])
    var_name : str
        Variable name (default: 'hs')
    steepness : float
        Steepness parameter for sigmoid blending (default: 10.0)
    tolerance_deg : float
        Tolerance for matching points between grids in degrees (default: 0.001)
    output_file : str or Path, optional
        Output file path
    use_quality_checks : bool
        If True, adjust blending weights based on data quality (default: True)
    min_consecutive_days : int
        Minimum consecutive days to flag as quality issue (default: 5)
    zero_threshold : float
        Threshold below which values are considered zero (default: 1e-6)
    constant_tolerance : float
        Tolerance for detecting constant values (default: 1e-6)
        
    Returns:
    --------
    xarray.Dataset : Merged dataset with all grids combined
    """
    from shapely.geometry import Point, Polygon
    from collections import defaultdict
    
    print(f"\n{'='*80}")
    print(f"Merging {len(grid_files)} grids for variable: {var_name}")
    if use_quality_checks:
        print(f"Quality checks: ENABLED (will favor grids without NaNs, zeros, or constant values)")
        print(f"  Parameters: min_consecutive_days={min_consecutive_days}, zero_threshold={zero_threshold}, constant_tolerance={constant_tolerance}")
    else:
        print(f"Quality checks: DISABLED")
    print(f"{'='*80}")
    
    # Load all datasets
    datasets = []
    polygons = []
    grid_names = []
    grid_coords = []
    
    def _extract_grid_token(file_path):
        """Extract grid token from file names like var_grid1_points500m.nc."""
        stem = Path(file_path).stem.replace('_points500m', '')
        if '_' in stem:
            return stem.rsplit('_', 1)[-1]
        return None
    
    def _get_lon_lat(ds, grid_file):
        """Return lon/lat arrays; fallback to sibling file if missing."""
        def _read_pair(x_names, y_names):
            for x_name, y_name in zip(x_names, y_names):
                has_x = x_name in ds.coords or x_name in ds.data_vars
                has_y = y_name in ds.coords or y_name in ds.data_vars
                if has_x and has_y:
                    return ds[x_name].values, ds[y_name].values
            return None

        # Prefer project-native names, then common lon/lat aliases.
        coord_pair = _read_pair(
            ['coord_x', 'lon', 'longitude', 'x'],
            ['coord_y', 'lat', 'latitude', 'y']
        )
        if coord_pair is not None:
            return coord_pair
        
        grid_token = _extract_grid_token(grid_file)
        if grid_token is not None:
            search_pattern = f"*_{grid_token}_points500m.nc"
            for candidate in sorted(Path(grid_file).parent.glob(search_pattern)):
                if candidate == Path(grid_file):
                    continue
                with xr.open_dataset(candidate) as ds_ref:
                    if ('coord_x' in ds_ref.coords or 'coord_x' in ds_ref.data_vars) and \
                       ('coord_y' in ds_ref.coords or 'coord_y' in ds_ref.data_vars):
                        print(f"  ⚠ Missing coords in {Path(grid_file).name}; using {candidate.name} (coord_x/coord_y)")
                        return ds_ref.coord_x.values, ds_ref.coord_y.values
                    if ('lon' in ds_ref.coords or 'lon' in ds_ref.data_vars) and \
                       ('lat' in ds_ref.coords or 'lat' in ds_ref.data_vars):
                        print(f"  ⚠ Missing coords in {Path(grid_file).name}; using {candidate.name} (lon/lat)")
                        return ds_ref.lon.values, ds_ref.lat.values
        
        raise ValueError(
            f"Could not find coordinate variables (lon/lat or coord_x/coord_y) in {Path(grid_file).name}"
        )
    
    for i, grid_file in enumerate(grid_files):
        grid_file = Path(grid_file)
        print(f"\n[{i+1}/{len(grid_files)}] Loading {grid_file.name}...")
        # Use chunking to enable lazy loading and reduce memory usage
        # Chunk by time dimension to allow processing without loading all data
        ds = xr.open_dataset(grid_file, chunks={'time': 10000})
        datasets.append(ds)
        
        grid_name = ds.attrs.get('grid', f'grid{i+1}')
        grid_names.append(grid_name)
        lon, lat = _get_lon_lat(ds, grid_file)
        
        # Get or create polygon
        if 'polygon_lon' in ds.data_vars and 'polygon_lat' in ds.data_vars:
            poly_coords = list(zip(ds.polygon_lon.values, ds.polygon_lat.values))
            polygon = Polygon(poly_coords)
            print(f"  ✓ Loaded polygon for {grid_name}")
        else:
            # Create convex hull from actual point coordinates to follow the dot pattern
            points = np.column_stack((lon, lat))
            # Remove any duplicate points
            points_unique = np.unique(points, axis=0)
            # Need at least 3 points for a convex hull
            if len(points_unique) >= 3:
                hull = ConvexHull(points_unique)
                # Get the hull vertices in order (already counterclockwise)
                hull_vertices = points_unique[hull.vertices]
                # Convert to list of [lon, lat] pairs and close the polygon
                polygon_coords = [[float(v[0]), float(v[1])] for v in hull_vertices]
                polygon_coords.append(polygon_coords[0])  # Close the polygon
                polygon = Polygon(polygon_coords)
                print(f"  ✓ Created convex hull polygon for {grid_name} ({len(hull_vertices)} vertices)")
            else:
                # Fallback to bounding box if not enough points
                lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
                lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
                buffer = 0.01
                polygon = Polygon([
                    [lon_min - buffer, lat_min - buffer],
                    [lon_max + buffer, lat_min - buffer],
                    [lon_max + buffer, lat_max + buffer],
                    [lon_min - buffer, lat_max + buffer],
                    [lon_min - buffer, lat_min - buffer]
                ])
                print(f"  ⚠ Created bounding box for {grid_name} (insufficient points for convex hull)")
        
        polygons.append(polygon)
        grid_coords.append((lon, lat))
        print(f"  Sites: {ds.dims['site']}, Timesteps: {ds.dims['time']}")
    
    # Get all unique coordinates from all grids
    print(f"\nCollecting all point locations...")
    all_coords = []
    coord_to_grids = defaultdict(list)  # Maps (lon, lat) -> list of (grid_idx, site_idx)
    
    for grid_idx, ds in enumerate(datasets):
        lons, lats = grid_coords[grid_idx]
        
        for site_idx in range(len(lons)):
            lon, lat = float(lons[site_idx]), float(lats[site_idx])
            coord_key = (round(lon, 6), round(lat, 6))  # Round to avoid floating point issues
            
            # Check if this coordinate already exists (within tolerance)
            found = False
            for existing_coord in coord_to_grids.keys():
                dist = np.sqrt((existing_coord[0] - lon)**2 + (existing_coord[1] - lat)**2)
                if dist < tolerance_deg:
                    coord_to_grids[existing_coord].append((grid_idx, site_idx))
                    found = True
                    break
            
            if not found:
                coord_to_grids[coord_key].append((grid_idx, site_idx))
                all_coords.append(coord_key)
    
    print(f"  Found {len(all_coords)} unique point locations")
    
    # Process each unique coordinate
    # NOTE: This is often the slowest step, so we make sure the progress bar
    # clearly shows which variable is currently being processed.
    print(f"\nProcessing points and blending overlapping regions for variable: {var_name}...")
    merged_data = []
    merged_coords = []
    merged_weights = []  # Store weights for each grid
    
    # Load times once for quality checks (if needed)
    times = None
    if use_quality_checks and len(datasets) > 0:
        time_array = datasets[0].time
        if hasattr(time_array.data, 'compute'):
            times = time_array.compute().values
        else:
            times = time_array.values
    
    # Make tqdm description explicit so it's easy to see in logs / notebooks
    tqdm_desc = f"  {var_name}: merging points"
    
    # Process in batches to reduce memory usage
    # For large datasets, process in batches and force garbage collection
    import gc
    # Adjust batch size based on number of points - smaller batches for very large datasets
    if len(all_coords) > 2000:
        batch_size = 50  # Smaller batches for large datasets to reduce memory pressure
    else:
        batch_size = 100  # Process 100 points at a time, then garbage collect
    
    for coord_idx, coord in enumerate(tqdm(all_coords, desc=tqdm_desc)):
        lon, lat = coord
        point = Point(lon, lat)
        grid_indices = coord_to_grids[coord]
        
        # Determine which grids contain this point
        containing_grids = []
        for grid_idx, site_idx in grid_indices:
            if polygons[grid_idx].contains(point) or polygons[grid_idx].touches(point):
                containing_grids.append((grid_idx, site_idx))
        
        if len(containing_grids) == 0:
            # Point not in any polygon, use closest grid
            containing_grids = [grid_indices[0]]
        
        # Get data and compute weights
        if len(containing_grids) == 1:
            # Single grid: use its value directly (even if it has quality issues, we have no alternative)
            grid_idx, site_idx = containing_grids[0]
            ds = datasets[grid_idx]
            # Load data - if chunked, this will load only the needed chunks
            data_array = ds[var_name].isel(site=site_idx)
            # For chunked arrays, compute() loads the data efficiently
            if hasattr(data_array.data, 'compute'):
                data = data_array.compute().values
            else:
                data = data_array.values
            weights = [0.0] * len(datasets)
            weights[grid_idx] = 1.0
            
            # Note: For single grid case, we still use the data even if it has quality issues
            # because there's no alternative grid to use. Quality checks are only applied
            # when multiple grids overlap and we can choose between them.
        else:
            # Multiple grids: blend using distance-based weights (with quality adjustment)
            # Compute weights based on distance to each grid's border
            # Farther from border (deeper inside grid) = higher weight
            distances = []
            grid_data_list = []
            grid_indices_list = []
            
            # Get data from all grids first
            for grid_idx, site_idx in containing_grids:
                dist = compute_distance_to_border(point, polygons[grid_idx])
                distances.append((grid_idx, dist))
                
                # Get data for this grid
                ds = datasets[grid_idx]
                data_array = ds[var_name].isel(site=site_idx)
                # For chunked arrays, compute() loads the data efficiently
                if hasattr(data_array.data, 'compute'):
                    grid_data = data_array.compute().values
                else:
                    grid_data = data_array.values
                grid_data_list.append(grid_data)
                grid_indices_list.append(grid_idx)
            
            # Use sigmoid-based weighting (same as compute_blending_weights)
            # Convert distances to km for sigmoid computation
            dist_km_list = [(grid_idx, d * 111.0) for grid_idx, d in distances]
            
            # Compute raw weights using sigmoid (farther from border = higher weight)
            raw_weights = []
            for grid_idx, dist_km in dist_km_list:
                raw_weight = 1.0 / (1.0 + np.exp(-steepness * dist_km))
                raw_weights.append((grid_idx, raw_weight))
            
            # Normalize spatial weights so they sum to 1
            total_raw_weight = sum(w for _, w in raw_weights)
            spatial_weights = []
            for grid_idx, raw_weight in raw_weights:
                spatial_weight = raw_weight / total_raw_weight if total_raw_weight > 0 else 1.0 / len(raw_weights)
                spatial_weights.append(spatial_weight)
            
            # Apply quality checks if enabled
            if use_quality_checks:
                quality_adjusted_weights = []
                for i, (grid_idx, grid_data) in enumerate(zip(grid_indices_list, grid_data_list)):
                    # Check quality for this grid's data
                    good_fraction, has_major_issue = _check_data_quality_simple(
                        grid_data, times, min_consecutive_days, zero_threshold, constant_tolerance
                    )
                    
                    # Adjust weight based on quality
                    # If major issue, reduce weight significantly
                    # If good fraction is low, reduce weight proportionally
                    if has_major_issue:
                        # Major issue: reduce weight to 10% of original
                        quality_factor = 0.1
                    elif good_fraction < 0.7:
                        # Some issues: reduce weight proportionally
                        quality_factor = good_fraction
                    else:
                        # Good quality: keep full weight
                        quality_factor = 1.0
                    
                    adjusted_weight = spatial_weights[i] * quality_factor
                    quality_adjusted_weights.append(adjusted_weight)
                
                # Renormalize adjusted weights so they sum to 1
                total_adjusted = sum(quality_adjusted_weights)
                if total_adjusted > 1e-10:
                    final_weights = [w / total_adjusted for w in quality_adjusted_weights]
                else:
                    # All grids have major issues, fall back to equal weights
                    final_weights = [1.0 / len(quality_adjusted_weights)] * len(quality_adjusted_weights)
            else:
                # No quality checks: use spatial weights directly
                final_weights = spatial_weights
            
            # Blend data using final weights
            weights = [0.0] * len(datasets)
            blended_data_array = None
            
            for i, (grid_idx, grid_data, weight) in enumerate(zip(grid_indices_list, grid_data_list, final_weights)):
                weights[grid_idx] = weight
                
                # Accumulate weighted data
                if blended_data_array is None:
                    blended_data_array = weight * grid_data
                else:
                    blended_data_array += weight * grid_data
            
            data = blended_data_array
        
        merged_data.append(data)
        merged_coords.append([lon, lat])
        merged_weights.append(weights)
        
        # Periodic garbage collection for large datasets to prevent memory buildup
        if (coord_idx + 1) % batch_size == 0:
            # Clear intermediate variables to free memory
            del data
            if 'grid_data_list' in locals():
                del grid_data_list
            if 'blended_data_array' in locals():
                del blended_data_array
            gc.collect()
    
    merged_data = np.array(merged_data)  # Shape: (n_sites, n_times)
    merged_coords = np.array(merged_coords)  # Shape: (n_sites, 2)
    
    # Get time coordinates from first dataset (all should have same time)
    times = datasets[0].time.values
    
    # Create new sequential IDs
    site_ids = [f"Merged_{i:07d}" for i in range(len(merged_data))]
    
    print(f"\n  Merged {len(merged_data)} points")
    print(f"  Time range: {pd.Timestamp(times[0])} to {pd.Timestamp(times[-1])}")
    print(f"  Total timesteps: {len(times)}")
    
    # Create merged dataset
    ds_merged = xr.Dataset(
        {
            var_name: (['time', 'site'], merged_data.T, {
                'long_name': datasets[0][var_name].attrs.get('long_name', var_name),
                'units': datasets[0][var_name].attrs.get('units', ''),
            }),
            'lon': (['site'], merged_coords[:, 0]),
            'lat': (['site'], merged_coords[:, 1]),
            'coord_x': (['site'], merged_coords[:, 0]),
            'coord_y': (['site'], merged_coords[:, 1]),
            'site_id': (['site'], site_ids),
        },
        coords={
            'time': times,
            'site': np.arange(len(merged_data))
        },
        attrs={
            'grids': ', '.join(grid_names),
            'variable': var_name,
            'n_grids': len(grid_files),
            'steepness': steepness,
        }
    )
    
    # Add weight information as data variables
    for i, grid_name in enumerate(grid_names):
        weights_array = np.array([w[i] for w in merged_weights])
        ds_merged[f'weight_{grid_name}'] = (['site'], weights_array)
    
    # Save if output file specified
    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        encoding = {
            var_name: {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
            'lon': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
            'lat': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
            'coord_x': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
            'coord_y': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
        }
        
        # Add encoding for weight variables
        for grid_name in grid_names:
            encoding[f'weight_{grid_name}'] = {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
        
        ds_merged.to_netcdf(output_file, encoding=encoding)
        file_size = output_file.stat().st_size / (1024**2)
        print(f"\n✓ Saved: {output_file.name} ({file_size:.1f} MB)")
    
    # Close all datasets
    for ds in datasets:
        ds.close()
    
    print(f"{'='*80}\n")
    return ds_merged


def combine_time_period_files(file_patterns, var_name='hs', output_file=None, sort_by_time=True):
    """
    Combine multiple NetCDF files from different time periods into a single file.
    
    This function is useful when processing long time periods in chunks (e.g., 1-2 years at a time)
    and then combining them into a single dataset.
    
    Parameters:
    -----------
    file_patterns : list of str or Path, or str (glob pattern)
        List of file paths to combine, or a glob pattern string (e.g., "hs_*_grid1.nc")
        Files should have the same variable, grid, and sites (only time should differ)
    var_name : str
        Variable name (default: 'hs')
    output_file : str or Path, optional
        Output file path. If None, creates combined_{var_name}_{grid}.nc
    sort_by_time : bool
        If True, sort files by time before combining (default: True)
        
    Returns:
    --------
    xarray.Dataset : Combined dataset with all time periods concatenated
    """
    from glob import glob
    
    print(f"\n{'='*80}")
    print(f"Combining time period files for {var_name}")
    print(f"{'='*80}")
    
    # Handle glob pattern or list of files
    if isinstance(file_patterns, str):
        file_list = sorted(glob(file_patterns))
        print(f"Found {len(file_list)} files matching pattern: {file_patterns}")
    else:
        file_list = [str(f) for f in file_patterns]
        print(f"Combining {len(file_list)} files")
    
    if len(file_list) == 0:
        raise ValueError("No files found to combine!")
    
    # Load all datasets
    print(f"\nLoading datasets...")
    datasets = []
    for i, filepath in enumerate(file_list):
        print(f"  [{i+1}/{len(file_list)}] Loading {Path(filepath).name}...")
        ds = xr.open_dataset(filepath)
        datasets.append(ds)
    
    # Check that all datasets have the same structure (except time)
    print(f"\nValidating datasets...")
    first_ds = datasets[0]
    grid_name = first_ds.attrs.get('grid', 'unknown')
    n_sites = first_ds.dims['site']
    
    for i, ds in enumerate(datasets[1:], 1):
        if ds.dims['site'] != n_sites:
            raise ValueError(f"File {i+1} has {ds.dims['site']} sites, expected {n_sites}")
        if ds.attrs.get('grid') != grid_name:
            print(f"  ⚠ Warning: File {i+1} has different grid name: {ds.attrs.get('grid')}")
    
    # Sort by time if requested
    if sort_by_time:
        print(f"\nSorting datasets by time...")
        time_starts = [pd.Timestamp(ds.time.values[0]) for ds in datasets]
        sorted_indices = np.argsort(time_starts)
        datasets = [datasets[i] for i in sorted_indices]
        file_list = [file_list[i] for i in sorted_indices]
        print(f"  Time range: {time_starts[sorted_indices[0]]} to {time_starts[sorted_indices[-1]]}")
    
    # Concatenate along time dimension
    print(f"\nConcatenating along time dimension...")
    combined_ds = xr.concat(datasets, dim='time')
    
    # Sort by time to ensure chronological order
    combined_ds = combined_ds.sortby('time')
    
    # Get time range
    time_start = pd.Timestamp(combined_ds.time.values[0])
    time_end = pd.Timestamp(combined_ds.time.values[-1])
    
    print(f"  Combined time range: {time_start} to {time_end}")
    print(f"  Total timesteps: {combined_ds.dims['time']}")
    print(f"  Total sites: {combined_ds.dims['site']}")
    
    # Update attributes
    combined_ds.attrs.update({
        'title': f'BinWaves {var_name} Time Series - Combined',
        'description': f'Combined time series from multiple time periods',
        'grid': grid_name,
        'n_sites': int(combined_ds.dims['site']),
        'n_timesteps': int(combined_ds.dims['time']),
        'time_start': str(time_start),
        'time_end': str(time_end),
        'source_files': len(file_list),
        'combined_from': [Path(f).name for f in file_list],
    })
    
    # Save if output file specified
    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare encoding
        encoding = {
            var_name: {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
            'lon': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
            'lat': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
        }
        
        # Add polygon encoding if present
        if 'polygon_lon' in combined_ds.data_vars:
            encoding['polygon_lon'] = {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
            encoding['polygon_lat'] = {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
        
        print(f"\nSaving combined dataset...")
        combined_ds.to_netcdf(output_file, encoding=encoding)
        file_size = output_file.stat().st_size / (1024**2)
        print(f"✓ Saved: {output_file.name} ({file_size:.1f} MB)")
    
    # Close all datasets
    for ds in datasets:
        ds.close()
    
    print(f"{'='*80}\n")
    
    return combined_ds


def netcdf_to_geojson_statistics(nc_file, output_geojson, time_start=None, time_end=None):
    """
    Read NetCDF time series files and generate a GeoJSON file with wave statistics.
    
    This function reads NetCDF files containing time series data (hs, tp, tm02, dm, dp)
    from the structure: {grid}/outputs/output_variables/{var}/nc_{var}_{grid}_{year}.nc
    and computes statistics (mean, min, max) for each site, then saves them as a GeoJSON file.
    
    Parameters:
    -----------
    nc_file : str or Path
        Path to any NetCDF time series file (e.g., grid1/outputs/output_variables/hs/nc_hs_grid1_2016.nc)
        The function will automatically find all variable files (hs, tp, tm02, dm, dp) in the same structure
    output_geojson : str or Path
        Output GeoJSON file path
    time_start : str, optional
        Start time for statistics (e.g., "2016-01-01"). If None, uses all available time
    time_end : str, optional
        End time for statistics (e.g., "2016-12-31"). If None, uses all available time
        
    Returns:
    --------
    dict : GeoJSON FeatureCollection dictionary
    """
    nc_file = Path(nc_file)
    output_geojson = Path(output_geojson)
    
    print(f"\n{'='*80}")
    print(f"Generating GeoJSON statistics from NetCDF time series")
    print(f"{'='*80}")
    print(f"Input NetCDF: {nc_file.name}")
    print(f"Output GeoJSON: {output_geojson.name}")
    
    # Extract grid name and year from path and filename
    # Path structure: .../grid1/outputs/output_variables/hs/nc_hs_grid1_2016.nc
    path_parts = nc_file.parts
    grid_name = None
    year = None
    
    # Find grid name from path (look for grid1, grid2, grid3, grid4)
    for part in path_parts:
        if part.startswith('grid') and part[4:].isdigit():
            grid_name = part
            break
    
    # Extract year from filename (e.g., nc_hs_grid1_2016.nc -> 2016)
    filename_parts = nc_file.stem.split('_')
    for part in filename_parts:
        if part.isdigit() and len(part) == 4:  # Year should be 4 digits
            year = part
            break
    
    if grid_name is None:
        raise ValueError(f"Could not extract grid name from path: {nc_file}")
    if year is None:
        raise ValueError(f"Could not extract year from filename: {nc_file.name}")
    
    print(f"  Detected grid: {grid_name}, year: {year}")
    
    # Find base directory (should be at: .../grid1/outputs/output_variables/)
    # File structure: .../grid1/outputs/output_variables/{var}/nc_{var}_{grid}_{year}.nc
    # So base_dir is parent.parent (two levels up from the file)
    base_dir = nc_file.parent.parent
    
    # Verify base_dir exists and has the expected structure
    if not base_dir.exists():
        raise ValueError(f"Base directory does not exist: {base_dir}")
    
    # Variables to process
    variables = ['hs', 'tp', 'tm02', 'dm', 'dp']
    var_files = {}
    var_datasets = {}
    var_data = {}
    
    print(f"\nLoading NetCDF files from: {base_dir}")
    print(f"  Looking for files in subdirectories: {', '.join(variables)}")
    
    # Load each variable file
    for var in variables:
        var_dir = base_dir / var
        var_file = var_dir / f"nc_{var}_{grid_name}_{year}.nc"
        
        if var_file.exists():
            print(f"  ✓ Loading {var} from: {var_file}")
            ds = xr.open_dataset(var_file)
            var_files[var] = var_file
            var_datasets[var] = ds
            var_data[var] = ds[var]
        else:
            print(f"  ✗ Could not find {var} file: {var_file}")
            print(f"     Checked path: {var_file.absolute()}")
            raise ValueError(f"Could not find {var} data file: {var_file}")
    
    # Use the first dataset to get coordinates and site info
    # All datasets should have the same structure
    ds_ref = var_datasets[variables[0]]
    
    # Get coordinates - prefer coord_x/coord_y, fallback to lon/lat
    if 'coord_x' in ds_ref.coords:
        lon = ds_ref.coord_x.values
    elif 'lon' in ds_ref.coords or 'lon' in ds_ref.data_vars:
        lon = ds_ref.lon.values
    else:
        lon = var_data['hs'].coord_x.values if 'coord_x' in var_data['hs'].coords else var_data['hs'].lon.values
    
    if 'coord_y' in ds_ref.coords:
        lat = ds_ref.coord_y.values
    elif 'lat' in ds_ref.coords or 'lat' in ds_ref.data_vars:
        lat = ds_ref.lat.values
    else:
        lat = var_data['hs'].coord_y.values if 'coord_y' in var_data['hs'].coords else var_data['hs'].lat.values
    
    # Get site IDs if available
    if 'site_id' in ds_ref.data_vars:
        site_ids = ds_ref.site_id.values
    elif 'site_id' in ds_ref.coords:
        site_ids = ds_ref.site_id.values
    else:
        n_sites = len(lon)
        site_ids = [f"BinWaves_{i:07d}" for i in range(n_sites)]
    
    # Filter by time if specified
    if time_start is not None or time_end is not None:
        print(f"\nFiltering time period: {time_start} to {time_end}")
        for var in variables:
            var_data[var] = var_data[var].sel(time=slice(time_start, time_end))
    
    print(f"\nComputing statistics for {len(lon)} sites...")
    if len(var_data['hs'].time) > 0:
        print(f"  Time range: {pd.Timestamp(var_data['hs'].time.values[0])} to {pd.Timestamp(var_data['hs'].time.values[-1])}")
        print(f"  Total timesteps: {len(var_data['hs'].time)}")
    
    # Compute statistics for each site
    features = []
    for site_idx in tqdm(range(len(lon)), desc="  Processing sites"):
        try:
            # Extract time series for this site for all variables
            var_ts = {}
            for var in variables:
                var_ts[var] = var_data[var].isel(site=site_idx).values
            
            # Compute statistics for each variable
            stats = {}
            for var in variables:
                ts = var_ts[var]
                stats[f"{var}_mean"] = float(np.nanmean(ts))
                stats[f"{var}_min"] = float(np.nanmin(ts))
                stats[f"{var}_max"] = float(np.nanmax(ts))
            
            # For directional variables (dm, dp), compute circular mean
            for var in ['dm', 'dp']:
                if var in var_ts:
                    ts = var_ts[var]
                    # Circular mean for directional data
                    dir_rad = np.deg2rad(ts)
                    dir_sin = np.nanmean(np.sin(dir_rad))
                    dir_cos = np.nanmean(np.cos(dir_rad))
                    stats[f"{var}_mean"] = float(np.rad2deg(np.arctan2(dir_sin, dir_cos)) % 360)
            
            # Create GeoJSON feature
            feature = {
                "type": "Feature",
                "properties": {
                    "id": str(site_ids[site_idx]) if isinstance(site_ids[site_idx], (str, np.str_)) else f"BinWaves_{site_idx:07d}",
                    "grid": grid_name,
                    "dataset_index": int(site_idx),
                    "original_index": int(site_idx),
                    # Wave statistics
                    "Hs_mean": stats["hs_mean"],
                    "Hs_min": stats["hs_min"],
                    "Hs_max": stats["hs_max"],
                    "Tp_mean": stats["tp_mean"],
                    "Tp_min": stats["tp_min"],
                    "Tp_max": stats["tp_max"],
                    "Tm02_mean": stats["tm02_mean"],
                    "Tm02_min": stats["tm02_min"],
                    "Tm02_max": stats["tm02_max"],
                    "Dm_mean": stats["dm_mean"],
                    "Dm_min": stats["dm_min"],
                    "Dm_max": stats["dm_max"],
                    "Dp_mean": stats["dp_mean"],
                    "Dp_min": stats["dp_min"],
                    "Dp_max": stats["dp_max"],
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lon[site_idx]), float(lat[site_idx])]
                }
            }
            features.append(feature)
            
        except Exception as e:
            print(f"    ✗ Error processing site {site_idx}: {e}")
            continue
    
    # Create GeoJSON FeatureCollection
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }
    
    # Save to file
    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    with open(output_geojson, 'w') as f:
        json.dump(geojson_data, f, indent=2)
    
    file_size = output_geojson.stat().st_size / (1024**2)
    print(f"\n{'='*80}")
    print(f"✓ Saved GeoJSON: {output_geojson.name} ({file_size:.1f} MB)")
    print(f"  Total features: {len(features)}")
    print(f"{'='*80}\n")
    
    # Close all datasets
    for var in variables:
        if var in var_datasets:
            var_datasets[var].close()
    
    return geojson_data


def compute_timeseries_statistics_json(
    grid_name,
    timeseries_dir,
    time_start=None,
    time_end=None,
    output_json=None
):
    """
    Compute statistics from NetCDF time series files and save to GeoJSON format.
    
    This function reads hs, tp, and dir NetCDF files for a given grid,
    computes statistics (mean, min, max) for each site, and saves them
    to a GeoJSON file matching the format of wave_statistics_gridA_cutted.geojson.
    
    Parameters:
    -----------
    grid_name : str
        Name of the grid (e.g., "grid1")
    timeseries_dir : str or Path
        Directory containing the NetCDF time series files
    time_start : str, optional
        Start time for statistics filtering (e.g., "2000-01-01")
        If None, uses all available time
    time_end : str, optional
        End time for statistics filtering (e.g., "2000-12-31")
        If None, uses all available time
    output_json : str or Path, optional
        Output GeoJSON file path. If None, creates:
        wave_statistics_{grid_name}.geojson
        
    Returns:
    --------
    dict : GeoJSON FeatureCollection dictionary
    """
    timeseries_dir = Path(timeseries_dir)
    
    print(f"\n{'='*80}")
    print(f"Computing statistics from NetCDF time series")
    print(f"{'='*80}")
    print(f"Grid: {grid_name}")
    print(f"Directory: {timeseries_dir}")
    
    # Find NetCDF files for this grid
    hs_files = list(timeseries_dir.glob(f"hs_*_{grid_name}.nc"))
    tp_files = list(timeseries_dir.glob(f"tp_*_{grid_name}.nc"))
    dir_files = list(timeseries_dir.glob(f"dir_*_{grid_name}.nc"))
    
    if not hs_files:
        raise ValueError(f"No hs files found for {grid_name} in {timeseries_dir}")
    if not tp_files:
        raise ValueError(f"No tp files found for {grid_name} in {timeseries_dir}")
    if not dir_files:
        raise ValueError(f"No dir files found for {grid_name} in {timeseries_dir}")
    
    # Use the first matching file (assuming single time period or most recent)
    # If multiple files, use the one matching the time period if specified
    hs_file = hs_files[0]
    tp_file = tp_files[0]
    dir_file = dir_files[0]
    
    # If time period specified, try to find matching files
    if time_start and time_end:
        time_start_str = pd.Timestamp(time_start).strftime('%Y%m%d')
        time_end_str = pd.Timestamp(time_end).strftime('%Y%m%d')
        
        pattern_hs = f"hs_{time_start_str}_{time_end_str}_{grid_name}.nc"
        pattern_tp = f"tp_{time_start_str}_{time_end_str}_{grid_name}.nc"
        pattern_dir = f"dir_{time_start_str}_{time_end_str}_{grid_name}.nc"
        
        matching_hs = [f for f in hs_files if f.name == pattern_hs]
        matching_tp = [f for f in tp_files if f.name == pattern_tp]
        matching_dir = [f for f in dir_files if f.name == pattern_dir]
        
        if matching_hs:
            hs_file = matching_hs[0]
        if matching_tp:
            tp_file = matching_tp[0]
        if matching_dir:
            dir_file = matching_dir[0]
    
    print(f"\nLoading NetCDF files...")
    print(f"  hs: {hs_file.name}")
    print(f"  tp: {tp_file.name}")
    print(f"  dir: {dir_file.name}")
    
    # Load datasets
    ds_hs = xr.open_dataset(hs_file)
    ds_tp = xr.open_dataset(tp_file)
    ds_dir = xr.open_dataset(dir_file)
    
    # Get data arrays
    hs_data = ds_hs.hs
    tp_data = ds_tp.tp
    dir_data = ds_dir.dir
    
    # Filter by time if specified
    if time_start is not None or time_end is not None:
        print(f"\nFiltering time period: {time_start} to {time_end}")
        hs_data = hs_data.sel(time=slice(time_start, time_end))
        tp_data = tp_data.sel(time=slice(time_start, time_end))
        dir_data = dir_data.sel(time=slice(time_start, time_end))
    
    # Get coordinates and site info
    lon = ds_hs.lon.values
    lat = ds_hs.lat.values
    
    # Get site IDs if available
    if 'site_id' in ds_hs.data_vars:
        site_ids = ds_hs.site_id.values
    else:
        n_sites = len(lon)
        site_ids = [f"BinWaves_{i:07d}" for i in range(n_sites)]
    
    n_sites = len(lon)
    n_timesteps = len(hs_data.time)
    
    print(f"\nComputing statistics...")
    print(f"  Sites: {n_sites}")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  Time range: {pd.Timestamp(hs_data.time.values[0])} to {pd.Timestamp(hs_data.time.values[-1])}")
    
    # Compute statistics for each site and create GeoJSON features
    features = []
    
    for site_idx in tqdm(range(n_sites), desc="  Processing sites"):
        try:
            # Extract time series for this site
            hs_ts = hs_data.isel(site=site_idx).values
            tp_ts = tp_data.isel(site=site_idx).values
            dir_ts = dir_data.isel(site=site_idx).values
            
            # Compute statistics
            hs_mean = float(np.nanmean(hs_ts))
            hs_min = float(np.nanmin(hs_ts))
            hs_max = float(np.nanmax(hs_ts))
            
            # Handle NaN values for tp and dir
            tp_mean = float(np.nanmean(tp_ts)) if not np.all(np.isnan(tp_ts)) else np.nan
            tp_min = float(np.nanmin(tp_ts)) if not np.all(np.isnan(tp_ts)) else np.nan
            tp_max = float(np.nanmax(tp_ts)) if not np.all(np.isnan(tp_ts)) else np.nan
            
            # Circular mean for directional data
            dir_valid = dir_ts[~np.isnan(dir_ts)]
            if len(dir_valid) > 0:
                dir_rad = np.deg2rad(dir_valid)
                dir_sin = np.nanmean(np.sin(dir_rad))
                dir_cos = np.nanmean(np.cos(dir_rad))
                dir_mean = float(np.rad2deg(np.arctan2(dir_sin, dir_cos)) % 360)
                dir_min = float(np.nanmin(dir_valid))
                dir_max = float(np.nanmax(dir_valid))
            else:
                dir_mean = np.nan
                dir_min = np.nan
                dir_max = np.nan
            
            # Create GeoJSON feature matching the format of wave_statistics_gridA_cutted.geojson
            site_id_str = str(site_ids[site_idx]) if isinstance(site_ids[site_idx], (str, np.str_)) else f"BinWaves_{site_idx:07d}"
            
            feature = {
                "type": "Feature",
                "properties": {
                    "id": site_id_str,
                    "grid": grid_name,
                    "dataset_index": int(site_idx),
                    "original_index": int(site_idx),  # Use dataset_index as original_index if not available
                    "Hs_mean": hs_mean,
                    "Hs_min": hs_min,
                    "Hs_max": hs_max,
                    "Tp_mean": tp_mean if not np.isnan(tp_mean) else None,
                    "Tp_min": tp_min if not np.isnan(tp_min) else None,
                    "Tp_max": tp_max if not np.isnan(tp_max) else None,
                    "Dp_mean": dir_mean if not np.isnan(dir_mean) else None,
                    "Dp_min": dir_min if not np.isnan(dir_min) else None,
                    "Dp_max": dir_max if not np.isnan(dir_max) else None,
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lon[site_idx]), float(lat[site_idx])]
                }
            }
            features.append(feature)
            
        except Exception as e:
            print(f"    ✗ Error processing site {site_idx}: {e}")
            continue
    
    # Create GeoJSON FeatureCollection
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }
    
    # Determine output file path
    if output_json is None:
        output_json = timeseries_dir / f"wave_statistics_{grid_name}.geojson"
    else:
        output_json = Path(output_json)
        # Ensure .geojson extension
        if output_json.suffix.lower() != '.geojson':
            output_json = output_json.with_suffix('.geojson')
    
    # Save to GeoJSON file with custom handling for NaN values
    output_json.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSON with NaN as literal (matching the example file format)
    # Note: This creates technically invalid JSON, but matches the example file format
    json_str = json.dumps(geojson_data, indent=2)
    # Replace null with NaN (unquoted) for numeric fields to match example format
    # Use string replacement - replace "null" with literal NaN (unquoted)
    nan_str = 'NaN'  # String literal for NaN
    json_str = json_str.replace('"Tp_mean": null', '"Tp_mean": ' + nan_str)
    json_str = json_str.replace('"Tp_min": null', '"Tp_min": ' + nan_str)
    json_str = json_str.replace('"Tp_max": null', '"Tp_max": ' + nan_str)
    json_str = json_str.replace('"Dp_mean": null', '"Dp_mean": ' + nan_str)
    json_str = json_str.replace('"Dp_min": null', '"Dp_min": ' + nan_str)
    json_str = json_str.replace('"Dp_max": null', '"Dp_max": ' + nan_str)
    
    with open(output_json, 'w') as f:
        f.write(json_str)
    
    file_size = output_json.stat().st_size / (1024**2)
    print(f"\n{'='*80}")
    print(f"✓ Saved GeoJSON statistics: {output_json.name} ({file_size:.2f} MB)")
    print(f"  Total sites: {len(features)}")
    print(f"{'='*80}\n")
    
    # Close datasets
    ds_hs.close()
    ds_tp.close()
    ds_dir.close()
    
    return geojson_data


