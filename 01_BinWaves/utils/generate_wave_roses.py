#!/usr/bin/env python3
"""
Generate wave rose plots for each buoy in the NetCDF file.
Each plot is saved as a separate .png file with transparent background.
Note: PNG format is used instead of JPG to support transparency.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import cm
from windrose import WindroseAxes

# Configuration
INPUT_FILE = "./inputs/buoys&tideGauges/buoy_data_NorthCarolina.nc"
OUTPUT_DIR = "./figures/wave_roses"
DPI = 300

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the NetCDF file
print(f"Loading data from {INPUT_FILE}...")
ds = xr.open_dataset(INPUT_FILE)

# Get all buoy IDs
buoy_ids = ds.buoy_id.values
print(f"Found {len(buoy_ids)} buoys: {buoy_ids}")

# Process each buoy
for buoy_id in buoy_ids:
    print(f"\nProcessing buoy: {buoy_id}")
    
    # Select data for this buoy
    buoy_data = ds.sel(buoy_id=buoy_id)
    
    # Extract wave direction and significant wave height
    dir_buoy = buoy_data.Dir_Buoy.values
    hs_buoy = buoy_data.Hs_Buoy.values
    
    # Remove NaN values
    valid_mask = ~(np.isnan(dir_buoy) | np.isnan(hs_buoy))
    dir_buoy_clean = dir_buoy[valid_mask]
    hs_buoy_clean = hs_buoy[valid_mask]
    
    if len(dir_buoy_clean) == 0:
        print(f"  Warning: No valid data for buoy {buoy_id}, skipping...")
        continue
    
    print(f"  Valid data points: {len(dir_buoy_clean)}")
    
    # Calculate appropriate bins based on data distribution for this buoy
    hs_max = float(np.max(hs_buoy_clean))
    hs_95th = float(np.percentile(hs_buoy_clean, 95))
    hs_99th = float(np.percentile(hs_buoy_clean, 99))
    
    # Create bins that cover the data range appropriately
    # Start with fine bins for lower values (matching user's example)
    bins = [0, 1, 1.5, 2]
    
    # Add intermediate bins up to 95th percentile with 0.5m increments
    current = 2.0
    while current < hs_95th:
        current += 0.5
        bins.append(current)
    
    # Add bins up to 95th percentile (instead of max) for better color distribution
    # This ensures warmer colors (yellow/orange/red) are more visible
    target_max = hs_95th * 1.2  # Use 95th percentile with 20% margin
    # Round up to nearest 0.5m
    target_max = np.ceil(target_max * 2) / 2
    
    # Continue adding bins up to target_max
    while current < target_max:
        if current < 5:
            current += 0.5
        elif current < 10:
            current += 1.0
        else:
            current += 2.0
        if current <= target_max:
            bins.append(float(current))
    
    # Remove duplicates and sort, convert to float
    bins = sorted([float(b) for b in set(bins)])
    
    print(f"  Wave height range: {np.min(hs_buoy_clean):.2f}m - {hs_max:.2f}m")
    print(f"  95th percentile: {hs_95th:.2f}m, 99th percentile: {hs_99th:.2f}m")
    print(f"  Using bins: {bins}")
    
    # Create figure with windrose projection using WindroseAxes
    # Make figure taller to accommodate legend below
    fig = plt.figure(figsize=(8, 9), facecolor='black')
    ax = WindroseAxes.from_ax(fig=fig, rect=[0.1, 0.1, 0.8, 0.8])
    
    # Set radial scale based on 95th percentile instead of max for better visibility
    # This ensures higher wave heights are more visible
    radial_max = hs_95th * 1.1  # Use 95th percentile with 10% margin
    ax.set_rmax(radial_max)
    
    # Set background to black
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    # Plot wave rose
    ax.bar(
        dir_buoy_clean,
        hs_buoy_clean,
        normed=True,
        opening=0.8,
        bins=bins,
        cmap=cm.RdYlGn_r,
        edgecolor='white'
    )
    
    # Set all text and labels to white for black background
    # Radial labels (frequency/percentage numbers)
    ax.tick_params(colors='white', labelsize=10)
    
    # Get all text elements and set to white
    for text in ax.texts:
        text.set_color('white')
    
    # Directional labels (N, S, E, W, etc.)
    ax.set_thetagrids(angles=np.arange(0, 360, 45), labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], 
                      color='white', fontsize=12, fontweight='bold')
    
    # Set radial grid labels to white
    ax.set_rlabel_position(225)  # Position radial labels
    for label in ax.get_yticklabels():
        label.set_color('white')
    
    # Add legend for wave height at bottom right (vertical, single column)
    # Position legend outside the plot area at bottom right, not overlapping
    legend = ax.legend(loc='lower right', bbox_to_anchor=(1.2, 0.0), title='Hs (m)', ncol=1, frameon=True)
    legend.get_frame().set_facecolor('lightgray')
    legend.get_frame().set_edgecolor('black')
    # Set legend text color to black (since legend has light gray background)
    for text in legend.get_texts():
        text.set_color('black')
    legend.get_title().set_color('black')
    
    # Save figure as PNG (supports transparency; JPG does not)
    output_filename = os.path.join(OUTPUT_DIR, f'wave_rose_buoy_{buoy_id}.png')
    plt.savefig(
        output_filename,
        dpi=DPI,
        bbox_inches='tight',
        transparent=True,
        format='png'
    )
    plt.close()
    
    print(f"  Saved: {output_filename}")

print(f"\n✓ Completed! Generated {len(buoy_ids)} wave rose plots in {OUTPUT_DIR}")
ds.close()
