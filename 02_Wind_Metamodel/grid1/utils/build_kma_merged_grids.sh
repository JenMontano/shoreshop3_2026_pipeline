#!/usr/bin/env bash
# Build cropped_variables-format NetCDFs with BinWaves + KMA cluster SWAN bulk (grid1).
# Input:  ShoreShop cropped_variables (hs_grid1_masked.nc, phs0_grid1_masked.nc, ...)
# Output: grid1/outputs/BinWaves_BMUS/
#   phs0_grid1_BinWaves_BMUS.nc, ptp0_grid1_BinWaves_BMUS.nc, dp0_grid1_BinWaves_BMUS.nc,
#   hs_grid1_BinWaves_BMUS.nc, tp_grid1_BinWaves_BMUS.nc, dp_grid1_BinWaves_BMUS.nc,
#   dm_grid1_BinWaves_BMUS.nc, tm02_grid1_BinWaves_BMUS.nc
# Same seapoints, coords, and chunk layout as the ShoreShop cropped_variables folder.
# Usage: bash utils/build_kma_merged_grids.sh
# Pass --no-dm or --no-tm02 to skip optional outputs.

set -euo pipefail
GRID1="/lustre/geocean/WORK/users/montanoj/personal/Wind_Metamodel/grid1"
INPUT="/lustre/geocean/WORK/users/montanoj/personal/ShoreShop2026/outputs/cropped_variables"
OUTPUT="$GRID1/outputs/BinWaves_BMUS"
PYTHON="/nfs/home/geocean/montanoj/miniforge3/envs/bluemath-dev/bin/python3"

cd "$GRID1"
export PYTHONPATH="$GRID1:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

"$PYTHON" -m utils.build_kma_merged_grids \
  --input-folder "$INPUT" \
  --output-folder "$OUTPUT" \
  --grid-id 1 \
  --project-root "$GRID1" \
  --cluster-cases-root inputs/CASES_ONLY_WIND \
  --kma-bmu-csv outputs/KMA/nearest_centroids_idxs_kma_pcs.csv \
  --time-chunk 8760 \
  --overwrite \
  "$@"
