#!/usr/bin/env bash
# Build cropped_variables-format NetCDFs with BinWaves + KMA cluster SWAN bulk (grid2).
# Input:  ShoreShop cropped_variables (hs_grid2_masked.nc, phs0_grid2_masked.nc, ...)
# Output: grid2/outputs/BinWaves_BMUS/
# Usage: bash utils/build_kma_merged_grids.sh
# Pass --no-dm or --no-tm02 to skip optional outputs.

set -euo pipefail
GRID="/lustre/geocean/WORK/users/montanoj/personal/Wind_Metamodel/grid2"
INPUT="/lustre/geocean/WORK/users/montanoj/personal/ShoreShop2026/outputs/cropped_variables"
OUTPUT="$GRID/outputs/BinWaves_BMUS"
PYTHON="/nfs/home/geocean/montanoj/miniforge3/envs/bluemath-dev/bin/python3"

cd "$GRID"
export PYTHONPATH="$GRID:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

"$PYTHON" -m utils.build_kma_merged_grids \
  --input-folder "$INPUT" \
  --output-folder "$OUTPUT" \
  --grid-id 2 \
  --project-root "$GRID" \
  --cluster-cases-root inputs/CASES_ONLY_WIND \
  --kma-bmu-csv outputs/KMA/nearest_centroids_idxs_kma_pcs.csv \
  --time-chunk 8760 \
  --overwrite \
  "$@"
