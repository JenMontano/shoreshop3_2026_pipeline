#!/usr/bin/env bash
# Build cropped_variables-format NetCDFs with BinWaves + KMA cluster SWAN bulk (grid4).
# Input:  ShoreShop cropped_variables (hs_grid4_masked.nc, phs0_grid4_masked.nc, ...)
# Output: grid4/outputs/BinWaves_BMUS/
# Usage: bash utils/build_kma_merged_grids.sh
# Pass --no-dm or --no-tm02 to skip optional outputs.

set -euo pipefail
GRID4="/lustre/geocean/WORK/users/montanoj/personal/Wind_Metamodel/grid4"
INPUT="/lustre/geocean/WORK/users/montanoj/personal/ShoreShop2026/outputs/cropped_variables"
OUTPUT="$GRID4/outputs/BinWaves_BMUS"
PYTHON="/nfs/home/geocean/montanoj/miniforge3/envs/bluemath-dev/bin/python3"

cd "$GRID4"
export PYTHONPATH="$GRID4:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

"$PYTHON" -m utils.build_kma_merged_grids \
  --input-folder "$INPUT" \
  --output-folder "$OUTPUT" \
  --grid-id 4 \
  --project-root "$GRID4" \
  --cluster-cases-root inputs/CASES_ONLY_WIND \
  --kma-bmu-csv outputs/KMA/nearest_centroids_idxs_kma_pcs.csv \
  --time-chunk 8760 \
  --overwrite \
  "$@"
