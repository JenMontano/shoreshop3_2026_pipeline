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
GRID="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$GRID/../.." && pwd)"
INPUT="$REPO_ROOT/01_BinWaves/outputs/cropped_variables"
OUTPUT="$GRID/outputs/BinWaves_BMUS"
PYTHON="${PYTHON:-python3}"

cd "$GRID"
export PYTHONPATH="$GRID:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

"$PYTHON" -m utils.build_kma_merged_grids \
  --input-folder "$INPUT" \
  --output-folder "$OUTPUT" \
  --grid-id 1 \
  --project-root "$GRID" \
  --cluster-cases-root inputs/CASES_ONLY_WIND \
  --kma-bmu-csv outputs/KMA/nearest_centroids_idxs_kma_pcs.csv \
  --time-chunk 8760 \
  --overwrite \
  "$@"
