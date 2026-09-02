#!/usr/bin/env bash
# Build the ShoreShop3_2026 documentation book (README → HTML).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if command -v conda >/dev/null 2>&1 && conda env list | grep -q bluemath-dev; then
  conda run -n bluemath-dev jupyter-book build docs
else
  jupyter-book build docs
fi

echo ""
echo "Open: file://${ROOT}/docs/_build/html/index.html"
