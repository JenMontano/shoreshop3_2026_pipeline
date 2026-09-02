#!/usr/bin/env bash
# Serve the built Jupyter Book over HTTP (for viewing on a remote cluster via SSH tunnel).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HTML="$ROOT/docs/_build/html"
PORT="${1:-8765}"

if [[ ! -f "$HTML/intro.html" ]]; then
  echo "Book not built yet. Run: bash docs/build.sh"
  exit 1
fi

echo "Serving ShoreShop3_2026 docs at http://127.0.0.1:${PORT}/"
echo ""
echo "On your laptop (new terminal), if SSH'd to this cluster:"
echo "  ssh -L ${PORT}:localhost:${PORT} montanoj@geocean05"
echo ""
echo "Then open in your browser:"
echo "  http://localhost:${PORT}/intro.html"
echo ""
echo "Press Ctrl+C to stop."
cd "$HTML"
python3 -m http.server "$PORT"
