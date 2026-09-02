# ShoreShop3_2026 — Jupyter Book

Static documentation site built from the repo README files. Stage chapters are symlinks to each folder README; the start page (`intro.md`) is a book-specific copy of the root overview with chapter links.

## Build locally

Requires **Python ≥ 3.9** and `jupyter-book` (works in `conda activate bluemath-dev` on this cluster).

**From the repo root:**

```bash
cd /path/to/ShoreShop3_2026
conda activate bluemath-dev   # optional but recommended here
bash docs/build.sh
```

**Already inside `docs/`:**

```bash
bash build.sh
```

Equivalent manual command from the repo root: `jupyter-book build docs`

Open the book in a **web browser** (not the Cursor editor — HTML preview does not work for remote cluster paths).

| Where you work | How to view |
|----------------|-------------|
| **On the cluster (desktop/browser there)** | Open `docs/_build/html/intro.html` in Firefox/Chrome |
| **SSH from laptop** | `bash docs/serve.sh` on cluster, then `ssh -L 8765:localhost:8765 you@geocean05`, open http://localhost:8765/intro.html |
| **In Cursor** | Read the markdown READMEs directly (`README.md`, `01_BinWaves/README.md`, …) — same content as the book |

Start page is **`intro.html`** (`index.html` only redirects there).

## After editing documentation

- **Stage READMEs** (`01_BinWaves/README.md`, …): edit, then re-run `bash docs/build.sh`. Symlinks in `docs/chapters/` pick them up automatically.
- **Root overview** (`README.md`): GitHub-facing links (`01_BinWaves/README.md`, …). After changing prose there, regenerate the book intro:

```bash
python3 docs/sync_intro_from_readme.py   # or copy link substitutions manually into docs/intro.md
```

If you only change stage READMEs, rebuilding the book is enough.

## Layout

| Path | Role |
|------|------|
| `docs/_config.yml` | Book title, theme, notebook execution off |
| `docs/_toc.yml` | Sidebar navigation |
| `docs/intro.md` | Book start page (chapter links; prose mirrors root `README.md`) |
| `docs/chapters/*.md` | Symlinks → each stage README |
| `docs/notebooks/*.ipynb` | Symlinks → **key process notebooks** (rendered, not executed) |
| `docs/_build/html/` | Generated site (gitignored) |

## Note on cross-links

- **GitHub:** use root `README.md` — links point at folder READMEs.
- **Jupyter Book:** use `intro.html` and the sidebar — `intro.md` links use `chapters/…`.
