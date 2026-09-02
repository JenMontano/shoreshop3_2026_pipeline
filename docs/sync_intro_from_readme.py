#!/usr/bin/env python3
"""Regenerate docs/intro.md from root README.md (book chapter links)."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
INTRO = REPO / "docs" / "intro.md"

REPLACEMENTS = [
    ("([build instructions](docs/README.md)).", "([Build instructions](README))."),
    (
        "see the [documentation map](#documentation-map) below.",
        "use the sidebar or the [documentation map](#documentation-map) below.",
    ),
    (
        "(also in the [Jupyter Book](docs/README.md) sidebar):",
        "(included in the Jupyter Book sidebar):",
    ),
    (
        "## Documentation map\n\n| Folder | Contents |",
        "(documentation-map)=\n## Documentation map\n\n| Chapter | Contents |",
    ),
    ("(01_BinWaves/README.md)", "(chapters/01_binwaves)"),
    ("(02_Wind_Metamodel/README.md)", "(chapters/02_wind_metamodel)"),
    ("(03_NC_Wave_Emulator/README.md)", "(chapters/03_wave_emulator)"),
    ("(03_NC_Wave_Emulator/03A_Stochastic_GCMs_NC/README.md)", "(chapters/03a_stochastic_gcm)"),
    ("(03_NC_Wave_Emulator/03B_Century_GCMs_Waves/README.md)", "(chapters/03b_century_gcm_waves)"),
    ("(04_NC_Cyclones/README.md)", "(chapters/04_cyclones)"),
    ("(04_NC_Cyclones/04A_Cyclones_Metamodel/README.md)", "(chapters/04a_cyclones_metamodel)"),
    ("(04_NC_Cyclones/04B_Cyclones_emulator/README.md)", "(chapters/04b_cyclones_emulator)"),
    ("[`04B/paths.py`](04_NC_Cyclones/04B_Cyclones_emulator/paths.py)", "`04B/paths.py`"),
]

GITHUB_CALLOUT = (
    "> **Jupyter Book:** the same overview is rendered at `docs/intro.html` — "
    "run `bash docs/build.sh` and `bash docs/serve.sh` ([build instructions](docs/README.md)).\n\n"
)
BOOK_HEADER = (
    "<!-- Book intro — prose synced with ../README.md; links target Jupyter Book chapters. -->\n\n"
)
BROWSE_BLOCK = (
    "> **Browse this book:** run `bash docs/build.sh`, then `bash docs/serve.sh` "
    "and open http://localhost:8765/intro.html (SSH tunnel if remote). "
    "See [Build instructions](README).\n\n"
)


def main() -> None:
    text = README.read_text(encoding="utf-8")
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    text = text.replace(GITHUB_CALLOUT, BOOK_HEADER)
    text = text.replace(
        "| [04B — Emulator & splice](chapters/04b_cyclones_emulator) | "
        "Spectra, PTM4 partitions, climate splice |\n\n---\n\n## References",
        "| [04B — Emulator & splice](chapters/04b_cyclones_emulator) | "
        "Spectra, PTM4 partitions, climate splice |\n\n" + BROWSE_BLOCK + "---\n\n## References",
    )
    INTRO.write_text(text, encoding="utf-8")
    print(f"Wrote {INTRO.relative_to(REPO)}")


if __name__ == "__main__":
    main()
