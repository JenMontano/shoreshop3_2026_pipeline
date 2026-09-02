"""Upstream paths for 04B — read directly from 04A / 03B outputs (no copies in inputs/)."""

from __future__ import annotations

from pathlib import Path

_04B = Path(__file__).resolve().parent
_04 = _04B.parent
_REPO = _04.parent

DIR_04A = _04 / "04A_Cyclones_Metamodel"
DIR_03B = _REPO / "03_NC_Wave_Emulator" / "03B_Century_GCMs_Waves"
DIR_01_BINWAVES = _REPO / "01_BinWaves"
DIR_02_WIND = _REPO / "02_Wind_Metamodel"

# Local 04B static inputs (geometry only — not duplicated upstream products)
DIR_INPUTS = _04B / "inputs"
GEBCO_FILE = DIR_01_BINWAVES / "inputs" / "gebco_bathymetry.nc"
ISOBATH_GEOJSON = DIR_INPUTS / "isobath_10m_points_500m.geojson"


def predicted_emu_tracks(ssp: str = "ssp245", *, mf: bool = False) -> Path:
    """04A metamodel wave fields per emulated cyclone (historical / ssp245 / ssp585)."""
    tag = {"245": "ssp245", "585": "ssp585", "ssp245": "ssp245", "ssp585": "ssp585"}[ssp]
    suffix = "_mf" if mf else ""
    return (
        DIR_04A
        / "outputs"
        / f"predicted_emu_tracks_ec_earth3_veg_lr_{tag}_corrected{suffix}.nc"
    )


def earth3_baseline_dir(ssp: str = "ssp245") -> Path:
    """03B ALR century-scale 500 m fields without cyclones."""
    tag = {"245": "ssp245", "585": "ssp585", "ssp245": "ssp245", "ssp585": "ssp585"}[ssp]
    return DIR_03B / "outputs" / f"earth3_veg_{tag}"
