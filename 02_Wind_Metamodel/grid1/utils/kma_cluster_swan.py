"""
Reconstruct hourly SWAN bulk parameters from KMA cluster (BMU) static cases.

For each timestep, look up the nearest 3-hourly BMU assignment, read the
corresponding ``output.mat`` under ``CASES_ONLY_WIND/{bmu:03d}/``, and optionally
combine wind-sea from those runs with BinWaves hindcast bulk (spectrum swell).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import loadmat

MAT_NAMES = ("output.mat", "outputs.mat")
VAR_MAP = {
    "Hsig": "Hsig",
    "TPsmoo": "Tps",
    "Tm02": "Tm02",
    "Dir": "Dir",
    "PkDir": "pdir",
}
PARTITION_IDS = (0, 1, 2, 3)

DEFAULT_CLUSTER_CASES_ROOT = Path("inputs/CASES_ONLY_WIND")
DEFAULT_KMA_BMU_CSV = Path("outputs/KMA/nearest_centroids_idxs_kma_pcs.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/kma_cluster_swan")
BINWAVES_PLUS_KMA_LABEL = "BinWaves + KMA clusters"


def _default_project_root() -> Path:
    """Parent folder of this utils package that owns ``inputs/CASES_ONLY_WIND``."""
    here = Path(__file__).resolve().parent
    for candidate in (here.parent, here.parent.parent):
        if (candidate / "inputs/CASES_ONLY_WIND").is_dir():
            return candidate
    return here.parent


def load_bmu_assignments(csv_path: str | Path) -> pd.Series:
    bmu = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
    return bmu["kma_bmus"].astype(int)


def bmu_for_hourly_times(bmu_series: pd.Series, case_times: pd.DatetimeIndex) -> pd.Series:
    """Map hourly timeline to nearest 3-hourly KMA cluster id."""
    return bmu_series.reindex(pd.DatetimeIndex(case_times), method="nearest").astype(int)


def find_cluster_mat(cluster_root: str | Path, cluster_id: int) -> Path | None:
    case_dir = Path(cluster_root) / f"{int(cluster_id):03d}"
    for name in MAT_NAMES:
        mat_path = case_dir / name
        if mat_path.is_file():
            return mat_path
    return None


def find_reference_cluster_mat(cluster_root: str | Path) -> Path:
    cluster_root = Path(cluster_root)
    for mat_path in sorted(cluster_root.glob("[0-9][0-9][0-9]/output.mat")):
        return mat_path
    for mat_path in sorted(cluster_root.glob("[0-9][0-9][0-9]/outputs.mat")):
        return mat_path
    raise FileNotFoundError(f"No cluster output.mat under {cluster_root}")


def infer_n_clusters(
    cluster_root: str | Path,
    kma_bmu_csv: str | Path | None = None,
) -> int:
    """Number of KMA cluster BMUs (default KMA run uses 250: ids 0..249)."""
    if kma_bmu_csv is not None:
        bmu_csv = Path(kma_bmu_csv)
        if bmu_csv.is_file():
            bmu = load_bmu_assignments(bmu_csv)
            return int(bmu.max()) + 1
    cluster_root = Path(cluster_root)
    cluster_ids = [
        int(entry.name)
        for entry in cluster_root.iterdir()
        if entry.is_dir() and entry.name.isdigit()
    ]
    if cluster_ids:
        return max(cluster_ids) + 1
    return 250


def nearest_grid_indices(
    mat_path: str | Path,
    target_lon: float,
    target_lat: float,
) -> tuple[int, int]:
    data = loadmat(mat_path, squeeze_me=True)
    lon2d = np.asarray(data["Xp"], dtype=float)
    lat2d = np.asarray(data["Yp"], dtype=float)
    dist2d = np.sqrt((lon2d - target_lon) ** 2 + (lat2d - target_lat) ** 2)
    iy, ix = np.unravel_index(np.nanargmin(dist2d), dist2d.shape)
    return int(iy), int(ix)


def _cluster_point_from_mat(mat_path: Path, iy: int, ix: int) -> dict[str, float]:
    data = loadmat(mat_path, squeeze_me=True)
    pt: dict[str, float] = {}
    for mat_key, label in VAR_MAP.items():
        if mat_key in data:
            pt[label] = float(np.asarray(data[mat_key])[iy, ix])
    if "Windv_x" in data and "Windv_y" in data:
        ux = float(np.asarray(data["Windv_x"])[iy, ix])
        vy = float(np.asarray(data["Windv_y"])[iy, ix])
        pt["wind_u"] = ux
        pt["wind_v"] = vy
        pt["wind"] = float(np.hypot(ux, vy))
    return pt


def extract_cluster_bmu_timeseries(
    cluster_root: str | Path,
    case_times: pd.DatetimeIndex,
    bmu_by_time: pd.Series,
    iy: int,
    ix: int,
) -> tuple[pd.DataFrame, int]:
    """Build hourly series from static SWAN runs (one folder per BMU)."""
    labels = list(VAR_MAP.values()) + ["wind", "wind_u", "wind_v", "kma_bmu"]
    bmu_arr = bmu_by_time.astype(int).to_numpy()
    series = {label: np.full(len(case_times), np.nan) for label in labels}
    series["kma_bmu"] = bmu_arr.astype(float)

    cluster_cache: dict[int, dict[str, float] | None] = {}
    n_mat = 0
    for cid in sorted(set(bmu_arr)):
        mat_path = find_cluster_mat(cluster_root, cid)
        if mat_path is None:
            cluster_cache[cid] = None
            continue
        cluster_cache[cid] = _cluster_point_from_mat(mat_path, iy, ix)
        n_mat += 1

    for cid, pt in cluster_cache.items():
        if not pt:
            continue
        mask = bmu_arr == cid
        for key, val in pt.items():
            if key in series:
                series[key][mask] = val

    return pd.DataFrame(series, index=case_times), n_mat


def _partition_hs_list_from_frame(
    spec_df: pd.DataFrame,
    *,
    partition_ids: Sequence[int] = PARTITION_IDS,
) -> list[np.ndarray]:
    return [
        spec_df[f"phs{pid}"].to_numpy(dtype=float)
        for pid in partition_ids
        if f"phs{pid}" in spec_df.columns
    ]


def _partition_dir_list_from_frame(
    spec_df: pd.DataFrame,
    *,
    partition_ids: Sequence[int] = PARTITION_IDS,
) -> list[np.ndarray]:
    return [
        spec_df[f"dp{pid}"].to_numpy(dtype=float)
        for pid in partition_ids
        if f"phs{pid}" in spec_df.columns and f"dp{pid}" in spec_df.columns
    ]


def _partition_period_list_from_frame(
    spec_df: pd.DataFrame,
    *,
    partition_ids: Sequence[int] = PARTITION_IDS,
) -> list[np.ndarray]:
    return [
        spec_df[f"ptp{pid}"].to_numpy(dtype=float)
        for pid in partition_ids
        if f"phs{pid}" in spec_df.columns and f"ptp{pid}" in spec_df.columns
    ]


def _has_partition_columns(spec_df: pd.DataFrame) -> bool:
    return any(f"phs{pid}" in spec_df.columns for pid in PARTITION_IDS)


def _mean_direction_from_hs_components(
    hs_components: Sequence[np.ndarray],
    dir_components: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Circular mean direction: atan2(Σ Hᵢ² sin θᵢ, Σ Hᵢ² cos θᵢ).

    ``hs_components`` and ``dir_components`` are equal-length sequences of arrays
    with identical broadcast shape.
    """
    if not hs_components:
        return np.full((), np.nan)

    hs = np.stack([np.asarray(h, dtype=float) for h in hs_components], axis=-1)
    dirs = np.stack([np.asarray(d, dtype=float) for d in dir_components], axis=-1)
    energy = np.where(np.isfinite(hs), hs ** 2, 0.0)
    valid = (energy > 0) & np.isfinite(dirs)
    sin_sum = np.sum(np.where(valid, energy * np.sin(np.deg2rad(dirs)), 0.0), axis=-1)
    cos_sum = np.sum(np.where(valid, energy * np.cos(np.deg2rad(dirs)), 0.0), axis=-1)
    weight = np.sum(np.where(valid, energy, 0.0), axis=-1)
    combined_dir = np.full(hs.shape[:-1], np.nan, dtype=float)
    mask = weight > 0
    combined_dir[mask] = np.rad2deg(np.arctan2(sin_sum[mask], cos_sum[mask])) % 360.0
    return combined_dir


def _harmonic_mean_period_from_hs_components(
    hs_components: Sequence[np.ndarray],
    period_components: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Energy-weighted harmonic mean period: T = Σ Hᵢ² / Σ(Hᵢ² / Tᵢ).
    """
    if not hs_components:
        return np.full((), np.nan)

    hs = np.stack([np.asarray(h, dtype=float) for h in hs_components], axis=-1)
    periods = np.stack([np.asarray(t, dtype=float) for t in period_components], axis=-1)
    energy = np.where(np.isfinite(hs), hs ** 2, 0.0)
    valid = (energy > 0) & np.isfinite(periods) & (periods > 0)
    energy_sum = np.sum(np.where(valid, energy, 0.0), axis=-1)
    combined_tm = np.full(hs.shape[:-1], np.nan, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        inv_t_sum = np.sum(np.where(valid, energy / periods, 0.0), axis=-1)
        mask = (energy_sum > 0) & (inv_t_sum > 0)
        combined_tm[mask] = energy_sum[mask] / inv_t_sum[mask]
    return combined_tm


def _combined_hs_from_components(hs_components: Sequence[np.ndarray]) -> np.ndarray:
    if not hs_components:
        return np.full((), np.nan)
    energy = np.zeros_like(np.asarray(hs_components[0], dtype=float))
    for hs in hs_components:
        h = np.asarray(hs, dtype=float)
        energy = energy + np.where(np.isfinite(h), h ** 2, 0.0)
    combined_hs = np.sqrt(np.maximum(energy, 0.0))
    return np.where(energy > 0, combined_hs, np.nan)


def _peak_tp_dp_from_bulk_components(
    e_w: np.ndarray,
    e_s: np.ndarray,
    tp_w: np.ndarray,
    tp_s: np.ndarray,
    dp_w: np.ndarray,
    dp_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Tp and peak direction from the higher bulk-energy component (BMU vs BinWaves)."""
    wind_wins = (e_w >= e_s) & (e_w > 0)
    spec_wins = (e_s > e_w) & (e_s > 0)
    combined_tp = np.full(np.broadcast(e_w, e_s).shape, np.nan, dtype=float)
    combined_dp = np.full(np.broadcast(e_w, e_s).shape, np.nan, dtype=float)
    combined_tp[wind_wins] = tp_w[wind_wins]
    combined_tp[spec_wins] = tp_s[spec_wins]
    combined_dp[wind_wins] = dp_w[wind_wins]
    combined_dp[spec_wins] = dp_s[spec_wins]
    return combined_tp, combined_dp


def _peak_tp_dp_from_partitions(
    phs_parts: Sequence[np.ndarray],
    ptp_parts: Sequence[np.ndarray],
    pdir_parts: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Tp and peak direction from the partition with largest Hs²."""
    hs = np.stack([np.asarray(h, dtype=float) for h in phs_parts], axis=-1)
    ptp = np.stack([np.asarray(t, dtype=float) for t in ptp_parts], axis=-1)
    pdir = np.stack([np.asarray(d, dtype=float) for d in pdir_parts], axis=-1)
    energy = np.where(np.isfinite(hs), hs ** 2, 0.0)
    has_energy = energy.sum(axis=-1) > 0
    idx = np.argmax(energy, axis=-1, keepdims=True)
    combined_tp = np.take_along_axis(ptp, idx, axis=-1).squeeze(-1)
    combined_dp = np.take_along_axis(pdir, idx, axis=-1).squeeze(-1)
    combined_tp = np.where(has_energy, combined_tp, np.nan)
    combined_dp = np.where(has_energy, combined_dp, np.nan)
    return combined_tp, combined_dp


def combine_partition0_with_bmu(
    phs0: np.ndarray,
    ptp0: np.ndarray,
    dp0: np.ndarray,
    hs_bmu: np.ndarray,
    tp_bmu: np.ndarray,
    dp_bmu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge BMU wind-sea into BinWaves partition 0.

    ``phs0_new = sqrt(phs0² + Hsig_bmu²)``; ``ptp0``/``dp0`` follow the higher-energy
    component (BMU vs ShoreShop phs0).
    """
    phs0 = np.asarray(phs0, dtype=float)
    hs_bmu = np.asarray(hs_bmu, dtype=float)
    phs0_new = _combined_hs_from_components([phs0, hs_bmu])
    e_bmu = np.where(np.isfinite(hs_bmu), hs_bmu ** 2, 0.0)
    e_phs0 = np.where(np.isfinite(phs0), phs0 ** 2, 0.0)
    ptp0_new, dp0_new = _peak_tp_dp_from_bulk_components(
        e_bmu,
        e_phs0,
        np.asarray(tp_bmu, dtype=float),
        np.asarray(ptp0, dtype=float),
        np.asarray(dp_bmu, dtype=float),
        np.asarray(dp0, dtype=float),
    )
    return phs0_new, ptp0_new, dp0_new


def bulk_waves_from_partitions(
    phs_parts: Sequence[np.ndarray],
    ptp_parts: Sequence[np.ndarray],
    pdir_parts: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bulk hs/tp/dp/dm/tm02 from partition fields 0–3.

    Hs uses ``sqrt(Σ phsᵢ²)``; peak Tp/dp from the dominant partition; mean direction
    and Tm02 use the existing partition-weighted formulas over all four partitions.
    """
    combined_hs = _combined_hs_from_components(phs_parts)
    combined_tp, combined_dp = _peak_tp_dp_from_partitions(phs_parts, ptp_parts, pdir_parts)
    combined_dm = _mean_direction_from_hs_components(phs_parts, pdir_parts)
    combined_tm = _harmonic_mean_period_from_hs_components(phs_parts, ptp_parts)
    return combined_hs, combined_tp, combined_dp, combined_dm, combined_tm


def combine_partitions_with_bmu_arrays(
    hs_bmu: np.ndarray,
    tp_bmu: np.ndarray,
    dp_bmu: np.ndarray,
    *,
    phs_bw: Sequence[np.ndarray],
    ptp_bw: Sequence[np.ndarray],
    pdir_bw: Sequence[np.ndarray],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Partition-first BinWaves + BMU combine.

    Returns ``(phs0_new, ptp0_new, dp0_new, hs, tp, dp, dm, tm02)``.
    BMU is merged only into partition 0; partitions 1–3 stay unchanged; bulk fields
    are derived from all four partitions.
    """
    if len(phs_bw) < 1:
        raise ValueError("phs_bw must include partition 0")
    phs0_new, ptp0_new, dp0_new = combine_partition0_with_bmu(
        phs_bw[0],
        ptp_bw[0],
        pdir_bw[0],
        hs_bmu,
        tp_bmu,
        dp_bmu,
    )
    phs_all = [phs0_new, *phs_bw[1:]]
    ptp_all = [ptp0_new, *ptp_bw[1:]]
    pdir_all = [dp0_new, *pdir_bw[1:]]
    hs, tp, dp, dm, tm02 = bulk_waves_from_partitions(phs_all, ptp_all, pdir_all)
    return phs0_new, ptp0_new, dp0_new, hs, tp, dp, dm, tm02


def combine_wind_and_spectrum(wind_df: pd.DataFrame, spec_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine BMU wind-sea with BinWaves swell partitions.

    Hs uses all partition heights with BMU merged into phs0. Bulk Tp/pdir/dm/Tm02 are
    derived from partitions 0–3 after that merge.
    """
    hs_w = wind_df["Hsig"].to_numpy(dtype=float)
    hs_s = spec_df["Hsig"].to_numpy(dtype=float)
    e_w = np.where(np.isfinite(hs_w), hs_w ** 2, 0.0)
    e_s = np.where(np.isfinite(hs_s), hs_s ** 2, 0.0)
    tp_w = wind_df["Tps"].to_numpy(dtype=float)
    tp_s = spec_df["Tps"].to_numpy(dtype=float)
    pd_w = wind_df["pdir"].to_numpy(dtype=float)
    pd_s = spec_df["pdir"].to_numpy(dtype=float)
    dir_w = wind_df["Dir"].to_numpy(dtype=float)
    tm_w = wind_df["Tm02"].to_numpy(dtype=float)

    combined_tp, combined_pdir = _peak_tp_dp_from_bulk_components(
        e_w, e_s, tp_w, tp_s, pd_w, pd_s
    )

    if _has_partition_columns(spec_df):
        phs_parts = _partition_hs_list_from_frame(spec_df)
        dir_parts = _partition_dir_list_from_frame(spec_df)
        period_parts = _partition_period_list_from_frame(spec_df)
        phs0_new, ptp0_new, dp0_new = combine_partition0_with_bmu(
            phs_parts[0],
            period_parts[0],
            dir_parts[0],
            hs_w,
            tp_w,
            pd_w,
        )
        phs_all = [phs0_new, *phs_parts[1:]]
        ptp_all = [ptp0_new, *period_parts[1:]]
        pdir_all = [dp0_new, *dir_parts[1:]]
        combined_hs, combined_tp, combined_pdir, combined_dir, combined_tm = bulk_waves_from_partitions(
            phs_all, ptp_all, pdir_all
        )
    else:
        combined_hs = np.sqrt(np.maximum(e_w, 0.0) + np.maximum(e_s, 0.0))
        combined_hs = np.where((e_w > 0) | (e_s > 0), combined_hs, np.nan)
        dir_s = spec_df["Dir"].to_numpy(dtype=float) if "Dir" in spec_df.columns else pd_s
        tm_s = spec_df["Tm02"].to_numpy(dtype=float) if "Tm02" in spec_df.columns else tp_s
        combined_dir = _mean_direction_from_hs_components([hs_w, hs_s], [dir_w, dir_s])
        combined_tm = _harmonic_mean_period_from_hs_components([hs_w, hs_s], [tm_w, tm_s])

    return pd.DataFrame(
        {
            "Hsig": combined_hs,
            "Tps": combined_tp,
            "Tm02": combined_tm,
            "pdir": combined_pdir,
            "Dir": combined_dir,
        },
        index=wind_df.index,
    )


def combine_wind_and_spectrum_arrays(
    hs_w: np.ndarray,
    hs_s: np.ndarray,
    tp_w: np.ndarray,
    tp_s: np.ndarray,
    dp_w: np.ndarray,
    dp_s: np.ndarray,
    dm_w: np.ndarray | None = None,
    dm_s: np.ndarray | None = None,
    *,
    phs_bw: Sequence[np.ndarray] | None = None,
    pdir_bw: Sequence[np.ndarray] | None = None,
    ptp_bw: Sequence[np.ndarray] | None = None,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """
    Vectorized combine for arrays shaped ``(..., site)`` or ``(time, site)``.

    When partition arrays are given, BMU is merged into phs0 and bulk fields are
    derived from partitions 0–3 (returns hs, tp, dp, dm, tm02).
    """
    hs_w = np.asarray(hs_w, dtype=float)
    hs_s = np.asarray(hs_s, dtype=float)
    tp_w = np.asarray(tp_w, dtype=float)
    tp_s = np.asarray(tp_s, dtype=float)
    dp_w = np.asarray(dp_w, dtype=float)
    dp_s = np.asarray(dp_s, dtype=float)

    e_w = np.where(np.isfinite(hs_w), hs_w ** 2, 0.0)
    e_s = np.where(np.isfinite(hs_s), hs_s ** 2, 0.0)

    combined_tp, combined_dp = _peak_tp_dp_from_bulk_components(
        e_w, e_s, tp_w, tp_s, dp_w, dp_s
    )

    use_partitions = phs_bw is not None and pdir_bw is not None and ptp_bw is not None
    if use_partitions:
        _, _, _, combined_hs, combined_tp, combined_dp, combined_dm, combined_tm = (
            combine_partitions_with_bmu_arrays(
                hs_w,
                tp_w,
                dp_w,
                phs_bw=phs_bw,
                ptp_bw=ptp_bw,
                pdir_bw=pdir_bw,
            )
        )
        return combined_hs, combined_tp, combined_dp, combined_dm, combined_tm

    combined_hs = np.sqrt(np.maximum(e_w, 0.0) + np.maximum(e_s, 0.0))
    combined_hs = np.where((e_w > 0) | (e_s > 0), combined_hs, np.nan)

    if dm_w is None or dm_s is None:
        return combined_hs, combined_tp, combined_dp

    combined_dm = _mean_direction_from_hs_components(
        [hs_w, hs_s],
        [np.asarray(dm_w, dtype=float), np.asarray(dm_s, dtype=float)],
    )
    return combined_hs, combined_tp, combined_dp, combined_dm


def combine_wind_and_spectrum_arrays_full(
    hs_w: np.ndarray,
    hs_s: np.ndarray,
    tp_w: np.ndarray,
    tp_s: np.ndarray,
    dp_w: np.ndarray,
    dp_s: np.ndarray,
    dir_w: np.ndarray,
    tm_w: np.ndarray,
    *,
    phs_bw: Sequence[np.ndarray],
    pdir_bw: Sequence[np.ndarray],
    ptp_bw: Sequence[np.ndarray],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Partition-first combine: BMU merged into phs0, bulk from partitions 0–3.

    Returns ``(phs0_new, ptp0_new, dp0_new, hs, tp, dp, dm, tm02)``.
    ``hs_s``/``tp_s``/``dp_s`` are ignored when partitions are supplied (kept for API
    compatibility).
    """
    _ = (hs_s, tp_s, dp_s, dir_w, tm_w)
    return combine_partitions_with_bmu_arrays(
        hs_w,
        tp_w,
        dp_w,
        phs_bw=phs_bw,
        ptp_bw=ptp_bw,
        pdir_bw=pdir_bw,
    )


# Public aliases for notebooks / spatial grids (same partition-weighted formulas).
mean_direction_from_hs_components = _mean_direction_from_hs_components
harmonic_mean_period_from_hs_components = _harmonic_mean_period_from_hs_components
combined_hs_from_components = _combined_hs_from_components


def combine_binwaves_partition_fields(
    phs_parts: Sequence[np.ndarray],
    pdir_parts: Sequence[np.ndarray],
    ptp_parts: Sequence[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Bulk-free BinWaves partition combine: hs, mean direction, optional Tm02."""
    hs = combined_hs_from_components(phs_parts)
    dm = mean_direction_from_hs_components(phs_parts, pdir_parts)
    tm = (
        harmonic_mean_period_from_hs_components(phs_parts, ptp_parts)
        if ptp_parts is not None
        else None
    )
    return hs, dm, tm


def _normalize_coordinate(coordinate: Sequence[float]) -> tuple[float, float, float, float]:
    from utils.gcm_comparison import _lat_lon_from_pair

    lat, lon = _lat_lon_from_pair(coordinate)
    return lat, lon, lon, lat


def _swan_point_filename(lat: float, lon: float, *, suffix: str = "") -> str:
    tag = f"_kma{suffix}" if suffix else "_kma"
    return f"swan_point_timeseries_lon{lon:.3f}_lat{lat:.3f}{tag}.nc"


def _load_binwaves_bulk_dataframe(
    lat: float,
    lon: float,
    case_times: pd.DatetimeIndex,
    *,
    historical_folder: str | Path,
    project_root: str | Path | None,
) -> pd.DataFrame:
    from utils.gcm_comparison import load_timeseries_at_coordinate, resolve_historical_folder

    hist_folder = resolve_historical_folder(
        historical_folder=historical_folder,
        project_root=project_root,
    )
    hs_path = Path(hist_folder) / "hs_merged_all.nc"
    tp_path = Path(hist_folder) / "tp_merged_all.nc"
    dp_path = Path(hist_folder) / "dp_merged_all.nc"

    load_kw = dict(
        aggregation="native",
        time_start=case_times[0],
        time_end=case_times[-1],
        project_root=project_root,
    )
    hs_da, _ = load_timeseries_at_coordinate(hs_path, (lat, lon), variable="hs", **load_kw)
    tp_da, _ = load_timeseries_at_coordinate(tp_path, (lat, lon), variable="tp", **load_kw)
    dp_da, _ = load_timeseries_at_coordinate(dp_path, (lat, lon), variable="dp", **load_kw)
    aligned = [hs_da, tp_da, dp_da]
    partition_arrays: dict[str, xr.DataArray] = {}
    for pid in PARTITION_IDS:
        phs_path = Path(hist_folder) / f"phs{pid}_merged_all.nc"
        ptp_path = Path(hist_folder) / f"ptp{pid}_merged_all.nc"
        dp_part_path = Path(hist_folder) / f"dp{pid}_merged_all.nc"
        if not (phs_path.is_file() and ptp_path.is_file() and dp_part_path.is_file()):
            continue
        phs_da, _ = load_timeseries_at_coordinate(
            phs_path, (lat, lon), variable=f"phs{pid}", **load_kw
        )
        ptp_da, _ = load_timeseries_at_coordinate(
            ptp_path, (lat, lon), variable=f"ptp{pid}", **load_kw
        )
        dp_part_da, _ = load_timeseries_at_coordinate(
            dp_part_path, (lat, lon), variable=f"dp{pid}", **load_kw
        )
        aligned.extend([phs_da, ptp_da, dp_part_da])
        partition_arrays[f"phs{pid}"] = phs_da
        partition_arrays[f"ptp{pid}"] = ptp_da
        partition_arrays[f"dp{pid}"] = dp_part_da

    aligned_data = xr.align(*aligned, join="inner")
    hs_a, tp_a, dp_a = aligned_data[:3]
    data = {
        "Hsig": hs_a.values,
        "Tps": tp_a.values,
        "Tm02": tp_a.values,
        "pdir": dp_a.values,
        "Dir": dp_a.values,
    }
    offset = 3
    for pid in PARTITION_IDS:
        key = f"phs{pid}"
        if key not in partition_arrays:
            continue
        phs_a, ptp_a, dp_part_a = aligned_data[offset : offset + 3]
        offset += 3
        data[f"phs{pid}"] = phs_a.values
        data[f"ptp{pid}"] = ptp_a.values
        data[f"dp{pid}"] = dp_part_a.values
    df = pd.DataFrame(data, index=pd.DatetimeIndex(hs_a["time"].values))
    return df.reindex(case_times, method="nearest")


def build_kma_cluster_swan_dataframe(
    coordinate: Sequence[float],
    *,
    time_start,
    time_end,
    cluster_cases_root: str | Path = DEFAULT_CLUSTER_CASES_ROOT,
    kma_bmu_csv: str | Path = DEFAULT_KMA_BMU_CSV,
    historical_folder: str | Path | None = None,
    combine_with_binwaves: bool = True,
    project_root: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Reconstruct hourly SWAN bulk at ``coordinate`` from KMA cluster cases.

  Returns ``(dataframe, info)`` with columns ``Hsig``, ``Tps``, ``pdir`` (and
    ``kma_bmu`` when not combined). When ``combine_with_binwaves=True``, returns
    BinWaves + KMA cluster combined bulk parameters.
    """
    from utils.gcm_comparison import resolve_path

    if project_root is None:
        project_root = _default_project_root()
    lat, lon, target_lon, target_lat = _normalize_coordinate(coordinate)
    cluster_root = resolve_path(cluster_cases_root, project_root)
    bmu_csv = resolve_path(kma_bmu_csv, project_root)

    case_times = pd.date_range(time_start, time_end, freq="1h")
    bmu_series = load_bmu_assignments(bmu_csv)
    bmu_by_time = bmu_for_hourly_times(bmu_series, case_times)

    ref_mat = find_reference_cluster_mat(cluster_root)
    iy, ix = nearest_grid_indices(ref_mat, target_lon, target_lat)
    cluster_df, n_mat = extract_cluster_bmu_timeseries(
        cluster_root, case_times, bmu_by_time, iy, ix
    )

    unique_bmus = sorted(int(v) for v in bmu_by_time.unique() if np.isfinite(v))
    missing_bmus = sorted(
        b for b in unique_bmus if find_cluster_mat(cluster_root, b) is None
    )
    if missing_bmus:
        print(
            f"WARN KMA clusters: missing SWAN output for BMUs "
            f"{missing_bmus[:20]}{'...' if len(missing_bmus) > 20 else ''}"
        )

    if combine_with_binwaves:
        if historical_folder is None:
            from utils.gcm_comparison import resolve_historical_folder

            historical_folder = resolve_historical_folder(project_root=project_root)
        binwaves_df = _load_binwaves_bulk_dataframe(
            lat, lon, case_times,
            historical_folder=historical_folder,
            project_root=project_root,
        )
        out_df = combine_wind_and_spectrum(cluster_df, binwaves_df)
        out_df["kma_bmu"] = cluster_df["kma_bmu"].values
    else:
        out_df = cluster_df

    info = {
        "lat": lat,
        "lon": lon,
        "target_lon": target_lon,
        "target_lat": target_lat,
        "grid_ij": (iy, ix),
        "reference_mat": str(ref_mat),
        "n_hours": len(case_times),
        "n_unique_bmus": len(unique_bmus),
        "n_cluster_mats_loaded": n_mat,
        "missing_bmus": missing_bmus,
        "combine_with_binwaves": combine_with_binwaves,
        "unique_bmus": unique_bmus,
    }
    print(
        f"KMA cluster SWAN @ ({lat:.3f}, {lon:.3f}): "
        f"{len(case_times)} hours, {len(unique_bmus)} BMUs, "
        f"{n_mat} cluster mats loaded"
        + (" (combined with BinWaves)" if combine_with_binwaves else "")
    )
    return out_df, info


def write_swan_point_netcdf(
    df: pd.DataFrame,
    nc_path: str | Path,
    *,
    lat: float,
    lon: float,
) -> Path:
    nc_path = Path(nc_path)
    nc_path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {
            "Hsig": ("time", df["Hsig"].to_numpy(dtype=float)),
            "Tps": ("time", df["Tps"].to_numpy(dtype=float)),
            "pdir": ("time", df["pdir"].to_numpy(dtype=float)),
        },
        coords={"time": pd.DatetimeIndex(df.index)},
        attrs={"latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(nc_path)
    return nc_path


def build_binwaves_plus_kma_swan_netcdf(
    coordinate: Sequence[float],
    *,
    time_start,
    time_end,
    cluster_cases_root: str | Path = DEFAULT_CLUSTER_CASES_ROOT,
    kma_bmu_csv: str | Path = DEFAULT_KMA_BMU_CSV,
    historical_folder: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    project_root: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Build (or reuse) a point NetCDF with BinWaves + KMA cluster combined bulk."""
    from utils.gcm_comparison import resolve_path

    if project_root is None:
        project_root = _default_project_root()
    lat, lon, _, _ = _normalize_coordinate(coordinate)
    out_dir = resolve_path(output_dir, project_root)
    t0 = pd.Timestamp(time_start).strftime("%Y%m%d%H")
    t1 = pd.Timestamp(time_end).strftime("%Y%m%d%H")
    nc_path = out_dir / _swan_point_filename(lat, lon, suffix=f"_{t0}_{t1}")

    if nc_path.is_file() and not overwrite:
        print(f"Reusing KMA cluster SWAN NetCDF: {nc_path}")
        return nc_path

    df, _info = build_kma_cluster_swan_dataframe(
        coordinate,
        time_start=time_start,
        time_end=time_end,
        cluster_cases_root=cluster_cases_root,
        kma_bmu_csv=kma_bmu_csv,
        historical_folder=historical_folder,
        combine_with_binwaves=True,
        project_root=project_root,
    )
    return write_swan_point_netcdf(df, nc_path, lat=lat, lon=lon)


def build_kma_combined_wave_dataarrays(
    coordinate: Sequence[float],
    *,
    time_start,
    time_end,
    aggregation: str = "native",
    cluster_cases_root: str | Path = DEFAULT_CLUSTER_CASES_ROOT,
    kma_bmu_csv: str | Path = DEFAULT_KMA_BMU_CSV,
    historical_folder: str | Path | None = None,
    combine_with_binwaves: bool = True,
    project_root: str | Path | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, dict]:
    """Return aggregated Hs/Tp/Dp DataArrays from KMA cluster SWAN (+ optional BinWaves)."""
    from utils.gcm_comparison import apply_aggregation

    df, info = build_kma_cluster_swan_dataframe(
        coordinate,
        time_start=time_start,
        time_end=time_end,
        cluster_cases_root=cluster_cases_root,
        kma_bmu_csv=kma_bmu_csv,
        historical_folder=historical_folder,
        combine_with_binwaves=combine_with_binwaves,
        project_root=project_root,
    )
    time_index = pd.DatetimeIndex(df.index)
    hs = xr.DataArray(df["Hsig"].to_numpy(dtype=float), coords={"time": time_index}, dims=["time"])
    tp = xr.DataArray(df["Tps"].to_numpy(dtype=float), coords={"time": time_index}, dims=["time"])
    dp = xr.DataArray(df["pdir"].to_numpy(dtype=float), coords={"time": time_index}, dims=["time"])

    hs = apply_aggregation(hs.rename("hs"), "hs", aggregation)
    tp = apply_aggregation(tp.rename("tp"), "tp", aggregation)
    dp = apply_aggregation(dp.rename("dp"), "dp", aggregation)
    hs_a, tp_a, dp_a = xr.align(hs, tp, dp, join="inner")
    info["aggregation"] = str(aggregation).lower()
    info["n_points"] = int(hs_a.sizes.get("time", hs_a.size))
    return hs_a, tp_a, dp_a, info
