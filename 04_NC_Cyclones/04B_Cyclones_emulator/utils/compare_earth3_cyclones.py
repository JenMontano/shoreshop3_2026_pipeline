"""Compare Original earth3_veg_lr vs earth3_veg_lr+cyclones.

Produces diagnostic figure types:
  1) stacked density histograms (HS, TP, DP) + exceedance
  2) side-by-side Q-Q plots
  3) monthly seasonal cycle

Supports SSP245 / SSP585 pair figures and a 4-scenario overlay
(original±cyclones for both SSPs) with a consistent color scheme:
  SSP245 original  → dark blue
  SSP245 +cyclones → light blue
  SSP585 original  → dark orange
  SSP585 +cyclones → gold / yellow
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_PROJECT = Path(__file__).resolve().parents[1]

import sys

if str(DEFAULT_PROJECT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_PROJECT))
from paths import earth3_baseline_dir  # noqa: E402

DEFAULT_ORIGINAL_DIR = earth3_baseline_dir("ssp245")
DEFAULT_WITH_CYC_DIR = DEFAULT_PROJECT / "outputs"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HISTORICAL_DIR = REPO_ROOT / "02_Wind_Metamodel/outputs/merged_500m_binwaves_bmus"
DEFAULT_BINWAVES_DIR = REPO_ROOT / "01_BinWaves/outputs/merged_500m"
DEFAULT_TARGET_LON = -75.7316
DEFAULT_TARGET_LAT = 36.1942

LABEL_ORIGINAL = "earth3_veg_lr SSP245"
LABEL_WITH_CYC = "earth3_veg_lr SSP245 + cyclones"
LABEL_HISTORICAL = "BinWaves+BMUS (1980-2023)"
LABEL_BINWAVES = "BinWaves"

# Scenario color palette (paired by SSP)
COLORS = {
    "historical": "#000000",        # black
    "ssp245_original": "#08306b",   # dark blue
    "ssp245_cyclones": "#6baed6",   # light blue
    "ssp585_original": "#ff4500",   # orangered
    "ssp585_cyclones": "#fec44f",   # gold / yellow
}

VAR_META = {
    "hs": {"file": "hs_500m.nc", "with_file": "hs_500m_with_cyclones.nc", "var": "hs", "ylabel": "Hs (m)", "title": "HS"},
    "tp": {"file": "tp_500m.nc", "with_file": "tp_500m_with_cyclones.nc", "var": "tp", "ylabel": "Tp (s)", "title": "TP"},
    "dp": {"file": "dp_500m.nc", "with_file": "dp_500m_with_cyclones.nc", "var": "dp", "ylabel": "Dp (deg)", "title": "DP"},
}


def nearest_site_index(lon: np.ndarray, lat: np.ndarray, target_lon: float, target_lat: float) -> tuple[int, float]:
    d = (lon.astype(np.float64) - target_lon) ** 2 + (lat.astype(np.float64) - target_lat) ** 2
    i = int(np.argmin(d))
    return i, float(np.sqrt(d[i]))


def load_site_series(
    nc_path: Path,
    var_name: str,
    *,
    target_lon: float = DEFAULT_TARGET_LON,
    target_lat: float = DEFAULT_TARGET_LAT,
    site_index: int | None = None,
) -> tuple[pd.DatetimeIndex, np.ndarray, int]:
    """Load one-site time series from a daily 500 m NetCDF."""
    with xr.open_dataset(nc_path) as ds:
        if var_name not in ds:
            candidates = [v for v in ds.data_vars if v.lower() == var_name.lower()]
            if not candidates:
                raise KeyError(f"{var_name} not in {nc_path}")
            var_name = candidates[0]
        lon = np.asarray(ds.lon.values)
        lat = np.asarray(ds.lat.values)
        if site_index is None:
            site_index, _ = nearest_site_index(lon, lat, target_lon, target_lat)
        times = pd.DatetimeIndex(ds.time.values)
        vals = np.asarray(ds[var_name].isel(site=site_index).values, dtype=np.float64)
    return times, vals, site_index


def load_historical_daily(
    variables: tuple[str, ...] = ("hs", "tp", "dp"),
    *,
    historical_dir: Path = DEFAULT_HISTORICAL_DIR,
    target_lon: float = DEFAULT_TARGET_LON,
    target_lat: float = DEFAULT_TARGET_LAT,
) -> dict[str, dict]:
    """Load historical BinWaves+BMUS series and aggregate to daily means.

    Hourly → daily **mean** for hs/tp/dp (matches earth3_veg daily climate
    series and typical seasonality plots). Using daily *max* for hs inflates
    monthly means by ~0.3 m (e.g. Jan 1.12 → 1.47).
    """
    historical_dir = Path(historical_dir)
    hs_path = historical_dir / "hs_500m.nc"
    if not hs_path.is_file():
        raise FileNotFoundError(hs_path)

    with xr.open_dataset(hs_path) as ds:
        lon = np.asarray(ds.lon.values)
        lat = np.asarray(ds.lat.values)
        site_index, dist = nearest_site_index(lon, lat, target_lon, target_lat)
        times = pd.DatetimeIndex(ds.time.values)
        hs = np.asarray(ds.hs.isel(site=site_index).values, dtype=np.float64)
        print(
            f"Historical site {site_index} "
            f"({float(lon[site_index]):.5f}, {float(lat[site_index]):.5f}) "
            f"dist={dist:.2e} deg  n={len(times)}"
        )

    df = pd.DataFrame({"hs": hs}, index=times)
    for key in variables:
        if key == "hs":
            continue
        path = historical_dir / VAR_META[key]["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        t, y, _ = load_site_series(
            path, VAR_META[key]["var"], target_lon=target_lon, target_lat=target_lat, site_index=site_index
        )
        df[key] = pd.Series(y, index=t)

    # Daily means (not daily max) so seasonality matches earth3_veg / user plots
    daily = df.resample("1D").mean()

    out: dict[str, dict] = {}
    for key in variables:
        vals = daily[key].to_numpy(dtype=np.float64)
        out[key] = {
            "time": pd.DatetimeIndex(daily.index),
            "values": vals,
            "site_index": site_index,
            "title": VAR_META[key]["title"],
            "ylabel": VAR_META[key]["ylabel"],
        }
        print(
            f"  historical daily-mean {key}: {np.isfinite(vals).sum()} days "
            f"({out[key]['time'][0].date()} → {out[key]['time'][-1].date()})"
        )
    return out


def climatological_monthly_mean(times: pd.DatetimeIndex, values: np.ndarray) -> np.ndarray:
    """Plain climatological monthly mean (mean of all timesteps in each calendar month)."""
    months = pd.DatetimeIndex(times).month
    vals = np.asarray(values, dtype=np.float64)
    return np.array([np.nanmean(vals[months == m]) for m in range(1, 13)])


def climatological_monthly_max(times: pd.DatetimeIndex, values: np.ndarray) -> np.ndarray:
    """Climatological monthly maximum (max of all timesteps in each calendar month)."""
    months = pd.DatetimeIndex(times).month
    vals = np.asarray(values, dtype=np.float64)
    return np.array([np.nanmax(vals[months == m]) for m in range(1, 13)])


def attach_historical(
    data: dict[str, dict],
    historical: dict[str, dict],
) -> dict[str, dict]:
    """Attach historical daily series onto pair or multi-scenario data dicts."""
    for key, h in historical.items():
        if key not in data:
            continue
        data[key]["historical"] = h["values"]
        data[key]["historical_time"] = h["time"]
    return data


def restrict_to_overlapping_period(
    data: dict[str, dict],
    *,
    include_historical: bool = True,
) -> tuple[dict[str, dict], pd.Timestamp, pd.Timestamp]:
    """Keep only timesteps in the common calendar window across all series.

    Returns (filtered_data, overlap_start, overlap_end).
    """
    starts, ends = [], []
    for d in data.values():
        t = pd.DatetimeIndex(d["time"])
        starts.append(t.min())
        ends.append(t.max())
        if include_historical and "historical_time" in d:
            ht = pd.DatetimeIndex(d["historical_time"])
            starts.append(ht.min())
            ends.append(ht.max())
    t0, t1 = max(starts), min(ends)
    if t0 > t1:
        raise ValueError(f"No overlapping period: start={t0}, end={t1}")

    out: dict[str, dict] = {}
    for key, d in data.items():
        nd = dict(d)
        t = pd.DatetimeIndex(d["time"])
        m = (t >= t0) & (t <= t1)
        nd["time"] = t[m]
        if "series" in d:
            nd["series"] = {sk: np.asarray(v)[m] for sk, v in d["series"].items()}
        if "original" in d:
            nd["original"] = np.asarray(d["original"])[m]
        if "with_cyclones" in d:
            nd["with_cyclones"] = np.asarray(d["with_cyclones"])[m]
        if include_historical and "historical_time" in d:
            ht = pd.DatetimeIndex(d["historical_time"])
            hm = (ht >= t0) & (ht <= t1)
            nd["historical_time"] = ht[hm]
            nd["historical"] = np.asarray(d["historical"])[hm]
        out[key] = nd
    print(f"Overlapping period: {t0.date()} → {t1.date()}")
    return out, t0, t1


def _with_file_name(key: str, with_suffix: str = "") -> str:
    base = VAR_META[key]["with_file"]
    if not with_suffix:
        return base
    # hs_500m_with_cyclones.nc -> hs_500m_with_cyclones_ssp585.nc
    return base.replace(".nc", f"{with_suffix}.nc")


def load_original_and_with_cyclones(
    variables: tuple[str, ...] = ("hs", "tp", "dp"),
    *,
    original_dir: Path = DEFAULT_ORIGINAL_DIR,
    with_cyc_dir: Path = DEFAULT_WITH_CYC_DIR,
    with_suffix: str = "",
    with_file_mode: str = "legacy",
    target_lon: float = DEFAULT_TARGET_LON,
    target_lat: float = DEFAULT_TARGET_LAT,
) -> dict[str, dict]:
    """Load paired original / with-cyclones series at the nearest site.

    ``with_file_mode``:
      - ``legacy``: ``{var}_500m_with_cyclones{suffix}.nc`` in ``with_cyc_dir``
      - ``same_name``: ``{var}_500m.nc`` in ``with_cyc_dir`` (all-sites splice dirs)
    """
    if with_file_mode not in {"legacy", "same_name"}:
        raise ValueError("with_file_mode must be 'legacy' or 'same_name'")
    original_dir = Path(original_dir)
    with_cyc_dir = Path(with_cyc_dir)
    out: dict[str, dict] = {}
    site_index = None
    for key in variables:
        meta = VAR_META[key]
        orig_path = original_dir / meta["file"]
        if with_file_mode == "same_name":
            cyc_path = with_cyc_dir / meta["file"]
        else:
            cyc_path = with_cyc_dir / _with_file_name(key, with_suffix)
        if not orig_path.is_file():
            raise FileNotFoundError(orig_path)
        if not cyc_path.is_file():
            raise FileNotFoundError(
                f"Missing {cyc_path}. Build it with splice_cyclones_* before plotting {key}."
            )
        t0, y0, site_index = load_site_series(
            orig_path, meta["var"], target_lon=target_lon, target_lat=target_lat, site_index=site_index
        )
        t1, y1, _ = load_site_series(
            cyc_path, meta["var"], target_lon=target_lon, target_lat=target_lat, site_index=site_index
        )
        df = pd.DataFrame({"orig": y0}, index=t0).join(pd.DataFrame({"cyc": y1}, index=t1), how="inner")
        out[key] = {
            "time": pd.DatetimeIndex(df.index),
            "original": df["orig"].to_numpy(),
            "with_cyclones": df["cyc"].to_numpy(),
            "site_index": site_index,
            "title": meta["title"],
            "ylabel": meta["ylabel"],
        }
    return out


def load_four_scenarios(
    variables: tuple[str, ...] = ("hs", "tp", "dp"),
    *,
    original_245: Path = earth3_baseline_dir("ssp245"),
    original_585: Path = earth3_baseline_dir("ssp585"),
    with_cyc_dir: Path = DEFAULT_WITH_CYC_DIR,
    with_cyc_245_dir: Path | None = None,
    with_cyc_585_dir: Path | None = None,
    with_suffix_245: str = "",
    with_suffix_585: str = "_ssp585",
    with_file_mode: str = "legacy",
    target_lon: float = DEFAULT_TARGET_LON,
    target_lat: float = DEFAULT_TARGET_LAT,
) -> dict[str, dict]:
    """Load original and +cyclones series for both SSP245 and SSP585."""
    dir_245 = Path(with_cyc_245_dir or with_cyc_dir)
    dir_585 = Path(with_cyc_585_dir or with_cyc_dir)
    pair_245 = load_original_and_with_cyclones(
        variables,
        original_dir=original_245,
        with_cyc_dir=dir_245,
        with_suffix=with_suffix_245,
        with_file_mode=with_file_mode,
        target_lon=target_lon,
        target_lat=target_lat,
    )
    pair_585 = load_original_and_with_cyclones(
        variables,
        original_dir=original_585,
        with_cyc_dir=dir_585,
        with_suffix=with_suffix_585,
        with_file_mode=with_file_mode,
        target_lon=target_lon,
        target_lat=target_lat,
    )
    out: dict[str, dict] = {}
    for key in variables:
        d245, d585 = pair_245[key], pair_585[key]
        df = (
            pd.DataFrame(
                {
                    "ssp245_original": d245["original"],
                    "ssp245_cyclones": d245["with_cyclones"],
                },
                index=d245["time"],
            )
            .join(
                pd.DataFrame(
                    {
                        "ssp585_original": d585["original"],
                        "ssp585_cyclones": d585["with_cyclones"],
                    },
                    index=d585["time"],
                ),
                how="inner",
            )
            .dropna(how="any")
        )
        out[key] = {
            "time": pd.DatetimeIndex(df.index),
            "series": {k: df[k].to_numpy() for k in df.columns},
            "site_index": d245["site_index"],
            "title": VAR_META[key]["title"],
            "ylabel": VAR_META[key]["ylabel"],
        }
        print(
            f"  {key}: site={out[key]['site_index']}  n={len(df)}  "
            f"({df.index[0].date()} → {df.index[-1].date()})"
        )
    return out


def _finite_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def _finite(a: np.ndarray) -> np.ndarray:
    return a[np.isfinite(a)]


def plot_density_histograms(
    data: dict[str, dict],
    *,
    output_png: Path,
    bins: int = 40,
    label_original: str = LABEL_ORIGINAL,
    label_with_cyc: str = LABEL_WITH_CYC,
    color_original: str = COLORS["ssp245_original"],
    color_with_cyc: str = COLORS["ssp245_cyclones"],
    title: str | None = None,
    include_historical: bool = False,
) -> Path:
    """Density + exceedance plots so extreme-tail changes are visible."""
    keys = [k for k in ("hs", "tp", "dp") if k in data]
    fig, axes = plt.subplots(len(keys), 2, figsize=(11, 3.2 * len(keys)), sharex=False)
    if len(keys) == 1:
        axes = np.asarray([axes])

    for row, key in enumerate(keys):
        d = data[key]
        a, b = _finite_pair(d["original"], d["with_cyclones"])
        hist = _finite(d["historical"]) if include_historical and "historical" in d else None
        ax_hist, ax_exc = axes[row, 0], axes[row, 1]

        samples = [a, b] + ([hist] if hist is not None and len(hist) else [])
        lo = float(min(s.min() for s in samples))
        hi = float(max(s.max() for s in samples))
        edges = np.linspace(lo, hi, bins + 1)

        if hist is not None and len(hist):
            ax_hist.hist(
                hist, bins=edges, density=True, alpha=0.35, color=COLORS["historical"], label=LABEL_HISTORICAL
            )
        ax_hist.hist(b, bins=edges, density=True, alpha=0.55, color=color_with_cyc, label=label_with_cyc)
        ax_hist.hist(a, bins=edges, density=True, alpha=0.55, color=color_original, label=label_original)
        ax_hist.set_title(f"{d['title']} density")
        ax_hist.set_ylabel("density")
        ax_hist.set_xlabel("value")
        ax_hist.grid(True, alpha=0.3)
        ax_hist.legend(loc="upper right", fontsize=8)

        xa = np.sort(a)
        xb = np.sort(b)
        pa = 1.0 - (np.arange(1, len(xa) + 1) / (len(xa) + 1.0))
        pb = 1.0 - (np.arange(1, len(xb) + 1) / (len(xb) + 1.0))
        if hist is not None and len(hist):
            xh = np.sort(hist)
            ph = 1.0 - (np.arange(1, len(xh) + 1) / (len(xh) + 1.0))
            ax_exc.semilogy(xh, ph, color=COLORS["historical"], lw=2.0, label=LABEL_HISTORICAL)
            ax_exc.axvline(xh.max(), color=COLORS["historical"], ls=":", lw=1.0, alpha=0.7)
        ax_exc.semilogy(xb, pb, color=color_with_cyc, lw=1.6, label=label_with_cyc)
        ax_exc.semilogy(xa, pa, color=color_original, lw=1.6, label=label_original)
        ax_exc.set_title(f"{d['title']} exceedance P(X≥x)")
        ax_exc.set_ylabel("P(X ≥ x)")
        ax_exc.set_xlabel("value")
        ax_exc.grid(True, which="both", alpha=0.3)
        ax_exc.legend(loc="upper right", fontsize=8)
        ax_exc.axvline(a.max(), color=color_original, ls=":", lw=1.0, alpha=0.7)
        ax_exc.axvline(b.max(), color=color_with_cyc, ls=":", lw=1.0, alpha=0.7)

    fig.suptitle(
        title
        or "Bulk density is similar (few replaced days); extremes show up in exceedance / Q-Q",
        y=1.01,
        fontsize=10,
    )
    fig.tight_layout()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_png}")
    return output_png


def _qq_probabilities(
    n_bulk: int = 180,
    *,
    q_lo: float = 0.01,
    q_hi: float = 0.99,
    include_extrema: bool = False,
) -> np.ndarray:
    """Probability levels for Q-Q plots.

    Bulk defaults to p1–p99. With ``include_extrema``, denser upper-tail
    levels through p100 are appended so rare cyclone extremes appear.
    """
    qs = np.linspace(q_lo, q_hi, n_bulk)
    if include_extrema and q_hi < 1.0:
        tail = np.array(
            [0.992, 0.995, 0.997, 0.998, 0.999, 0.9995, 0.9998, 0.9999, 1.0]
        )
        qs = np.unique(np.concatenate([qs, tail[tail > q_hi]]))
    elif include_extrema and q_hi >= 1.0:
        # already includes the max; still densify the far tail
        tail = np.array([0.995, 0.997, 0.998, 0.999, 0.9995, 0.9998, 0.9999, 1.0])
        qs = np.unique(np.concatenate([qs, tail]))
    return qs


def plot_qq(
    data: dict[str, dict],
    *,
    output_png: Path,
    n_quantiles: int = 200,
    label_original: str = LABEL_ORIGINAL,
    label_with_cyc: str = LABEL_WITH_CYC,
    color_original: str = COLORS["ssp245_original"],
    color_with_cyc: str = COLORS["ssp245_cyclones"],
    title: str | None = None,
    include_historical: bool = False,
    include_extrema: bool = False,
    show_tail_zoom: bool = False,
) -> Path:
    """Q-Q plots.

    Default: x=original, y=+cyclones.
    With historical: x=historical, y=each climate series.

    ``include_extrema`` extends levels through p100.
    ``show_tail_zoom`` adds a second row zoomed on p90–p100.
    """
    keys = [k for k in ("hs", "tp", "dp") if k in data]
    n_rows = 2 if show_tail_zoom else 1
    fig, axes = plt.subplots(
        n_rows,
        len(keys),
        figsize=(4.4 * len(keys), 4.2 * n_rows),
        squeeze=False,
    )

    row_specs = [(0, _qq_probabilities(n_quantiles, include_extrema=include_extrema), "")]
    if show_tail_zoom:
        row_specs.append(
            (
                1,
                _qq_probabilities(120, q_lo=0.90, q_hi=1.0, include_extrema=True),
                "upper tail (p90–p100)",
            )
        )

    for row, qs, row_note in row_specs:
        for col, key in enumerate(keys):
            ax = axes[row, col]
            d = data[key]
            if include_historical and "historical" in d:
                h = _finite(d["historical"])
                a = _finite(d["original"])
                b = _finite(d["with_cyclones"])
                qh = np.quantile(h, qs)
                qa = np.quantile(a, qs)
                qb = np.quantile(b, qs)
                lims = (
                    float(min(qh.min(), qa.min(), qb.min())),
                    float(max(qh.max(), qa.max(), qb.max())),
                )
                ax.plot(lims, lims, "k--", lw=1.0, zorder=1)
                ax.scatter(qh, qa, s=10, c=color_original, alpha=0.85, zorder=2, label=label_original)
                ax.scatter(qh, qb, s=10, c=color_with_cyc, alpha=0.85, zorder=3, label=label_with_cyc)
                ax.set_xlabel(LABEL_HISTORICAL)
                ax.set_ylabel("climate / +cyclones")
                ax.legend(loc="upper left", fontsize=7)
            else:
                a, b = _finite_pair(d["original"], d["with_cyclones"])
                qa = np.quantile(a, qs)
                qb = np.quantile(b, qs)
                lims = (float(min(qa.min(), qb.min())), float(max(qa.max(), qb.max())))
                ax.plot(lims, lims, "k--", lw=1.0, zorder=1)
                ax.scatter(qa, qb, s=8, c=color_with_cyc, alpha=0.8, zorder=2)
                ax.set_xlabel(label_original)
                ax.set_ylabel(label_with_cyc)
            ttl = d["title"] if not row_note else f"{d['title']} — {row_note}"
            ax.set_title(ttl)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        title
        or (
            f"Q-Q vs {LABEL_HISTORICAL}"
            if include_historical
            else "Q-Q: earth3_veg_lr original vs +cyclones"
        ),
        y=1.02,
    )
    fig.tight_layout()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_png}")
    return output_png


def plot_seasonal_cycle(
    data: dict[str, dict],
    *,
    variables: tuple[str, ...] = ("hs", "tp", "dp"),
    output_png: Path,
    label_original: str = LABEL_ORIGINAL,
    label_with_cyc: str = LABEL_WITH_CYC,
    color_original: str = COLORS["ssp245_original"],
    color_with_cyc: str = COLORS["ssp245_cyclones"],
    title: str | None = None,
    include_historical: bool = False,
) -> Path:
    """Monthly mean seasonal cycle panels for HS / TP / DP."""
    keys = [k for k in variables if k in data]
    if not keys:
        raise KeyError("no requested variables in data")

    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 3.2 * len(keys)), sharex=True)
    if len(keys) == 1:
        axes = [axes]
    x = np.arange(1, 13)

    for ax, key in zip(axes, keys):
        d = data[key]
        months = d["time"].month
        orig_m = np.array([np.nanmean(d["original"][months == m]) for m in range(1, 13)])
        cyc_m = np.array([np.nanmean(d["with_cyclones"][months == m]) for m in range(1, 13)])
        if include_historical and "historical" in d:
            hm = d["historical_time"].month
            hist_m = np.array([np.nanmean(d["historical"][hm == m]) for m in range(1, 13)])
            ax.plot(
                x,
                hist_m,
                "-o",
                color=COLORS["historical"],
                lw=2.0,
                markersize=6,
                label=LABEL_HISTORICAL,
                zorder=3,
            )
        ax.plot(x, orig_m, "-o", color=color_original, lw=1.5, markersize=6, label=label_original)
        ax.plot(x, cyc_m, "-o", color=color_with_cyc, lw=1.5, markersize=6, label=label_with_cyc)
        ax.set_ylabel(d["ylabel"])
        ax.set_title(f"{d['title']} seasonal cycle")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xticks(x)
    axes[-1].set_xlabel("Month")
    if title:
        fig.suptitle(title, y=1.01, fontsize=11)
    fig.tight_layout()

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_png}")
    return output_png


SCENARIO_LABELS = {
    "historical": LABEL_HISTORICAL,
    "ssp245_original": "earth3_veg_lr SSP245",
    "ssp245_cyclones": "earth3_veg_lr SSP245 + cyclones",
    "ssp585_original": "earth3_veg_lr SSP585",
    "ssp585_cyclones": "earth3_veg_lr SSP585 + cyclones",
}
SCENARIO_ORDER = (
    "ssp245_original",
    "ssp245_cyclones",
    "ssp585_original",
    "ssp585_cyclones",
)


def plot_density_histograms_four(
    data: dict[str, dict],
    *,
    output_png: Path,
    bins: int = 40,
    include_historical: bool = False,
) -> Path:
    """Density + exceedance for all four scenarios (+ optional historical)."""
    keys = [k for k in ("hs", "tp", "dp") if k in data]
    fig, axes = plt.subplots(len(keys), 2, figsize=(12, 3.4 * len(keys)), sharex=False)
    if len(keys) == 1:
        axes = np.asarray([axes])

    for row, key in enumerate(keys):
        d = data[key]
        series = {k: _finite(d["series"][k]) for k in SCENARIO_ORDER}
        hist = _finite(d["historical"]) if include_historical and "historical" in d else None
        ax_hist, ax_exc = axes[row, 0], axes[row, 1]

        samples = list(series.values()) + ([hist] if hist is not None and len(hist) else [])
        lo = float(min(v.min() for v in samples if len(v)))
        hi = float(max(v.max() for v in samples if len(v)))
        edges = np.linspace(lo, hi, bins + 1)

        if hist is not None and len(hist):
            ax_hist.hist(
                hist,
                bins=edges,
                density=True,
                alpha=0.35,
                color=COLORS["historical"],
                label=LABEL_HISTORICAL,
            )
            xs = np.sort(hist)
            ps = 1.0 - (np.arange(1, len(xs) + 1) / (len(xs) + 1.0))
            ax_exc.semilogy(xs, ps, color=COLORS["historical"], lw=2.0, label=LABEL_HISTORICAL)
            ax_exc.axvline(xs.max(), color=COLORS["historical"], ls=":", lw=1.0, alpha=0.6)

        for sk in SCENARIO_ORDER:
            ax_hist.hist(
                series[sk],
                bins=edges,
                density=True,
                alpha=0.40,
                color=COLORS[sk],
                label=SCENARIO_LABELS[sk],
            )
            xs = np.sort(series[sk])
            ps = 1.0 - (np.arange(1, len(xs) + 1) / (len(xs) + 1.0))
            ax_exc.semilogy(xs, ps, color=COLORS[sk], lw=1.7, label=SCENARIO_LABELS[sk])
            ax_exc.axvline(xs.max(), color=COLORS[sk], ls=":", lw=1.0, alpha=0.6)

        ax_hist.set_title(f"{d['title']} density")
        ax_hist.set_ylabel("density")
        ax_hist.set_xlabel("value")
        ax_hist.grid(True, alpha=0.3)
        ax_hist.legend(loc="upper right", fontsize=7)

        ax_exc.set_title(f"{d['title']} exceedance P(X≥x)")
        ax_exc.set_ylabel("P(X ≥ x)")
        ax_exc.set_xlabel("value")
        ax_exc.grid(True, which="both", alpha=0.3)
        ax_exc.legend(loc="upper right", fontsize=7)

    fig.suptitle(
        f"{LABEL_HISTORICAL} (black) + earth3_veg_lr SSP245 (blues) vs SSP585 (oranges)"
        if include_historical
        else "earth3_veg_lr SSP245 (blues) vs SSP585 (oranges): original vs +cyclones",
        y=1.01,
        fontsize=11,
    )
    fig.tight_layout()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_png}")
    return output_png


def plot_qq_four(
    data: dict[str, dict],
    *,
    output_png: Path,
    n_quantiles: int = 200,
    include_historical: bool = False,
    include_extrema: bool = False,
    show_tail_zoom: bool = False,
) -> Path:
    """Q-Q plots for four scenarios.

    Default: +cyclones vs own original.
    With historical: each series vs historical (x=historical).

    ``include_extrema`` extends levels through p100.
    ``show_tail_zoom`` adds a second row zoomed on p90–p100 (makes rare
    cyclone extremes visible without compressing the bulk panel).
    """
    keys = [k for k in ("hs", "tp", "dp") if k in data]
    n_rows = 2 if show_tail_zoom else 1
    fig, axes = plt.subplots(
        n_rows,
        len(keys),
        figsize=(4.4 * len(keys), 4.2 * n_rows),
        squeeze=False,
    )

    pairs = (
        ("ssp245_original", "ssp245_cyclones"),
        ("ssp585_original", "ssp585_cyclones"),
    )
    row_specs = [(0, _qq_probabilities(n_quantiles, include_extrema=include_extrema), "")]
    if show_tail_zoom:
        row_specs.append(
            (
                1,
                _qq_probabilities(120, q_lo=0.90, q_hi=1.0, include_extrema=True),
                "upper tail (p90–p100)",
            )
        )

    for row, qs, row_note in row_specs:
        for col, key in enumerate(keys):
            ax = axes[row, col]
            d = data[key]
            all_q = []
            if include_historical and "historical" in d:
                h = _finite(d["historical"])
                qh = np.quantile(h, qs)
                all_q.extend([qh.min(), qh.max()])
                for sk in SCENARIO_ORDER:
                    y = _finite(d["series"][sk])
                    qy = np.quantile(y, qs)
                    all_q.extend([qy.min(), qy.max()])
                    ax.scatter(
                        qh,
                        qy,
                        s=10,
                        c=COLORS[sk],
                        alpha=0.85,
                        zorder=2,
                        label=SCENARIO_LABELS[sk],
                    )
                ax.set_xlabel(LABEL_HISTORICAL)
                ax.set_ylabel("climate / +cyclones")
            else:
                for orig_k, cyc_k in pairs:
                    a = _finite(d["series"][orig_k])
                    b = _finite(d["series"][cyc_k])
                    qa = np.quantile(a, qs)
                    qb = np.quantile(b, qs)
                    all_q.extend([qa.min(), qa.max(), qb.min(), qb.max()])
                    ax.scatter(
                        qa,
                        qb,
                        s=10,
                        c=COLORS[cyc_k],
                        alpha=0.85,
                        zorder=2,
                        label=f"{SCENARIO_LABELS[cyc_k]} vs {SCENARIO_LABELS[orig_k]}",
                    )
                ax.set_xlabel("Original")
                ax.set_ylabel("+ cyclones")
            lims = (float(min(all_q)), float(max(all_q)))
            ax.plot(lims, lims, "k--", lw=1.0, zorder=1)
            ttl = d["title"] if not row_note else f"{d['title']} — {row_note}"
            ax.set_title(ttl)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left", fontsize=7)

    base = (
        f"Q-Q vs {LABEL_HISTORICAL}: earth3_veg_lr SSP245/SSP585"
        if include_historical
        else "Q-Q: earth3_veg_lr original vs +cyclones (SSP245 blue, SSP585 gold)"
    )
    if show_tail_zoom:
        base += "  |  top: p1–p99, bottom: p90–p100"
    elif include_extrema:
        base += "  |  quantiles through p100"
    fig.suptitle(base, y=1.02)
    fig.tight_layout()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_png}")
    return output_png


def plot_seasonal_cycle_four(
    data: dict[str, dict],
    *,
    variables: tuple[str, ...] = ("hs", "tp", "dp"),
    output_png: Path,
    include_historical: bool = False,
    overlapping_only: bool = False,
    stat: str = "mean",
) -> Path:
    """Seasonal cycle panels for all four scenarios (HS / TP / DP).

    ``stat``:
      - ``mean``: plain climatological monthly mean
      - ``max``: climatological monthly maximum (max over all years in that month)
    """
    if stat not in {"mean", "max"}:
        raise ValueError("stat must be 'mean' or 'max'")
    agg = climatological_monthly_mean if stat == "mean" else climatological_monthly_max
    stat_label = "monthly mean" if stat == "mean" else "monthly maximum"

    keys = [k for k in variables if k in data]
    if not keys:
        raise KeyError("no requested variables in data")

    overlap_note = ""
    plot_data = data
    if overlapping_only:
        plot_data, t0, t1 = restrict_to_overlapping_period(
            data, include_historical=include_historical
        )
        overlap_note = f" [overlapping {t0.date()} → {t1.date()}]"

    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 3.4 * len(keys)), sharex=True)
    if len(keys) == 1:
        axes = [axes]
    x = np.arange(1, 13)

    for ax, key in zip(axes, keys):
        d = plot_data[key]
        if include_historical and "historical" in d:
            hist_m = agg(d["historical_time"], d["historical"])
            ax.plot(
                x,
                hist_m,
                "-o",
                color=COLORS["historical"],
                lw=2.2,
                markersize=6,
                label=LABEL_HISTORICAL,
                zorder=3,
            )
        for sk in SCENARIO_ORDER:
            monthly = agg(d["time"], d["series"][sk])
            ax.plot(
                x,
                monthly,
                "-o",
                color=COLORS[sk],
                lw=1.7,
                markersize=6,
                label=SCENARIO_LABELS[sk],
            )
        ax.set_ylabel(d["ylabel"])
        ax.set_title(f"{d['title']} seasonal cycle ({stat_label})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)

    axes[-1].set_xticks(x)
    axes[-1].set_xlabel("Month")
    base = (
        f"{LABEL_HISTORICAL} (black) + earth3_veg_lr SSP245 (blues) vs SSP585 (oranges)"
        if include_historical
        else "earth3_veg_lr SSP245 (blues) vs SSP585 (oranges): seasonal cycle"
    )
    fig.suptitle(base + f" — {stat_label}" + overlap_note, y=1.01, fontsize=11)
    fig.tight_layout()

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_png}")
    return output_png


def plot_original_vs_cyclones_diagnostics(
    *,
    original_dir: Path = DEFAULT_ORIGINAL_DIR,
    with_cyc_dir: Path = DEFAULT_WITH_CYC_DIR,
    with_suffix: str = "",
    output_dir: Path | None = None,
    name_suffix: str = "",
    target_lon: float = DEFAULT_TARGET_LON,
    target_lat: float = DEFAULT_TARGET_LAT,
    variables: tuple[str, ...] = ("hs", "tp", "dp"),
    label_original: str = LABEL_ORIGINAL,
    label_with_cyc: str = LABEL_WITH_CYC,
    color_original: str = COLORS["ssp245_original"],
    color_with_cyc: str = COLORS["ssp245_cyclones"],
    scenario_tag: str = "SSP245",
    include_historical: bool = False,
    historical_dir: Path = DEFAULT_HISTORICAL_DIR,
) -> dict[str, Path]:
    """Create density, Q-Q and seasonal-cycle figures for the nearest site."""
    output_dir = Path(output_dir or DEFAULT_PROJECT / "outputs")
    data = load_original_and_with_cyclones(
        variables,
        original_dir=original_dir,
        with_cyc_dir=with_cyc_dir,
        with_suffix=with_suffix,
        target_lon=target_lon,
        target_lat=target_lat,
    )
    if include_historical:
        hist = load_historical_daily(
            variables,
            historical_dir=historical_dir,
            target_lon=target_lon,
            target_lat=target_lat,
        )
        attach_historical(data, hist)
    suf = name_suffix or ""
    paths = {
        "density": plot_density_histograms(
            data,
            output_png=output_dir / f"compare_density_hs_tp_dp{suf}.png",
            label_original=label_original,
            label_with_cyc=label_with_cyc,
            color_original=color_original,
            color_with_cyc=color_with_cyc,
            title=(
                f"{scenario_tag} + {LABEL_HISTORICAL}: density / exceedance"
                if include_historical
                else f"{scenario_tag}: bulk density similar; extremes in exceedance / Q-Q"
            ),
            include_historical=include_historical,
        ),
        "qq": plot_qq(
            data,
            output_png=output_dir / f"compare_qq_hs_tp_dp{suf}.png",
            label_original=label_original,
            label_with_cyc=label_with_cyc,
            color_original=color_original,
            color_with_cyc=color_with_cyc,
            title=(
                f"Q-Q: {scenario_tag} vs {LABEL_HISTORICAL}"
                if include_historical
                else f"Q-Q: {scenario_tag} original vs +cyclones"
            ),
            include_historical=include_historical,
        ),
        "seasonal": plot_seasonal_cycle(
            data,
            variables=("hs", "tp", "dp"),
            output_png=output_dir / f"compare_seasonal_hs{suf}.png",
            label_original=label_original,
            label_with_cyc=label_with_cyc,
            color_original=color_original,
            color_with_cyc=color_with_cyc,
            title=(
                f"{scenario_tag} + {LABEL_HISTORICAL}: seasonal cycle (HS / TP / DP)"
                if include_historical
                else f"{scenario_tag} seasonal cycle (HS / TP / DP)"
            ),
            include_historical=include_historical,
        ),
    }
    return paths


def plot_four_scenario_diagnostics(
    *,
    output_dir: Path | None = None,
    target_lon: float = DEFAULT_TARGET_LON,
    target_lat: float = DEFAULT_TARGET_LAT,
    variables: tuple[str, ...] = ("hs", "tp", "dp"),
    include_historical: bool = False,
    historical_dir: Path = DEFAULT_HISTORICAL_DIR,
    name_suffix: str = "",
    with_cyc_245_dir: Path | None = None,
    with_cyc_585_dir: Path | None = None,
    with_file_mode: str = "legacy",
    qq_show_tail_zoom: bool = True,
    seasonal_overlapping: bool = True,
    seasonal_maxima: bool = True,
) -> dict[str, Path]:
    """Create 4-way SSP245/SSP585 original/+cyclones comparison figures."""
    output_dir = Path(output_dir or DEFAULT_PROJECT / "outputs")
    data = load_four_scenarios(
        variables,
        target_lon=target_lon,
        target_lat=target_lat,
        with_cyc_245_dir=with_cyc_245_dir,
        with_cyc_585_dir=with_cyc_585_dir,
        with_file_mode=with_file_mode,
    )
    if include_historical:
        hist = load_historical_daily(
            variables,
            historical_dir=historical_dir,
            target_lon=target_lon,
            target_lat=target_lat,
        )
        attach_historical(data, hist)
    suf = name_suffix or ("_with_historical" if include_historical else "")
    paths: dict[str, Path] = {
        "density": plot_density_histograms_four(
            data,
            output_png=output_dir / f"compare_density_hs_tp_dp_all_scenarios{suf}.png",
            include_historical=include_historical,
        ),
        "qq": plot_qq_four(
            data,
            output_png=output_dir / f"compare_qq_hs_tp_dp_all_scenarios{suf}.png",
            include_historical=include_historical,
            show_tail_zoom=qq_show_tail_zoom,
        ),
        "seasonal": plot_seasonal_cycle_four(
            data,
            variables=("hs", "tp", "dp"),
            output_png=output_dir / f"compare_seasonal_hs_all_scenarios{suf}.png",
            include_historical=include_historical,
            overlapping_only=False,
            stat="mean",
        ),
    }
    if include_historical and seasonal_overlapping:
        paths["seasonal_overlapping"] = plot_seasonal_cycle_four(
            data,
            variables=("hs", "tp", "dp"),
            output_png=output_dir / f"compare_seasonal_hs_all_scenarios{suf}_overlapping.png",
            include_historical=True,
            overlapping_only=True,
            stat="mean",
        )
    if include_historical and seasonal_maxima:
        paths["seasonal_maxima"] = plot_seasonal_cycle_four(
            data,
            variables=("hs", "tp", "dp"),
            output_png=output_dir / f"compare_seasonal_hs_all_scenarios{suf}_maxima.png",
            include_historical=True,
            overlapping_only=False,
            stat="max",
        )
        paths["seasonal_overlapping_maxima"] = plot_seasonal_cycle_four(
            data,
            variables=("hs", "tp", "dp"),
            output_png=output_dir
            / f"compare_seasonal_hs_all_scenarios{suf}_overlapping_maxima.png",
            include_historical=True,
            overlapping_only=True,
            stat="max",
        )
    return paths


if __name__ == "__main__":
    # Refresh SSP245 pair with new blue palette
    plot_original_vs_cyclones_diagnostics(
        original_dir=earth3_baseline_dir("ssp245"),
        with_suffix="",
        name_suffix="",
        scenario_tag="earth3_veg_lr SSP245",
        label_original=SCENARIO_LABELS["ssp245_original"],
        label_with_cyc=SCENARIO_LABELS["ssp245_cyclones"],
        color_original=COLORS["ssp245_original"],
        color_with_cyc=COLORS["ssp245_cyclones"],
    )
    # SSP585 pair
    plot_original_vs_cyclones_diagnostics(
        original_dir=earth3_baseline_dir("ssp585"),
        with_suffix="_ssp585",
        name_suffix="_ssp585",
        scenario_tag="earth3_veg_lr SSP585",
        label_original=SCENARIO_LABELS["ssp585_original"],
        label_with_cyc=SCENARIO_LABELS["ssp585_cyclones"],
        color_original=COLORS["ssp585_original"],
        color_with_cyc=COLORS["ssp585_cyclones"],
    )
    plot_four_scenario_diagnostics()

    # New figures with historical (1980-2023) in black
    plot_original_vs_cyclones_diagnostics(
        original_dir=earth3_baseline_dir("ssp245"),
        with_suffix="",
        name_suffix="_with_historical",
        scenario_tag="earth3_veg_lr SSP245",
        label_original=SCENARIO_LABELS["ssp245_original"],
        label_with_cyc=SCENARIO_LABELS["ssp245_cyclones"],
        color_original=COLORS["ssp245_original"],
        color_with_cyc=COLORS["ssp245_cyclones"],
        include_historical=True,
    )
    plot_original_vs_cyclones_diagnostics(
        original_dir=earth3_baseline_dir("ssp585"),
        with_suffix="_ssp585",
        name_suffix="_ssp585_with_historical",
        scenario_tag="earth3_veg_lr SSP585",
        label_original=SCENARIO_LABELS["ssp585_original"],
        label_with_cyc=SCENARIO_LABELS["ssp585_cyclones"],
        color_original=COLORS["ssp585_original"],
        color_with_cyc=COLORS["ssp585_cyclones"],
        include_historical=True,
    )
    plot_four_scenario_diagnostics(include_historical=True)
