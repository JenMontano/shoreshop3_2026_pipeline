import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prepare_overlap_frames(pca_fit, df_new, npcs=10, align_sign=True):
    if not hasattr(pca_fit, "pcs_df"):
        raise ValueError("pca_fit.pcs_df not found")

    df_ref = pca_fit.pcs_df.copy()
    df_ref.index = pd.to_datetime(df_ref.index)

    df_new_local = df_new.copy()
    df_new_local.index = pd.to_datetime(df_new_local.index)

    common_time = df_ref.index.intersection(df_new_local.index)
    if len(common_time) == 0:
        raise ValueError("No overlap between reference and new PCs")

    npcs_eff = min(int(npcs), df_ref.shape[1], df_new_local.shape[1])
    pc_cols = [f"PC{i}" for i in range(1, npcs_eff + 1)]

    ref = df_ref.loc[common_time, pc_cols].copy()
    new = df_new_local.loc[common_time, pc_cols].copy()

    if align_sign:
        for c in pc_cols:
            r = np.corrcoef(ref[c], new[c])[0, 1]
            if np.isfinite(r) and r < 0:
                new[c] = -new[c]

    return ref, new, pc_cols


def plot_pc_pair_scatters(
    pca_fit,
    df_new,
    pairs=None,
    npcs=10,
    figsize=(22, 10),
    alpha=0.35,
    s=5,
    labels=("fit", "new"),
):
    if pairs is None:
        pairs = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 1)]

    ref, new, _ = _prepare_overlap_frames(pca_fit=pca_fit, df_new=df_new, npcs=npcs, align_sign=True)
    max_pc = ref.shape[1]

    fig, axes = plt.subplots(2, 5, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for j, (a, b) in enumerate(pairs[:10]):
        ax = axes[j]
        if a > max_pc or b > max_pc:
            ax.axis("off")
            continue

        xcol, ycol = f"PC{a}", f"PC{b}"
        ax.scatter(ref[xcol], ref[ycol], s=s, alpha=alpha, label=labels[0])
        ax.scatter(new[xcol], new[ycol], s=s, alpha=alpha, label=labels[1])
        ax.set_title(f"{xcol} vs {ycol}")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        if j == 0:
            ax.legend(loc="best", fontsize=8)

    for k in range(min(len(pairs), 10), len(axes)):
        axes[k].axis("off")

    plt.show()


def plot_pc_histograms(
    pca_fit,
    df_new,
    npcs=10,
    ncols=6,
    bins=60,
    figsize_width=22,
    row_height=3.6,
    labels=("Original_data", "Fitted 20CR"),
):
    ref, new, pc_cols = _prepare_overlap_frames(pca_fit=pca_fit, df_new=df_new, npcs=npcs, align_sign=True)

    nrows = int(np.ceil(len(pc_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_width, row_height * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for i, col in enumerate(pc_cols):
        ax = axes[i]
        ax.hist(ref[col], bins=bins, density=True, alpha=0.45, label=labels[0])
        ax.hist(new[col], bins=bins, density=True, alpha=0.45, label=labels[1])
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        if i == 0:
            ax.legend(loc="best", fontsize=8)

    for k in range(len(pc_cols), len(axes)):
        axes[k].axis("off")

    plt.show()


def plot_pc_time_series(
    pca_fit,
    df_new,
    npcs=10,
    ncols=5,
    nrows=2,
    figsize=(22, 6),
    labels=("Original data", "Fitted 20CR"),
):
    ref, new, pc_cols = _prepare_overlap_frames(pca_fit=pca_fit, df_new=df_new, npcs=npcs, align_sign=True)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        constrained_layout=True,
        sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for i, col in enumerate(pc_cols):
        if i >= len(axes):
            break
        ax = axes[i]
        ax.plot(ref.index, ref[col], linewidth=1.2, alpha=0.85, label=labels[0])
        ax.plot(new.index, new[col], linewidth=1.2, alpha=0.85, label=labels[1])
        ax.axhline(0, color="k", linewidth=0.6, alpha=0.25)
        ax.set_title(col)
        ax.set_ylabel(col)
        if i == 0:
            ax.legend(loc="best", fontsize=8)

    for k in range(len(pc_cols), len(axes)):
        axes[k].axis("off")

    plt.show()
