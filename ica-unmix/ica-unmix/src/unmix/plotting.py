"""Matplotlib figures for the write-up."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_waveforms(S, X, S_hat, sr, names, path, seconds: float = 2.0):
    n = int(seconds * sr)
    t = np.arange(n) / sr
    N, M = S.shape[0], X.shape[0]
    fig, axes = plt.subplots(3, max(N, M), figsize=(4 * max(N, M), 6), sharex=True, squeeze=False)
    for r, (title, data, lab) in enumerate([("Original", S, names), ("Mixed", X, [f"mic {i+1}" for i in range(M)]),
                                            ("Separated (ICA)", S_hat, names)]):
        for c in range(max(N, M)):
            ax = axes[r, c]
            if c < data.shape[0]:
                ax.plot(t, data[c, :n], lw=0.6)
                ax.set_title(f"{title}: {lab[c]}", fontsize=9)
            else:
                ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_corr_matrices(C_pca, C_ica, names, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, C, title in zip(axes, (C_pca, C_ica), ("PCA components", "FastICA components")):
        im = ax.imshow(np.abs(C), vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"|corr| with true sources\n{title}")
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(C.shape[0]), [f"est {i+1}" for i in range(C.shape[0])])
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                ax.text(j, i, f"{abs(C[i, j]):.2f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
