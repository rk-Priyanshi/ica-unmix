"""Metrics that account for ICA's scale, sign and permutation ambiguities."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment   # Hungarian algorithm (matching only)


def corr_matrix(S_hat: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Pearson correlation between every estimate (rows) and every true source (cols)."""
    a = (S_hat - S_hat.mean(1, keepdims=True)) / (S_hat.std(1, keepdims=True) + 1e-12)
    b = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-12)
    return a @ b.T / S.shape[1]


def align_sources(S_hat: np.ndarray, S: np.ndarray):
    """Resolve permutation, sign and scale.

    * permutation: Hungarian matching on |correlation|
    * sign + scale: least-squares gain  g = <s_hat, s> / <s_hat, s_hat>  (negative g = flipped)
    Returns aligned estimates (same order/scale as S), the permutation, and the corr matrix.
    """
    C = corr_matrix(S_hat, S)
    rows, cols = linear_sum_assignment(-np.abs(C))
    perm = np.empty(S.shape[0], dtype=int)
    perm[cols] = rows                                # perm[j] = estimate index matched to source j
    out = np.zeros_like(S)
    for j in range(S.shape[0]):
        e = S_hat[perm[j]]
        out[j] = e * (e @ S[j]) / (e @ e + 1e-12)
    return out, perm, C


def sdr(s_true: np.ndarray, s_est: np.ndarray) -> float:
    """Scale-invariant SDR (dB) of an already-aligned estimate: 10 log10(||s||^2 / ||s - s_est||^2)."""
    err = s_true - s_est
    return float(10 * np.log10((s_true @ s_true) / (err @ err + 1e-20)))


def evaluate(S_hat: np.ndarray, S: np.ndarray) -> dict:
    """Per-source |correlation| and SDR after alignment, plus their means."""
    aligned, perm, C = align_sources(S_hat, S)
    n = S.shape[0]
    corr = [float(abs(C[perm[j], j])) for j in range(n)]
    sdrs = [sdr(S[j], aligned[j]) for j in range(n)]
    return {"aligned": aligned, "perm": perm.tolist(), "corr": corr, "sdr_db": sdrs,
            "mean_corr": float(np.mean(corr)), "mean_sdr_db": float(np.mean(sdrs))}
