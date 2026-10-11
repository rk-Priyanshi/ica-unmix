"""Simulate a multi-microphone recording: X = A @ S (+ noise)."""
from __future__ import annotations
import numpy as np


def make_mixing_matrix(n_sources: int, n_mics: int | None = None, seed: int = 0,
                       max_cond: float = 20.0) -> np.ndarray:
    """Random positive mixing matrix A (n_mics x n_sources) with bounded condition number,
    so the problem is well-posed (ill-conditioned A makes any unmixing noise-sensitive)."""
    n_mics = n_mics or n_sources
    rng = np.random.default_rng(seed)
    while True:
        A = rng.uniform(0.2, 1.0, size=(n_mics, n_sources))
        if np.linalg.cond(A) < max_cond:
            return A


def mix(S: np.ndarray, A: np.ndarray) -> np.ndarray:
    """X = A S.  S: (N, T), A: (M, N)  ->  X: (M, T)."""
    return A @ S


def add_noise(X: np.ndarray, snr_db: float | None, seed: int = 0) -> np.ndarray:
    """Add white Gaussian sensor noise at the requested SNR in dB (None = no noise)."""
    if snr_db is None:
        return X
    rng = np.random.default_rng(seed)
    p_sig = np.mean(X ** 2, axis=1, keepdims=True)
    p_noise = p_sig / (10 ** (snr_db / 10))
    return X + rng.standard_normal(X.shape) * np.sqrt(p_noise)
