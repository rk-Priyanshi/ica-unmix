"""Centering and PCA whitening, implemented from scratch (covariance + eigendecomposition)."""
from __future__ import annotations
import numpy as np


def center(X: np.ndarray):
    """Subtract the per-channel mean. Returns (Xc, mean) with mean shape (M, 1)."""
    mean = X.mean(axis=1, keepdims=True)
    return X - mean, mean


def whiten(Xc: np.ndarray, n_components: int | None = None, eps: float = 1e-10):
    """PCA whitening.

    1. Covariance          C = Xc Xc^T / (T-1)
    2. Eigendecomposition  C = E D E^T
    3. Whitening matrix    V = D^{-1/2} E^T    =>  Z = V Xc has identity covariance.

    Keeping only the top `n_components` eigenvectors also reduces dimension
    (useful when there are more microphones than sources).

    Returns Z (K, T), V (K, M), V_inv (M, K) [de-whitening], eigenvalues (descending).
    """
    M, T = Xc.shape
    C = (Xc @ Xc.T) / (T - 1)
    eigvals, eigvecs = np.linalg.eigh(C)            # ascending for symmetric matrices
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    k = n_components or M
    d = np.maximum(eigvals[:k], eps)
    E = eigvecs[:, :k]
    V = (E / np.sqrt(d)).T                          # D^{-1/2} E^T
    V_inv = E * np.sqrt(d)                          # E D^{1/2}
    return V @ Xc, V, V_inv, eigvals
