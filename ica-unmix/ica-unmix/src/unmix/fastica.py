"""FastICA (Hyvarinen, 1999) from scratch. No sklearn / no library ICA."""
from __future__ import annotations
import numpy as np
from .preprocessing import center, whiten


# Contrast-function derivatives: each returns (g(u), g'(u)).
def _logcosh(u, alpha=1.0):
    g = np.tanh(alpha * u)
    return g, alpha * (1.0 - g ** 2)

def _exp(u, alpha=1.0):
    e = np.exp(-(u ** 2) / 2.0)
    return u * e, (1.0 - u ** 2) * e

def _cube(u, alpha=1.0):                            # kurtosis-based contrast
    return u ** 3, 3.0 * u ** 2

CONTRASTS = {"logcosh": _logcosh, "exp": _exp, "cube": _cube}


def _sym_decorrelation(W: np.ndarray) -> np.ndarray:
    """W <- (W W^T)^{-1/2} W : keeps rows orthonormal so units can't collapse onto one source."""
    s, u = np.linalg.eigh(W @ W.T)
    s = np.maximum(s, 1e-12)
    return (u * (1.0 / np.sqrt(s))) @ u.T @ W


def fastica(Z: np.ndarray, n_components: int | None = None, fun: str = "logcosh",
            alpha: float = 1.0, algorithm: str = "parallel", max_iter: int = 500,
            tol: float = 1e-6, seed: int = 0):
    """FastICA on WHITENED data Z (K, T).

    Fixed-point update for each unit w (maximises non-Gaussianity of w^T z):
        w+ = E[ z g(w^T z) ] - E[ g'(w^T z) ] w
    followed by orthogonalisation (symmetric for 'parallel', Gram-Schmidt for 'deflation').
    Converged when |<w_new, w_old>| -> 1 for every row (sign-invariant).

    Returns W (n_components, K) and the number of iterations used.
    """
    K, T = Z.shape
    n = n_components or K
    g_fn = CONTRASTS[fun]
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, K))

    if algorithm == "parallel":
        W = _sym_decorrelation(W)
        it = 0
        for it in range(1, max_iter + 1):
            g, gp = g_fn(W @ Z, alpha)
            W_new = (g @ Z.T) / T - gp.mean(axis=1, keepdims=True) * W
            W_new = _sym_decorrelation(W_new)
            lim = np.max(np.abs(np.abs(np.sum(W_new * W, axis=1)) - 1.0))
            W = W_new
            if lim < tol:
                break
        return W, it

    if algorithm == "deflation":
        W_out = np.zeros((n, K))
        total_it = 0
        for p in range(n):
            w = W[p] / np.linalg.norm(W[p])
            it = 0
            for it in range(1, max_iter + 1):
                g, gp = g_fn(w[None, :] @ Z, alpha)
                w_new = (Z @ g[0]) / T - gp.mean() * w
                w_new -= W_out[:p].T @ (W_out[:p] @ w_new)      # Gram-Schmidt vs found units
                w_new /= np.linalg.norm(w_new)
                done = abs(abs(w_new @ w) - 1.0) < tol
                w = w_new
                if done:
                    break
            W_out[p] = w
            total_it += it
        return W_out, total_it

    raise ValueError("algorithm must be 'parallel' or 'deflation'")


def ica_separate(X: np.ndarray, n_components: int | None = None, **kw):
    """Full pipeline: center -> PCA whiten -> FastICA.

    Returns dict with S_hat (n, T) and W_total (n, M) such that S_hat = W_total @ (X - mean),
    plus the mean, whitening matrix, whitened data and iteration count.
    """
    Xc, mean = center(X)
    Z, V, V_inv, eig = whiten(Xc, n_components)
    W, n_iter = fastica(Z, n_components, **kw)
    return {"S_hat": W @ Z, "W_total": W @ V, "W": W, "V": V, "mean": mean,
            "eigvals": eig, "n_iter": n_iter, "Z": Z}
