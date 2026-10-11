"""PCA as a 'separator' baseline: uncorrelated (2nd-order) but NOT independent."""
from __future__ import annotations
from .preprocessing import center, whiten


def pca_separate(X, n_components: int | None = None):
    """Return principal-component time courses (whitened, i.e. unit variance)."""
    Xc, _ = center(X)
    Z, *_ = whiten(Xc, n_components)
    return Z
