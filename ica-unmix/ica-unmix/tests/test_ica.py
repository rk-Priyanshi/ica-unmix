import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unmix import center, whiten, ica_separate, make_mixing_matrix, mix, evaluate, pca_separate


@pytest.fixture
def sources():
    rng = np.random.default_rng(0)
    T = 20000
    t = np.arange(T)
    S = np.stack([np.sign(np.sin(0.01 * t)),                 # square wave (sub-Gaussian)
                  rng.laplace(size=T),                        # Laplacian (super-Gaussian)
                  np.sin(0.037 * t) ** 3])
    return S / S.std(axis=1, keepdims=True)


def test_centering():
    X = np.random.default_rng(1).normal(3, 2, (3, 1000))
    Xc, _ = center(X)
    assert np.allclose(Xc.mean(axis=1), 0, atol=1e-12)


def test_whitening_identity_covariance(sources):
    X = mix(sources, make_mixing_matrix(3, seed=1))
    Xc, _ = center(X)
    Z, *_ = whiten(Xc)
    assert np.allclose(np.cov(Z), np.eye(3), atol=1e-8)


@pytest.mark.parametrize("algorithm", ["parallel", "deflation"])
@pytest.mark.parametrize("fun", ["logcosh", "exp", "cube"])
def test_ica_recovers_sources(sources, algorithm, fun):
    X = mix(sources, make_mixing_matrix(3, seed=2))
    res = ica_separate(X, 3, fun=fun, algorithm=algorithm)
    ev = evaluate(res["S_hat"], sources)
    assert ev["mean_corr"] > 0.98


def test_ica_beats_pca(sources):
    X = mix(sources, make_mixing_matrix(3, seed=3))
    assert evaluate(ica_separate(X, 3)["S_hat"], sources)["mean_sdr_db"] > \
           evaluate(pca_separate(X, 3), sources)["mean_sdr_db"] + 10


def test_more_mics_than_sources(sources):
    X = mix(sources, make_mixing_matrix(3, n_mics=5, seed=4))
    assert evaluate(ica_separate(X, 3)["S_hat"], sources)["mean_corr"] > 0.98
