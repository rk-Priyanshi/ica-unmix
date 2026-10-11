"""Unmix the Signal: ICA (FastICA) implemented from scratch with NumPy."""
from .audio_io import load_wav, save_wav, load_sources
from .mixing import make_mixing_matrix, mix, add_noise
from .preprocessing import center, whiten
from .fastica import fastica, ica_separate
from .pca_baseline import pca_separate
from .evaluation import align_sources, evaluate

__all__ = [
    "load_wav", "save_wav", "load_sources",
    "make_mixing_matrix", "mix", "add_noise",
    "center", "whiten", "fastica", "ica_separate",
    "pca_separate", "align_sources", "evaluate",
]
