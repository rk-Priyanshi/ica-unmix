"""Audio input/output helpers (WAV only, via scipy.io.wavfile)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.io import wavfile


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV file as a mono float64 array in [-1, 1]. Returns (signal, sample_rate)."""
    sr, data = wavfile.read(str(path))
    if np.issubdtype(data.dtype, np.integer):
        if data.dtype == np.uint8:                 # 8-bit WAV is unsigned
            data = (data.astype(np.float64) - 128.0) / 128.0
        else:
            info = np.iinfo(data.dtype)
            data = data.astype(np.float64) / max(abs(info.min), info.max)
    else:
        data = data.astype(np.float64)
    if data.ndim == 2:                              # stereo -> mono
        data = data.mean(axis=1)
    return data, int(sr)


def save_wav(path: str | Path, x: np.ndarray, sr: int, peak: float = 0.95) -> None:
    """Save a 1-D signal as 16-bit PCM WAV, peak-normalised (ICA has arbitrary scale)."""
    x = np.asarray(x, dtype=np.float64)
    m = np.max(np.abs(x))
    if m > 0:
        x = x / m * peak
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sr, (x * 32767).astype(np.int16))


def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Cheap linear-interpolation resampler (fine for a demo project)."""
    if sr_in == sr_out:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x)


def load_sources(paths, sr: int | None = None, max_seconds: float | None = None):
    """Load N WAV files -> S with shape (N, T). Resamples to a common rate, trims to equal length,
    and scales every source to unit variance."""
    sigs, rates = zip(*(load_wav(p) for p in paths))
    sr = sr or rates[0]
    sigs = [_resample_linear(s, r, sr) for s, r in zip(sigs, rates)]
    T = min(len(s) for s in sigs)
    if max_seconds:
        T = min(T, int(max_seconds * sr))
    S = np.stack([s[:T] for s in sigs])
    S = S / (np.std(S, axis=1, keepdims=True) + 1e-12)
    return S, sr
