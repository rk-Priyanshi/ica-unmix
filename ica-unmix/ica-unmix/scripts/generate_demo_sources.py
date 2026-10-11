"""Create 3 synthetic demo sources (speech-like, music-like, ambient-like) so the pipeline runs
out of the box. For the real submission, REPLACE these with real recordings in data/sources/
and list them in SOURCES.md.

Note: ICA can separate at most ONE Gaussian source, so the 'ambient' track is built to be
non-Gaussian (modulated Laplacian noise) rather than pure white Gaussian noise.
"""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unmix.audio_io import save_wav

SR, DUR = 16000, 10
t = np.arange(SR * DUR) / SR
rng = np.random.default_rng(42)


def speech_like():
    f0 = 120 + 30 * np.sin(2 * np.pi * 0.7 * t)                  # wobbling pitch
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 12))          # harmonic stack
    syll = np.clip(np.sin(2 * np.pi * 3.0 * t), 0, None) ** 2      # syllable-rate envelope
    return x * syll


def music_like():
    notes = [261.6, 329.6, 392.0, 523.3]                          # C major arpeggio
    x = np.zeros_like(t)
    for i, f in enumerate(notes * 8):
        a, b = int(i * len(t) / 32), int((i + 1) * len(t) / 32)
        tt = np.arange(b - a) / SR
        x[a:b] = np.sin(2 * np.pi * f * tt) * np.exp(-3 * tt) + 0.4 * np.sin(2 * np.pi * 2 * f * tt) * np.exp(-5 * tt)
    return x


def ambient_like():
    noise = rng.laplace(size=len(t))
    k = np.ones(40) / 40                                          # gentle low-pass
    noise = np.convolve(noise, k, mode="same")
    return noise * (0.6 + 0.4 * np.sin(2 * np.pi * 0.3 * t))


out = Path(__file__).resolve().parents[1] / "data" / "sources"
for name, fn in [("speech_like", speech_like), ("music_like", music_like), ("ambient_like", ambient_like)]:
    save_wav(out / f"{name}.wav", fn(), SR)
    print("wrote", out / f"{name}.wav")
