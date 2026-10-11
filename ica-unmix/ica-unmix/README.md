# Unmix the Signal — ICA from Scratch

Recover clean audio sources from mixed recordings with **Independent Component Analysis**,
implemented with NumPy only (no library ICA): centering → PCA whitening → FastICA → alignment → metrics.

```
X = A S      (mix)      Xc = X - mean      Z = V Xc  (whiten)      Ŝ = W Z = (W V) Xc      (unmix)
```

## Quick start
```bash
pip install -r requirements.txt
python scripts/generate_demo_sources.py          # or drop your own WAVs into data/sources/
python run_pipeline.py --sources data/sources/*.wav --snr 30
python scripts/snr_sweep.py                      # SNR experiment, ICA vs PCA
python -m pytest -q                              # unit tests
```
Audio sources used: see [SOURCES.md](SOURCES.md).

## Repository layout
| Path | Purpose |
|------|---------|
| `src/unmix/audio_io.py` | load / save WAV, resample, equalise length |
| `src/unmix/mixing.py` | mixing matrix, `X = A S`, sensor noise at a chosen SNR |
| `src/unmix/preprocessing.py` | centering + PCA whitening (covariance + `eigh`) |
| `src/unmix/fastica.py` | FastICA: parallel (symmetric) & deflation, contrasts `logcosh`, `exp`, `cube` |
| `src/unmix/pca_baseline.py` | PCA-only "separation" for comparison |
| `src/unmix/evaluation.py` | permutation/sign/scale alignment, correlation, SI-SDR |
| `src/unmix/plotting.py` | figures |
| `run_pipeline.py` | end-to-end CLI |
| `scripts/` | demo-source generator, SNR sweep |
| `tests/` | whitening, recovery, ICA > PCA, more mics than sources |
| `notebooks/report.ipynb` | write-up with results and plots |
| `data/mixed/`, `outputs/separated/` | mixed and separated `.wav` deliverables |

## Design choices
* **Whitening** (`Z = D^{-1/2} Eᵀ Xc`) makes Cov(Z) = I, so the remaining unmixing matrix is a pure *rotation* — `N(N-1)/2` unknowns instead of `N²`.
* **FastICA update:** `w⁺ = E[z g(wᵀz)] − E[g′(wᵀz)] w`, then symmetric orthogonalisation `W ← (WWᵀ)^{-1/2} W` to stop two units locking onto one source.
* **Contrast `logcosh`** (default) is robust for speech/music (super-Gaussian); `cube` is kurtosis-based and outlier-sensitive.
* **Ambiguities:** ICA cannot know source order, sign or loudness. Evaluation uses Hungarian matching on |corr|, then a least-squares gain per source (handles sign and scale).
* **Metrics:** Pearson |correlation| and scale-invariant SDR `10 log10(‖s‖² / ‖s − ŝ‖²)` after alignment. SDR is sensitive to residual interference *and* noise; correlation is quick and intuitive. Both are reported.
* **Limits:** instantaneous (non-convolutive) mixing, ≤1 Gaussian source, at least as many mixtures as sources. Real room recordings have delays/reverb (convolutive mixing) — this violates the model and degrades results.

## Results (demo sources, 3×3, 30 dB sensor noise)
| Method  | mean \|corr\| | mean SDR (dB) |
|---------|------|------|
| PCA     | 0.74 | 3.5  |
| FastICA | 0.998 | 25.0 |

See `outputs/plots/` (waveforms, correlation matrices, SNR sweep) and `outputs/metrics.json`. Re-run with your real sources and update this table.
