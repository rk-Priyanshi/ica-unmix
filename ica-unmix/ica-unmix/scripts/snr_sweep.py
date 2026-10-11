"""Experiment: separation quality vs. sensor-noise SNR, ICA vs PCA, several random mixing matrices."""
from pathlib import Path
import sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from unmix import load_sources, make_mixing_matrix, mix, add_noise, ica_separate, pca_separate, evaluate

paths = sorted(glob.glob(str(ROOT / "data" / "sources" / "*.wav")))
S, sr = load_sources(paths, max_seconds=10)
N = len(paths)
snrs = [0, 5, 10, 15, 20, 30, 40, None]          # None = noiseless
trials = 5
results = {"snr": [], "ica_sdr": [], "pca_sdr": [], "ica_corr": [], "pca_corr": []}

for snr in snrs:
    ica_s, pca_s, ica_c, pca_c = [], [], [], []
    for seed in range(trials):
        A = make_mixing_matrix(N, seed=seed)
        X = add_noise(mix(S, A), snr, seed=seed)
        e_i = evaluate(ica_separate(X, N, seed=seed)["S_hat"], S)
        e_p = evaluate(pca_separate(X, N), S)
        ica_s.append(e_i["mean_sdr_db"]); pca_s.append(e_p["mean_sdr_db"])
        ica_c.append(e_i["mean_corr"]);   pca_c.append(e_p["mean_corr"])
    results["snr"].append("inf" if snr is None else snr)
    results["ica_sdr"].append(float(np.mean(ica_s))); results["pca_sdr"].append(float(np.mean(pca_s)))
    results["ica_corr"].append(float(np.mean(ica_c))); results["pca_corr"].append(float(np.mean(pca_c)))
    print(f"SNR={snr!s:>5}  ICA SDR={np.mean(ica_s):6.2f} dB  PCA SDR={np.mean(pca_s):6.2f} dB")

(ROOT / "outputs").mkdir(exist_ok=True)
(ROOT / "outputs" / "snr_sweep.json").write_text(json.dumps(results, indent=2))
x = np.arange(len(snrs)); labels = [str(s) if s is not None else "inf" for s in snrs]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, key, ylab in zip(ax, ("sdr", "corr"), ("mean SDR (dB)", "mean |correlation|")):
    a.plot(x, results[f"ica_{key}"], "o-", label="FastICA")
    a.plot(x, results[f"pca_{key}"], "s--", label="PCA")
    a.set_xticks(x, labels); a.set_xlabel("input SNR (dB)"); a.set_ylabel(ylab); a.grid(alpha=.3); a.legend()
fig.suptitle(f"Separation quality vs SNR (mean of {trials} random mixing matrices)")
fig.tight_layout(); fig.savefig(ROOT / "outputs" / "plots" / "snr_sweep.png", dpi=130)
