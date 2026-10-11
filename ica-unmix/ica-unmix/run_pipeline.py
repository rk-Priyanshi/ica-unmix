"""End-to-end pipeline: load -> mix -> center -> whiten -> FastICA -> align -> save -> evaluate -> plots.

Example:
    python run_pipeline.py --sources data/sources/*.wav --snr 30
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from unmix import (load_sources, save_wav, make_mixing_matrix, mix, add_noise,
                   ica_separate, pca_separate, evaluate)
from unmix.evaluation import corr_matrix
from unmix.plotting import plot_waveforms, plot_corr_matrices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True, help="N clean WAV files")
    ap.add_argument("--mics", type=int, default=None, help="number of mixtures (>= N). default N")
    ap.add_argument("--snr", type=float, default=None, help="sensor-noise SNR in dB (default: noiseless)")
    ap.add_argument("--fun", default="logcosh", choices=["logcosh", "exp", "cube"])
    ap.add_argument("--algorithm", default="parallel", choices=["parallel", "deflation"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    out = Path(a.out)
    sep_dir, plot_dir = out / "separated", out / "plots"
    mixed_dir = Path("data/mixed")

    # 1. Load + mix ------------------------------------------------------------------
    S, sr = load_sources(a.sources, max_seconds=a.max_seconds)
    N = S.shape[0]
    A = make_mixing_matrix(N, a.mics, seed=a.seed)
    X = add_noise(mix(S, A), a.snr, seed=a.seed)
    for i, x in enumerate(X):
        save_wav(mixed_dir / f"mixed_{i+1}.wav", x, sr)
    print(f"{N} sources, {X.shape[0]} mixtures, {S.shape[1]/sr:.1f}s @ {sr} Hz, cond(A)={np.linalg.cond(A):.1f}")

    # 2-3. Center + whiten + FastICA --------------------------------------------------
    res = ica_separate(X, n_components=N, fun=a.fun, algorithm=a.algorithm, seed=a.seed)
    print(f"FastICA ({a.algorithm}, {a.fun}) converged in {res['n_iter']} iterations")

    # 4. Evaluate ICA vs PCA ------------------------------------------------------------
    ica_eval = evaluate(res["S_hat"], S)
    pca_eval = evaluate(pca_separate(X, N), S)

    # 5. Export aligned separated audio -----------------------------------------------
    names = [Path(p).stem for p in a.sources]
    for j, name in enumerate(names):
        save_wav(sep_dir / f"separated_{name}.wav", ica_eval["aligned"][j], sr)

    metrics = {
        "config": vars(a) | {"sample_rate": sr, "cond_A": float(np.linalg.cond(A))},
        "ica": {k: ica_eval[k] for k in ("perm", "corr", "sdr_db", "mean_corr", "mean_sdr_db")},
        "pca": {k: pca_eval[k] for k in ("perm", "corr", "sdr_db", "mean_corr", "mean_sdr_db")},
        "sources": names,
    }
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n           mean |corr|   mean SDR (dB)")
    print(f"  PCA      {pca_eval['mean_corr']:10.3f}   {pca_eval['mean_sdr_db']:10.2f}")
    print(f"  FastICA  {ica_eval['mean_corr']:10.3f}   {ica_eval['mean_sdr_db']:10.2f}")
    for n, c, s in zip(names, ica_eval["corr"], ica_eval["sdr_db"]):
        print(f"    {n:<16} corr={c:.3f}  SDR={s:6.2f} dB")

    # 6. Plots -----------------------------------------------------------------------
    plot_waveforms(S, X, ica_eval["aligned"], sr, names, plot_dir / "waveforms.png")
    plot_corr_matrices(corr_matrix(pca_separate(X, N), S), corr_matrix(res["S_hat"], S), names,
                       plot_dir / "correlation_pca_vs_ica.png")
    print(f"\nSaved audio -> {sep_dir}, plots -> {plot_dir}, metrics -> {out/'metrics.json'}")


if __name__ == "__main__":
    main()
