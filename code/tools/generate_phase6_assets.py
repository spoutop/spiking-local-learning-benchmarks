#!/usr/bin/env python3
import csv
import os
from collections import defaultdict

# Writable local caches for matplotlib/fontconfig in restricted environments.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, ".cache")
MPL_CACHE_DIR = os.path.join(CACHE_DIR, "matplotlib")
FC_CACHE_DIR = os.path.join(CACHE_DIR, "fontconfig")
for _d in (CACHE_DIR, MPL_CACHE_DIR, FC_CACHE_DIR):
    os.makedirs(_d, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", CACHE_DIR)
os.environ.setdefault("MPLCONFIGDIR", MPL_CACHE_DIR)
os.environ.setdefault("FC_CACHEDIR", FC_CACHE_DIR)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.datasets import load_digits
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


FIGS_DIR = os.path.join(ROOT, "figs")
TABLES_DIR = os.path.join(ROOT, "tables")
RESULTS_DIR = os.path.join(ROOT, "results")

SPLIT_SEED = 2026
SEEDS = [11, 23, 37, 41, 53]
N_CLASSES = 10
WINDOW_S = 0.12


def _one_hot(y, n_classes):
    out = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def _softmax(logits):
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def _population_encode(x, k, sigma, lambda_max_hz, rng):
    centers = np.linspace(0.0, 1.0, k, dtype=np.float32)[None, None, :]
    x3 = x[:, :, None]
    rates = lambda_max_hz * np.exp(-((x3 - centers) ** 2) / (2.0 * sigma * sigma))
    lam = rates * WINDOW_S
    counts = rng.poisson(lam).astype(np.float32)
    return counts.reshape(x.shape[0], x.shape[1] * k)


def _train_hybrid_default(x_train, y_train, x_test, seed):
    rng = np.random.default_rng(seed)
    xtr = _population_encode(x_train, k=4, sigma=0.25, lambda_max_hz=200.0, rng=rng)
    xte = _population_encode(x_test, k=4, sigma=0.25, lambda_max_hz=200.0, rng=rng)
    d = xtr.shape[1]
    w = rng.normal(0.0, 0.01, size=(N_CLASSES, d)).astype(np.float32)
    b = np.zeros(N_CLASSES, dtype=np.float32)
    ytr_oh = _one_hot(y_train, N_CLASSES)
    lr = 0.003
    epochs = 18
    homeo_decay = 0.98

    for _ in range(epochs):
        idx = rng.permutation(xtr.shape[0])
        for i in idx:
            r = xtr[i]
            y = ytr_oh[i]
            p = _softmax(w @ r + b)
            delta = y - p
            w += lr * delta[:, None] * r[None, :]
            b += lr * delta
        row_norm = np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
        w = homeo_decay * w / row_norm

    y_pred = np.argmax(xte @ w.T + b[None, :], axis=1)
    spikes_per_sample = np.sum(xte, axis=1)
    return y_pred, spikes_per_sample


def _split_data():
    digits = load_digits()
    x = (digits.data.astype(np.float32) / 16.0).clip(0.0, 1.0)
    y = digits.target.astype(np.int64)
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SPLIT_SEED
    )
    x_train, _x_val, y_train, _y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=SPLIT_SEED
    )
    return x_train, y_train, x_test, y_test


def _read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.nan, np.nan
    return float(np.mean(arr)), float(np.std(arr, ddof=1) if arr.size > 1 else 0.0)


def build_extra_metrics_table():
    x_train, y_train, x_test, y_test = _split_data()
    per_class_f1_by_seed = []
    mean_spikes = []

    for seed in SEEDS:
        y_pred, spikes = _train_hybrid_default(x_train, y_train, x_test, seed)
        f1_pc = f1_score(y_test, y_pred, average=None, labels=np.arange(N_CLASSES))
        per_class_f1_by_seed.append(f1_pc)
        mean_spikes.append(float(np.mean(spikes)))

    f1_mat = np.vstack(per_class_f1_by_seed)
    f1_mean = np.mean(f1_mat, axis=0)
    worst_idx = np.argsort(f1_mean)[:2]

    baseline_rows = _read_csv(os.path.join(RESULTS_DIR, "baselines_raw.csv"))
    by_method = defaultdict(list)
    for r in baseline_rows:
        by_method[r["method"]].append(r)

    def _col_mean(method, col):
        vals = [float(r[col]) for r in by_method.get(method, []) if r.get(col, "").strip()]
        return float(np.mean(vals)) if vals else float("nan")

    def _col_mean_std(method, col):
        vals = [float(r[col]) for r in by_method.get(method, []) if r.get(col, "").strip()]
        if not vals:
            return float("nan"), float("nan")
        arr = np.asarray(vals, dtype=np.float64)
        return float(np.mean(arr)), float(np.std(arr, ddof=1) if arr.size > 1 else 0.0)

    hybrid_params = 10 * (64 * 4) + 10
    stdp_params = 96 * (64 * 4) + 96
    hybrid_inf_forward_ms = _col_mean(
        "hybrid_encoder_local_readout", "infer_forward_median_ms_per_sample"
    )
    stdp_inf_forward_ms = _col_mean("pure_stdp_ei", "infer_forward_median_ms_per_sample")
    # Fallback for backward compatibility with older CSVs.
    if np.isnan(hybrid_inf_forward_ms):
        hybrid_inf_forward_ms = _col_mean("hybrid_encoder_local_readout", "infer_time_ms_per_sample")
    if np.isnan(stdp_inf_forward_ms):
        stdp_inf_forward_ms = _col_mean("pure_stdp_ei", "infer_time_ms_per_sample")
    hybrid_inf_e2e_ms = _col_mean(
        "hybrid_encoder_local_readout", "infer_end_to_end_median_ms_per_sample"
    )
    stdp_inf_e2e_ms = _col_mean("pure_stdp_ei", "infer_end_to_end_median_ms_per_sample")
    timing_repeats = _col_mean("hybrid_encoder_local_readout", "infer_timing_repeats")
    stdp_sat_low_m, stdp_sat_low_s = _col_mean_std("pure_stdp_ei", "stdp_weight_saturation_pct")
    stdp_sat_up_m, stdp_sat_up_s = _col_mean_std(
        "pure_stdp_ei", "stdp_weight_saturation_upper_pct"
    )
    stdp_margin_m, stdp_margin_s = _col_mean_std("pure_stdp_ei", "stdp_winner_margin_mean")
    spike_m, spike_s = _mean_std(mean_spikes)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Additional SNN diagnostics (hybrid default; STDP rows explicitly marked). Inference timing is reported as median over 100 repeats on full-test vectorized batches ($N=360$): forward-only excludes encoding and I/O, while end-to-end includes encoding and forward pass; no learning/update steps are included. Parameter counts include model weights and threshold terms; STDP-style counts also include excitatory prototype vectors ($96\times256$).}"
    )
    lines.append(r"  \scriptsize")
    lines.append(r"  \setlength{\tabcolsep}{3pt}")
    lines.append(r"  \resizebox{\linewidth}{!}{%")
    lines.append(r"  \begin{tabular}{lc}")
    lines.append(r"    \toprule")
    lines.append(r"    Metric & Value \\")
    lines.append(r"    \midrule")
    lines.append(
        f"    Hybrid default: worst class F1 (digit {int(worst_idx[0])}) & {f1_mean[worst_idx[0]]:.2f} \\\\"
    )
    lines.append(
        f"    Hybrid default: 2nd worst class F1 (digit {int(worst_idx[1])}) & {f1_mean[worst_idx[1]]:.2f} \\\\"
    )
    lines.append(
        f"    Hybrid default: input spikes/sample (mean $\\pm$ std across seeds) & {spike_m:.1f} $\\pm$ {spike_s:.1f} \\\\"
    )
    lines.append(
        f"    STDP branch: weight saturation (\\% at lower bound) & {stdp_sat_low_m:.2f} $\\pm$ {stdp_sat_low_s:.2f} \\\\"
    )
    lines.append(
        f"    STDP branch: weight saturation (\\% at upper bound) & {stdp_sat_up_m:.2f} $\\pm$ {stdp_sat_up_s:.2f} \\\\"
    )
    lines.append(
        f"    STDP branch: winner margin (activation units, normalized dot-product) & {stdp_margin_m:.4f} $\\pm$ {stdp_margin_s:.4f} \\\\"
    )
    lines.append(f"    Hybrid parameter count & {hybrid_params} \\\\")
    lines.append(f"    STDP-style parameter count & {stdp_params} \\\\")
    rep = int(timing_repeats) if not np.isnan(timing_repeats) else 100
    lines.append(
        f"    Hybrid timing: forward-only median ($\\mu$s/sample, {rep} repeats) & {hybrid_inf_forward_ms * 1000.0:.3f} \\\\"
    )
    lines.append(
        f"    Hybrid timing: end-to-end median ($\\mu$s/sample, {rep} repeats) & {hybrid_inf_e2e_ms * 1000.0:.3f} \\\\"
    )
    lines.append(
        f"    STDP timing: forward-only median ($\\mu$s/sample, {rep} repeats) & {stdp_inf_forward_ms * 1000.0:.3f} \\\\"
    )
    lines.append(
        f"    STDP timing: end-to-end median ($\\mu$s/sample, {rep} repeats) & {stdp_inf_e2e_ms * 1000.0:.3f} \\\\"
    )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}%")
    lines.append(r"  }")
    lines.append(r"  \label{tab:extra_metrics}")
    lines.append(r"\end{table}")
    lines.append("")

    out_path = os.path.join(TABLES_DIR, "extra_metrics.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


def build_architecture_figure():
    os.makedirs(FIGS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, txt, fc):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.0,
            edgecolor="black",
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=1.1)
        ax.add_patch(a)

    box(0.03, 0.38, 0.14, 0.24, "8x8 Input\npixels", "#e8f1ff")
    box(0.23, 0.38, 0.17, 0.24, "Population\nPoisson encoder\n(K=4)", "#e9f8ef")
    box(0.46, 0.56, 0.18, 0.20, "E1/I1\ncompetition\n(local STDP)", "#fff4d6")
    box(0.46, 0.20, 0.18, 0.20, "Hybrid rate\nreadout\n(local delta)", "#ffe5e5")
    box(0.72, 0.56, 0.20, 0.20, "E2/I2 decision\nreward-modulated\nupdates", "#f3e8ff")
    box(0.72, 0.20, 0.20, 0.20, "Softmax decision\n(rate-based; no timing\ncredit assignment)", "#f3e8ff")

    arrow(0.17, 0.50, 0.23, 0.50)
    arrow(0.40, 0.50, 0.46, 0.66)
    arrow(0.40, 0.50, 0.46, 0.30)
    arrow(0.64, 0.66, 0.72, 0.66)
    arrow(0.64, 0.30, 0.72, 0.30)
    ax.text(0.55, 0.84, "Timing-based STDP branch (competitive proxy)", fontsize=8, ha="center")
    ax.text(0.55, 0.08, "Practical rate-readout benchmark branch", fontsize=8, ha="center")

    out_pdf = os.path.join(FIGS_DIR, "architecture_diagram.pdf")
    out_png = os.path.join(FIGS_DIR, "architecture_diagram.png")
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def main():
    build_extra_metrics_table()
    build_architecture_figure()


if __name__ == "__main__":
    main()
