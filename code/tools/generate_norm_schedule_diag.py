#!/usr/bin/env python3
import os

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
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


OUT_FIG = os.path.join(ROOT, "figs", "norm_schedule_diag.png")

SEEDS = [11, 23, 37, 41, 53]
SPLIT_SEED = 2026
N_CLASSES = 10


def _softmax(logits):
    z = logits - np.max(logits)
    expz = np.exp(z)
    return expz / np.sum(expz)


def _one_hot(y, n_classes):
    out = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def _population_encode(x, k, sigma, lambda_max_hz, rng):
    centers = np.linspace(0.0, 1.0, k, dtype=np.float32)[None, None, :]
    x3 = x[:, :, None]
    rates = lambda_max_hz * np.exp(-((x3 - centers) ** 2) / (2.0 * sigma * sigma))
    lam = rates * 0.12
    counts = rng.poisson(lam).astype(np.float32)
    return counts.reshape(x.shape[0], x.shape[1] * k)


def _split_data():
    digits = load_digits()
    x = (digits.data.astype(np.float32) / 16.0).clip(0.0, 1.0)
    y = digits.target.astype(np.int64)
    x_trainval, _x_test, y_trainval, _y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SPLIT_SEED
    )
    x_train, _x_val, y_train, _y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=SPLIT_SEED
    )
    return x_train, y_train


def _trace_row_norms(homeostasis_mode):
    x_train, y_train = _split_data()
    epochs = 18
    lr = 0.003
    traces = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        xtr = _population_encode(x_train, k=4, sigma=0.25, lambda_max_hz=200.0, rng=rng)
        ytr_oh = _one_hot(y_train, N_CLASSES)
        w = rng.normal(0.0, 0.01, size=(N_CLASSES, xtr.shape[1])).astype(np.float32)
        b = np.zeros(N_CLASSES, dtype=np.float32)
        epoch_trace = []
        for epoch_idx in range(epochs):
            idx = rng.permutation(xtr.shape[0])
            for i in idx:
                r = xtr[i]
                y = ytr_oh[i]
                p = _softmax(w @ r + b)
                delta = y - p
                w += lr * delta[:, None] * r[None, :]
                b += lr * delta

            if homeostasis_mode == "on":
                row_norm = np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
                w = 0.98 * w / row_norm
            elif homeostasis_mode == "gentle" and (epoch_idx + 1) % 5 == 0:
                row_norm = np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
                w = 0.995 * w / row_norm

            epoch_trace.append(float(np.mean(np.linalg.norm(w, axis=1))))
        traces.append(epoch_trace)
    arr = np.asarray(traces, dtype=np.float64)
    return arr.mean(axis=0), arr.std(axis=0)


def main():
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    x_axis = np.arange(1, 19)
    curves = {
        "norm on": _trace_row_norms("on"),
        "norm gentle": _trace_row_norms("gentle"),
        "norm off": _trace_row_norms("off"),
    }

    plt.figure(figsize=(6.0, 3.2))
    colors = {"norm on": "#1f77b4", "norm gentle": "#ff7f0e", "norm off": "#2ca02c"}
    for label, (m, s) in curves.items():
        plt.plot(x_axis, m, label=label, linewidth=2.0, color=colors[label])
        plt.fill_between(x_axis, m - s, m + s, alpha=0.18, color=colors[label], linewidth=0)

    plt.xlabel("Epoch")
    plt.ylabel("Mean class-row norm")
    plt.title("Normalization schedule diagnostic (mean ± std across seeds)")
    plt.grid(alpha=0.2)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=180)
    plt.close()
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
