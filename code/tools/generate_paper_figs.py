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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


FIGS_DIR = os.path.join(ROOT, "figs")
SPLIT_SEED = 2026
N_CLASSES = 10


def _softmax(logits):
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def _one_hot(y, n_classes):
    out = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def _population_encode(x, k, sigma, lambda_max_hz, window_s, rng):
    centers = np.linspace(0.0, 1.0, k, dtype=np.float32)[None, None, :]
    x3 = x[:, :, None]
    rates = lambda_max_hz * np.exp(-((x3 - centers) ** 2) / (2.0 * sigma * sigma))
    lam = rates * window_s
    counts = rng.poisson(lam).astype(np.float32)
    return counts.reshape(x.shape[0], x.shape[1] * k)


def train_hybrid_with_curve(
    x_train,
    y_train,
    x_test,
    y_test,
    seed=23,
    k=4,
    sigma=0.25,
    lambda_max_hz=200.0,
    window_s=0.12,
    homeostasis=True,
    reward_shaping="signed",
    epochs=18,
    lr=0.003,
):
    rng = np.random.default_rng(seed)
    xtr = _population_encode(x_train, k, sigma, lambda_max_hz, window_s, rng)
    xte = _population_encode(x_test, k, sigma, lambda_max_hz, window_s, rng)
    d = xtr.shape[1]
    w = rng.normal(0.0, 0.01, size=(N_CLASSES, d)).astype(np.float32)
    b = np.zeros(N_CLASSES, dtype=np.float32)
    ytr_oh = _one_hot(y_train, N_CLASSES)

    train_acc = []
    test_acc = []
    homeo_decay = 0.98

    for _ in range(epochs):
        idx = rng.permutation(xtr.shape[0])
        for i in idx:
            r = xtr[i]
            y = ytr_oh[i]
            p = _softmax(w @ r + b)
            if reward_shaping == "signed":
                delta = y - p
            else:
                delta = y * (1.0 - p)
            w += lr * delta[:, None] * r[None, :]
            b += lr * delta

        if homeostasis:
            row_norm = np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
            w = homeo_decay * w / row_norm

        y_pred_train = np.argmax(xtr @ w.T + b[None, :], axis=1)
        y_pred_test = np.argmax(xte @ w.T + b[None, :], axis=1)
        train_acc.append(np.mean(y_pred_train == y_train) * 100.0)
        test_acc.append(np.mean(y_pred_test == y_test) * 100.0)

    final_pred = np.argmax(xte @ w.T + b[None, :], axis=1)
    return np.array(train_acc), np.array(test_acc), final_pred


def main():
    os.makedirs(FIGS_DIR, exist_ok=True)
    digits = load_digits()
    x = (digits.data.astype(np.float32) / 16.0).clip(0.0, 1.0)
    y = digits.target.astype(np.int64)
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SPLIT_SEED
    )
    x_train, _, y_train, _ = train_test_split(
        x_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=SPLIT_SEED
    )

    train_acc, test_acc, y_pred = train_hybrid_with_curve(
        x_train, y_train, x_test, y_test
    )
    train_acc_off, test_acc_off, _y_pred_off = train_hybrid_with_curve(
        x_train, y_train, x_test, y_test, homeostasis=False
    )

    # Accuracy curve
    plt.figure(figsize=(7, 4.2))
    epochs = np.arange(1, len(train_acc) + 1)
    plt.plot(epochs, train_acc, label="Default (norm-on) train", linewidth=2.0)
    plt.plot(epochs, test_acc, label="Default (norm-on) test", linewidth=2.0)
    plt.plot(
        epochs,
        test_acc_off,
        linestyle="--",
        linewidth=2.0,
        label="Norm-off test",
    )
    plt.plot(
        epochs,
        train_acc_off,
        linestyle="--",
        linewidth=1.6,
        alpha=0.7,
        label="Norm-off train",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Hybrid Local-Readout Accuracy Curves (Default vs Norm-off)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "accuracy.png"), dpi=180)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(10), normalize="true")
    plt.figure(figsize=(6.4, 5.4))
    im = plt.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Normalized Confusion Matrix (test n={y_test.shape[0]})")
    for i in range(10):
        for j in range(10):
            txt = f"{cm[i, j]:.2f}"
            color = "white" if cm[i, j] > 0.5 else "black"
            plt.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "confusion.png"), dpi=180)
    plt.close()

    # Sample predictions
    rng = np.random.default_rng(123)
    idx = rng.choice(np.arange(x_test.shape[0]), size=25, replace=False)
    fig, axes = plt.subplots(5, 5, figsize=(9.5, 8.6))
    for ax, ii in zip(axes.ravel(), idx):
        img = (x_test[ii] * 16.0).reshape(8, 8)
        pred = int(y_pred[ii])
        true = int(y_test[ii])
        ok = pred == true
        ax.imshow(img, cmap="gray_r", interpolation="nearest")
        ax.set_title(f"T:{true} P:{pred}", fontsize=9, color=("green" if ok else "red"))
        ax.axis("off")
    fig.suptitle("Random Test Samples with Predictions", y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "samples.png"), dpi=180)
    plt.close()

    print(f"Wrote figures to {FIGS_DIR}")


if __name__ == "__main__":
    main()
