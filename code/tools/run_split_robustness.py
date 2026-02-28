#!/usr/bin/env python3
import csv
import hashlib
import os
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = os.path.join(ROOT, "results", "split_robustness_raw.csv")

SPLIT_SEEDS = [2026, 2027, 2028]
MODEL_SEEDS = [11, 23, 37, 41, 53]
N_CLASSES = 10
WINDOW_S = 0.12

CONDITIONS = {
    "hybrid_default_norm_on": {
        "k": 4,
        "sigma": 0.25,
        "lambda_max_hz": 200.0,
        "homeostasis": "on",
        "reward_shaping": "signed",
    },
    "hybrid_best_norm_off": {
        "k": 4,
        "sigma": 0.25,
        "lambda_max_hz": 200.0,
        "homeostasis": "off",
        "reward_shaping": "signed",
    },
}


def _one_hot(y, n_classes):
    out = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def _softmax(logits):
    z = logits - np.max(logits)
    expz = np.exp(z)
    return expz / np.sum(expz)


def _accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred) * 100.0)


def _macro_f1(y_true, y_pred, n_classes=N_CLASSES):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        if tp == 0 and (fp > 0 or fn > 0):
            f1s.append(0.0)
            continue
        if tp == 0 and fp == 0 and fn == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
        f1s.append(float(f1))
    return float(np.mean(f1s))


def _population_encode(x, k, sigma, lambda_max_hz, rng):
    centers = np.linspace(0.0, 1.0, k, dtype=np.float32)[None, None, :]
    x3 = x[:, :, None]
    rates = lambda_max_hz * np.exp(-((x3 - centers) ** 2) / (2.0 * sigma * sigma))
    lam = rates * WINDOW_S
    counts = rng.poisson(lam).astype(np.float32)
    return counts.reshape(x.shape[0], x.shape[1] * k)


def _split_data(split_seed):
    digits = load_digits()
    x = (digits.data.astype(np.float32) / 16.0).clip(0.0, 1.0)
    y = digits.target.astype(np.int64)
    idx_all = np.arange(x.shape[0], dtype=np.int64)
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=split_seed
    )
    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=0.2, stratify=y, random_state=split_seed
    )
    x_train, _x_val, y_train, _y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=split_seed
    )
    return x_train, y_train, x_test, y_test, idx_test


def _fit_hybrid_local_readout(
    x_train,
    y_train,
    x_test,
    y_test,
    seed,
    k,
    sigma,
    lambda_max_hz,
    homeostasis,
    reward_shaping,
):
    rng = np.random.default_rng(seed)
    xtr = _population_encode(x_train, k=k, sigma=sigma, lambda_max_hz=lambda_max_hz, rng=rng)
    xte = _population_encode(x_test, k=k, sigma=sigma, lambda_max_hz=lambda_max_hz, rng=rng)

    d = xtr.shape[1]
    w = rng.normal(0.0, 0.01, size=(N_CLASSES, d)).astype(np.float32)
    b = np.zeros(N_CLASSES, dtype=np.float32)
    ytr_oh = _one_hot(y_train, N_CLASSES)

    lr = 0.003
    epochs = 18
    homeo_decay = 0.98

    t0 = time.perf_counter()
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
        if homeostasis == "on":
            row_norm = np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
            w = homeo_decay * w / row_norm
    train_time_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    logits = xte @ w.T + b[None, :]
    y_pred = np.argmax(logits, axis=1)
    infer_time_ms = (time.perf_counter() - t1) * 1000.0 / xte.shape[0]

    return (
        _accuracy(y_test, y_pred),
        _macro_f1(y_test, y_pred),
        float(train_time_s),
        float(infer_time_ms),
    )


def _write_rows(path, rows):
    fieldnames = [
        "split_seed",
        "condition",
        "model_seed",
        "test_accuracy",
        "macro_f1",
        "train_time_s",
        "infer_time_ms_per_sample",
        "infer_excludes_encoding",
        "infer_scope",
        "notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cfg_hash(cfg):
    items = sorted(cfg.items())
    return hashlib.sha256(repr(items).encode("utf-8")).hexdigest()


def main():
    rows = []
    expected_seed_set = set(MODEL_SEEDS)
    split_test_sizes = {}
    cfg_hash_by_condition = {condition: _cfg_hash(cfg) for condition, cfg in CONDITIONS.items()}

    for split_seed in SPLIT_SEEDS:
        x_train, y_train, x_test, y_test, idx_test = _split_data(split_seed)
        split_test_sizes[split_seed] = len(idx_test)
        assert len(np.unique(idx_test)) == len(idx_test), (
            f"Split {split_seed}: duplicate test indices detected"
        )
        if split_seed == SPLIT_SEEDS[0]:
            expected_test_size = len(idx_test)
        assert len(idx_test) == expected_test_size, (
            f"Split {split_seed}: unexpected test size {len(idx_test)} != {expected_test_size}"
        )

        for condition, cfg in CONDITIONS.items():
            # Guardrail: no split-specific tuning.
            assert _cfg_hash(cfg) == cfg_hash_by_condition[condition], (
                f"Split {split_seed}, condition {condition}: hyperparameters changed"
            )
            seen_model_seeds = set()
            for model_seed in MODEL_SEEDS:
                seen_model_seeds.add(model_seed)
                acc, macro_f1, train_s, infer_ms = _fit_hybrid_local_readout(
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    seed=model_seed,
                    k=cfg["k"],
                    sigma=cfg["sigma"],
                    lambda_max_hz=cfg["lambda_max_hz"],
                    homeostasis=cfg["homeostasis"],
                    reward_shaping=cfg["reward_shaping"],
                )
                rows.append(
                    {
                        "split_seed": str(split_seed),
                        "condition": condition,
                        "model_seed": str(model_seed),
                        "test_accuracy": f"{acc:.6f}",
                        "macro_f1": f"{macro_f1:.6f}",
                        "train_time_s": f"{train_s:.6f}",
                        "infer_time_ms_per_sample": f"{infer_ms:.6f}",
                        "infer_excludes_encoding": "true",
                        "infer_scope": "forward pass only; excludes encoding and data loading",
                        "notes": "phase_strong_accept_split_robustness",
                    }
                )
            assert seen_model_seeds == expected_seed_set, (
                f"Split {split_seed}, condition {condition}: model seed set mismatch"
            )

    _write_rows(OUT_CSV, rows)
    print(f"Wrote {OUT_CSV} ({len(rows)} rows)")
    print(f"Per-split test sizes: {split_test_sizes}")


if __name__ == "__main__":
    main()
