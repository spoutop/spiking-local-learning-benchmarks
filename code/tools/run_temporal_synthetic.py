#!/usr/bin/env python3
import csv
import os
import time

import numpy as np
from sklearn.model_selection import train_test_split


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = os.path.join(ROOT, "results", "temporal_synthetic_raw.csv")

SEEDS = [11, 23, 37, 41, 53]
SPLIT_SEED = 2026
N_CLASSES = 2
TIMING_REPEATS = 100


def _format_float(x):
    return f"{x:.6f}"


def _softmax(logits):
    z = logits - np.max(logits)
    expz = np.exp(z)
    return expz / np.sum(expz)


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


def _accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred) * 100.0)


def _median_timing_ms_per_sample(num_samples, repeats, fn):
    n = max(1, int(num_samples))
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        runs.append((time.perf_counter() - t0) * 1000.0 / n)
    return float(np.median(np.asarray(runs, dtype=np.float64)))


def _one_hot(y, n_classes):
    out = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def _generate_sequences(seed, n_samples=2400, t_steps=20, n_channels=2):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n_samples, dtype=np.int64)
    x = np.zeros((n_samples, t_steps, n_channels), dtype=np.float32)

    early = slice(2, 8)
    late = slice(12, 18)
    base_rate = 0.03
    high_rate = 0.25
    low_rate = 0.07

    for i in range(n_samples):
        if y[i] == 0:
            # class 0: channel 0 early, channel 1 late
            rates = np.full((t_steps, n_channels), base_rate, dtype=np.float32)
            rates[early, 0] = high_rate
            rates[late, 1] = high_rate
            rates[late, 0] = low_rate
            rates[early, 1] = low_rate
        else:
            # class 1: channel 1 early, channel 0 late
            rates = np.full((t_steps, n_channels), base_rate, dtype=np.float32)
            rates[early, 1] = high_rate
            rates[late, 0] = high_rate
            rates[late, 1] = low_rate
            rates[early, 0] = low_rate
        x[i] = rng.poisson(rates).astype(np.float32)

    return x, y


def _split_data(x, y):
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SPLIT_SEED
    )
    x_train, _x_val, y_train, _y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=SPLIT_SEED
    )
    return x_train, y_train, x_test, y_test


def _features_count(x):
    return np.sum(x, axis=1)


def _features_timebin(x, n_bins=4):
    n_samples, t_steps, n_channels = x.shape
    bin_size = t_steps // n_bins
    feats = []
    for b in range(n_bins):
        s = b * bin_size
        e = (b + 1) * bin_size if b < n_bins - 1 else t_steps
        feats.append(np.sum(x[:, s:e, :], axis=1))
    out = np.concatenate(feats, axis=1)
    return out.reshape(n_samples, n_channels * n_bins)


def _fit_local_readout(x_train, y_train, x_test, y_test, seed):
    rng = np.random.default_rng(seed)
    d = x_train.shape[1]
    w = rng.normal(0.0, 0.01, size=(N_CLASSES, d)).astype(np.float32)
    b = np.zeros(N_CLASSES, dtype=np.float32)
    ytr_oh = _one_hot(y_train, N_CLASSES)
    lr = 0.02
    epochs = 12

    t0 = time.perf_counter()
    for _ in range(epochs):
        idx = rng.permutation(x_train.shape[0])
        for i in idx:
            r = x_train[i]
            y = ytr_oh[i]
            p = _softmax(w @ r + b)
            delta = y - p
            w += lr * delta[:, None] * r[None, :]
            b += lr * delta
    train_time_s = time.perf_counter() - t0

    logits = x_test @ w.T + b[None, :]
    y_pred = np.argmax(logits, axis=1)
    acc = _accuracy(y_test, y_pred)
    macro_f1 = _macro_f1(y_test, y_pred)
    infer_ms = _median_timing_ms_per_sample(
        num_samples=x_test.shape[0], repeats=TIMING_REPEATS, fn=lambda: x_test @ w.T + b[None, :]
    )
    return acc, macro_f1, train_time_s, infer_ms


def _run_condition(condition, feat_fn):
    rows = []
    for seed in SEEDS:
        x, y = _generate_sequences(seed=seed)
        x_train, y_train, x_test, y_test = _split_data(x, y)
        f_train = feat_fn(x_train)
        f_test = feat_fn(x_test)
        acc, f1, train_s, infer_ms = _fit_local_readout(f_train, y_train, f_test, y_test, seed=seed)
        rows.append(
            {
                "benchmark": "temporal_order_synthetic",
                "condition": condition,
                "seed": str(seed),
                "test_accuracy": _format_float(acc),
                "macro_f1": _format_float(f1),
                "train_time_s": _format_float(train_s),
                "infer_time_ms_per_sample": _format_float(infer_ms),
                "notes": "phase_temporal_synthetic",
            }
        )
    return rows


def main():
    rows = []
    rows.extend(_run_condition("count_readout_local", _features_count))
    rows.extend(_run_condition("timebin_readout_local", _features_timebin))
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "benchmark",
                "condition",
                "seed",
                "test_accuracy",
                "macro_f1",
                "train_time_s",
                "infer_time_ms_per_sample",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
