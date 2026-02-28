#!/usr/bin/env python3
import csv
import os
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES_CSV = os.path.join(ROOT, "results", "baselines_raw.csv")
ABLATIONS_CSV = os.path.join(ROOT, "results", "ablations_raw.csv")

SEEDS = [11, 23, 37, 41, 53]
ROBUST_ABLATION_EXTRA_SEEDS = [67, 79, 83, 97]
ROBUST_ABLATION_SEEDS = SEEDS + ROBUST_ABLATION_EXTRA_SEEDS
SPLIT_SEED = 2026
N_CLASSES = 10
WINDOW_S = 0.12
TIMING_REPEATS = 100
TIMING_BATCHING = "full_test_vectorized"


def _load_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _upsert_baseline(rows, payload):
    method = payload["method"]
    seed = str(payload["seed"])
    for row in rows:
        if row.get("method") == method and str(row.get("seed", "")).strip() == seed:
            row.update(payload)
            return
    rows.append(payload)


def _upsert_ablation(rows, payload):
    factor = payload["factor"]
    setting = str(payload["setting"])
    seed = str(payload["seed"])
    for row in rows:
        if (
            row.get("factor") == factor
            and str(row.get("setting", "")).strip() == setting
            and str(row.get("seed", "")).strip() == seed
        ):
            row.update(payload)
            return
    rows.append(payload)


def _format_float(x):
    return f"{x:.6f}"


def _one_hot(y, n_classes):
    out = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


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


def _population_encode(x, k, sigma, lambda_max_hz, rng):
    centers = np.linspace(0.0, 1.0, k, dtype=np.float32)[None, None, :]
    x3 = x[:, :, None]
    rates = lambda_max_hz * np.exp(-((x3 - centers) ** 2) / (2.0 * sigma * sigma))
    lam = rates * WINDOW_S
    counts = rng.poisson(lam).astype(np.float32)
    return counts.reshape(x.shape[0], x.shape[1] * k)


def _median_timing_ms_per_sample(num_samples, repeats, fn):
    n = max(1, int(num_samples))
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        runs.append((time.perf_counter() - t0) * 1000.0 / n)
    return float(np.median(np.asarray(runs, dtype=np.float64)))


def _fit_hybrid_local_readout(
    x_train,
    y_train,
    x_test,
    y_test,
    k=4,
    sigma=0.25,
    lambda_max_hz=200.0,
    homeostasis="on",
    reward_shaping="signed",
    seed=11,
    timing_repeats=TIMING_REPEATS,
    measure_end_to_end=True,
    epochs=18,
):
    rng = np.random.default_rng(seed)
    xtr = _population_encode(x_train, k=k, sigma=sigma, lambda_max_hz=lambda_max_hz, rng=rng)
    xte = _population_encode(x_test, k=k, sigma=sigma, lambda_max_hz=lambda_max_hz, rng=rng)
    d = xtr.shape[1]
    w = rng.normal(0.0, 0.01, size=(N_CLASSES, d)).astype(np.float32)
    b = np.zeros(N_CLASSES, dtype=np.float32)
    ytr_oh = _one_hot(y_train, N_CLASSES)

    lr = 0.003
    homeo_decay = 0.98
    gentle_homeo_decay = 0.995
    gentle_interval_epochs = 5

    t0 = time.perf_counter()
    for epoch_idx in range(epochs):
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
        elif homeostasis == "gentle" and (epoch_idx + 1) % gentle_interval_epochs == 0:
            row_norm = np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
            w = gentle_homeo_decay * w / row_norm
    train_time_s = time.perf_counter() - t0

    logits = xte @ w.T + b[None, :]
    y_pred = np.argmax(logits, axis=1)

    infer_forward_ms = _median_timing_ms_per_sample(
        num_samples=xte.shape[0],
        repeats=timing_repeats,
        fn=lambda: xte @ w.T + b[None, :],
    )

    if measure_end_to_end:
        def _end_to_end():
            enc_rng = np.random.default_rng(seed)
            xte_e2e = _population_encode(
                x_test, k=k, sigma=sigma, lambda_max_hz=lambda_max_hz, rng=enc_rng
            )
            _ = xte_e2e @ w.T + b[None, :]

        infer_end_to_end_ms = _median_timing_ms_per_sample(
            num_samples=xte.shape[0],
            repeats=timing_repeats,
            fn=_end_to_end,
        )
    else:
        infer_end_to_end_ms = infer_forward_ms

    acc = _accuracy(y_test, y_pred)
    macro_f1 = _macro_f1(y_test, y_pred)
    return acc, macro_f1, train_time_s, infer_forward_ms, infer_end_to_end_ms


def _fit_stdp_competitive(
    x_train,
    y_train,
    x_test,
    y_test,
    k=4,
    sigma=0.25,
    lambda_max_hz=200.0,
    homeostasis="on",
    reward_shaping="signed",
    seed=11,
    timing_repeats=TIMING_REPEATS,
    measure_end_to_end=True,
    n_exc=96,
    epochs=9,
):
    rng = np.random.default_rng(seed)
    xtr = _population_encode(x_train, k=k, sigma=sigma, lambda_max_hz=lambda_max_hz, rng=rng)
    xte = _population_encode(x_test, k=k, sigma=sigma, lambda_max_hz=lambda_max_hz, rng=rng)
    xtr = xtr / (np.linalg.norm(xtr, axis=1, keepdims=True) + 1e-8)
    xte = xte / (np.linalg.norm(xte, axis=1, keepdims=True) + 1e-8)

    d = xtr.shape[1]
    w = rng.random((n_exc, d), dtype=np.float32)
    w /= (np.linalg.norm(w, axis=1, keepdims=True) + 1e-8)
    thr = np.zeros(n_exc, dtype=np.float32)
    lr_plus = 0.08
    lr_minus = 0.01
    thr_inc = 0.05
    thr_decay = 0.995

    t0 = time.perf_counter()
    for _ in range(epochs):
        idx = rng.permutation(xtr.shape[0])
        for i in idx:
            x = xtr[i]
            act = w @ x - thr
            winner = int(np.argmax(act))
            w[winner] += lr_plus * (x - w[winner])
            if reward_shaping == "signed":
                second = int(np.argpartition(act, -2)[-2])
                if second != winner:
                    w[second] -= lr_minus * x
            w[winner] = np.clip(w[winner], 0.0, None)
            w[winner] /= (np.linalg.norm(w[winner]) + 1e-8)
            if homeostasis == "on":
                thr[winner] += thr_inc
                thr *= thr_decay
    train_time_s = time.perf_counter() - t0

    votes = np.zeros((n_exc, N_CLASSES), dtype=np.int32)
    train_winners = np.argmax(xtr @ w.T - thr[None, :], axis=1)
    for wi, yi in zip(train_winners, y_train):
        votes[wi, yi] += 1
    neuron_label = np.argmax(votes, axis=1)

    test_act = xte @ w.T - thr[None, :]
    test_winners = np.argmax(test_act, axis=1)
    y_pred = neuron_label[test_winners]

    infer_forward_ms = _median_timing_ms_per_sample(
        num_samples=xte.shape[0],
        repeats=timing_repeats,
        fn=lambda: np.argmax(xte @ w.T - thr[None, :], axis=1),
    )

    if measure_end_to_end:
        def _end_to_end():
            enc_rng = np.random.default_rng(seed)
            xte_e2e = _population_encode(
                x_test, k=k, sigma=sigma, lambda_max_hz=lambda_max_hz, rng=enc_rng
            )
            xte_e2e = xte_e2e / (np.linalg.norm(xte_e2e, axis=1, keepdims=True) + 1e-8)
            _ = np.argmax(xte_e2e @ w.T - thr[None, :], axis=1)

        infer_end_to_end_ms = _median_timing_ms_per_sample(
            num_samples=xte.shape[0],
            repeats=timing_repeats,
            fn=_end_to_end,
        )
    else:
        infer_end_to_end_ms = infer_forward_ms

    stdp_weight_saturation_pct = float(np.mean(w <= 1e-8) * 100.0)
    stdp_weight_saturation_upper_pct = float(np.mean(w >= (1.0 - 1e-8)) * 100.0)
    top2 = np.partition(test_act, -2, axis=1)[:, -2:]
    stdp_winner_margin_mean = float(np.mean(top2[:, 1] - top2[:, 0]))

    acc = _accuracy(y_test, y_pred)
    macro_f1 = _macro_f1(y_test, y_pred)
    return (
        acc,
        macro_f1,
        train_time_s,
        infer_forward_ms,
        infer_end_to_end_ms,
        stdp_weight_saturation_pct,
        stdp_weight_saturation_upper_pct,
        stdp_winner_margin_mean,
    )


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


def _run_baseline_models(baseline_rows, x_train, y_train, x_test, y_test):
    for seed in SEEDS:
        h_acc, h_f1, h_train_s, h_inf_forward_ms, h_inf_end_to_end_ms = _fit_hybrid_local_readout(
            x_train, y_train, x_test, y_test, seed=seed
        )
        _upsert_baseline(
            baseline_rows,
            {
                "method": "hybrid_encoder_local_readout",
                "seed": str(seed),
                "test_accuracy": _format_float(h_acc),
                "macro_f1": _format_float(h_f1),
                "train_time_s": _format_float(h_train_s),
                "infer_time_ms_per_sample": _format_float(h_inf_forward_ms),
                "infer_forward_median_ms_per_sample": _format_float(h_inf_forward_ms),
                "infer_end_to_end_median_ms_per_sample": _format_float(h_inf_end_to_end_ms),
                "infer_timing_repeats": str(TIMING_REPEATS),
                "infer_batching": TIMING_BATCHING,
                "stdp_weight_saturation_pct": "",
                "stdp_weight_saturation_upper_pct": "",
                "stdp_winner_margin_mean": "",
                "notes": "phase2_auto_hybrid_local_rule",
            },
        )

        (
            s_acc,
            s_f1,
            s_train_s,
            s_inf_forward_ms,
            s_inf_end_to_end_ms,
            s_sat_low_pct,
            s_sat_up_pct,
            s_margin,
        ) = _fit_stdp_competitive(x_train, y_train, x_test, y_test, seed=seed)
        _upsert_baseline(
            baseline_rows,
            {
                "method": "pure_stdp_ei",
                "seed": str(seed),
                "test_accuracy": _format_float(s_acc),
                "macro_f1": _format_float(s_f1),
                "train_time_s": _format_float(s_train_s),
                "infer_time_ms_per_sample": _format_float(s_inf_forward_ms),
                "infer_forward_median_ms_per_sample": _format_float(s_inf_forward_ms),
                "infer_end_to_end_median_ms_per_sample": _format_float(s_inf_end_to_end_ms),
                "infer_timing_repeats": str(TIMING_REPEATS),
                "infer_batching": TIMING_BATCHING,
                "stdp_weight_saturation_pct": _format_float(s_sat_low_pct),
                "stdp_weight_saturation_upper_pct": _format_float(s_sat_up_pct),
                "stdp_winner_margin_mean": _format_float(s_margin),
                "notes": "phase2_auto_stdp_competitive_proxy",
            },
        )


def _run_ablations(ablation_rows, x_train, y_train, x_test, y_test):
    defaults = {
        "k": 4,
        "sigma": 0.25,
        "lambda_max_hz": 200.0,
        "homeostasis": "on",
        "reward_shaping": "signed",
    }
    grid = [
        ("K", ["1", "2", "4", "6"]),
        ("sigma", ["0.15", "0.25", "0.35"]),
        ("lambda_max_hz", ["100", "150", "200", "250"]),
        ("homeostasis", ["on", "gentle", "off"]),
        ("reward_shaping", ["positive_only", "signed"]),
    ]

    for factor, settings in grid:
        for setting in settings:
            seeds_for_setting = (
                ROBUST_ABLATION_SEEDS if factor in ("homeostasis", "reward_shaping") else SEEDS
            )
            for seed in seeds_for_setting:
                cfg = dict(defaults)
                if factor == "K":
                    cfg["k"] = int(setting)
                elif factor == "sigma":
                    cfg["sigma"] = float(setting)
                elif factor == "lambda_max_hz":
                    cfg["lambda_max_hz"] = float(setting)
                elif factor == "homeostasis":
                    cfg["homeostasis"] = setting
                elif factor == "reward_shaping":
                    cfg["reward_shaping"] = setting

                acc, f1, train_s, inf_forward_ms, _inf_e2e_ms = _fit_hybrid_local_readout(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    k=cfg["k"],
                    sigma=cfg["sigma"],
                    lambda_max_hz=cfg["lambda_max_hz"],
                    homeostasis=cfg["homeostasis"],
                    reward_shaping=cfg["reward_shaping"],
                    seed=seed,
                    timing_repeats=1,
                    measure_end_to_end=False,
                )
                _upsert_ablation(
                    ablation_rows,
                    {
                        "factor": factor,
                        "setting": str(setting),
                        "seed": str(seed),
                        "test_accuracy": _format_float(acc),
                        "macro_f1": _format_float(f1),
                        "train_time_s": _format_float(train_s),
                        "infer_time_ms_per_sample": _format_float(inf_forward_ms),
                        "notes": "phase2_auto_hybrid_ablation",
                    },
                )


def main():
    x_train, y_train, x_test, y_test = _split_data()

    baseline_rows = _load_rows(BASELINES_CSV)
    _run_baseline_models(baseline_rows, x_train, y_train, x_test, y_test)
    _write_rows(
        BASELINES_CSV,
        baseline_rows,
        [
            "method",
            "seed",
            "test_accuracy",
            "macro_f1",
            "train_time_s",
            "infer_time_ms_per_sample",
            "infer_forward_median_ms_per_sample",
            "infer_end_to_end_median_ms_per_sample",
            "infer_timing_repeats",
            "infer_batching",
            "stdp_weight_saturation_pct",
            "stdp_weight_saturation_upper_pct",
            "stdp_winner_margin_mean",
            "notes",
        ],
    )

    ablation_rows = _load_rows(ABLATIONS_CSV)
    _run_ablations(ablation_rows, x_train, y_train, x_test, y_test)
    _write_rows(
        ABLATIONS_CSV,
        ablation_rows,
        [
            "factor",
            "setting",
            "seed",
            "test_accuracy",
            "macro_f1",
            "train_time_s",
            "infer_time_ms_per_sample",
            "notes",
        ],
    )
    print("Updated baselines and ablations for hybrid/STDP Phase 2 runs.")


if __name__ == "__main__":
    main()
