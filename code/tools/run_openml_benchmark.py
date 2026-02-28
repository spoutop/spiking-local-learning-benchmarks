#!/usr/bin/env python3
import csv
import os
import time

# Runtime/cache guards for sandboxed executions.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, ".cache")
MPL_CACHE_DIR = os.path.join(CACHE_DIR, "matplotlib")
FC_CACHE_DIR = os.path.join(CACHE_DIR, "fontconfig")
SK_CACHE_DIR = os.path.join(CACHE_DIR, "sklearn")
for _d in (CACHE_DIR, MPL_CACHE_DIR, FC_CACHE_DIR, SK_CACHE_DIR):
    os.makedirs(_d, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", CACHE_DIR)
os.environ.setdefault("MPLCONFIGDIR", MPL_CACHE_DIR)
os.environ.setdefault("FC_CACHEDIR", FC_CACHE_DIR)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_phase2_models import _fit_hybrid_local_readout, _fit_stdp_competitive


OUT_CSV = os.path.join(ROOT, "results", "openml_benchmark_raw.csv")
DATA_HOME = SK_CACHE_DIR

SEEDS = [11, 23, 37, 41, 53]
SPLIT_SEED = 2026
WINDOW_S = 0.12
TRAIN_SUBSET_N = 12000


def _format_float(x):
    return f"{x:.6f}"


def _load_mnist_openml():
    os.makedirs(DATA_HOME, exist_ok=True)
    try:
        x, y = fetch_openml(
            "mnist_784",
            version=1,
            as_frame=False,
            parser="auto",
            return_X_y=True,
            data_home=DATA_HOME,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to fetch OpenML dataset 'mnist_784'. "
            f"Check network access and OpenML availability. Root cause: {exc}"
        ) from exc
    x = x.astype("float32") / 255.0
    y = y.astype("int64")
    return x, y


def _split_data(x, y):
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SPLIT_SEED
    )
    x_train, _x_val, y_train, _y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=SPLIT_SEED
    )
    if x_train.shape[0] > TRAIN_SUBSET_N:
        rng = np.random.default_rng(SPLIT_SEED)
        idx = rng.choice(x_train.shape[0], size=TRAIN_SUBSET_N, replace=False)
        x_train = x_train[idx]
        y_train = y_train[idx]
    return x_train, y_train, x_test, y_test


def _population_encode(x, k, sigma, lambda_max_hz, rng):
    centers = np.linspace(0.0, 1.0, k, dtype=np.float32)[None, None, :]
    x3 = x[:, :, None]
    rates = lambda_max_hz * np.exp(-((x3 - centers) ** 2) / (2.0 * sigma * sigma))
    lam = rates * WINDOW_S
    counts = rng.poisson(lam).astype(np.float32)
    return counts.reshape(x.shape[0], x.shape[1] * k)


def _eval_logreg_encoded(x_train, y_train, x_test, y_test, seed):
    rng = np.random.default_rng(seed)
    x_train_enc = _population_encode(x_train, 4, 0.25, 200.0, rng)
    x_test_enc = _population_encode(x_test, 4, 0.25, 200.0, rng)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    alpha=1e-4,
                    max_iter=1000,
                    tol=1e-3,
                    random_state=seed,
                ),
            ),
        ]
    )
    t0 = time.perf_counter()
    model.fit(x_train_enc, y_train)
    train_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    y_pred = model.predict(x_test_enc)
    infer_ms = (time.perf_counter() - t1) * 1000.0 / max(1, len(y_test))
    acc = float(np.mean(y_pred == y_test) * 100.0)
    f1 = 0.0
    # Macro-F1 without sklearn dependency churn
    for c in range(10):
        tp = np.sum((y_test == c) & (y_pred == c))
        fp = np.sum((y_test != c) & (y_pred == c))
        fn = np.sum((y_test == c) & (y_pred != c))
        if tp == 0 and (fp > 0 or fn > 0):
            continue
        if tp == 0 and fp == 0 and fn == 0:
            continue
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f1 += float(2.0 * p * r / (p + r + 1e-12))
    f1 /= 10.0
    return acc, f1, train_s, infer_ms


def main():
    x, y = _load_mnist_openml()
    x_train, y_train, x_test, y_test = _split_data(x, y)

    rows = []
    for seed in SEEDS:
        for condition, homeostasis, reward_shaping in [
            ("hybrid_default_norm_on_signed", "on", "signed"),
            ("hybrid_best_norm_off_signed", "off", "signed"),
            ("hybrid_norm_on_pos_only", "on", "positive_only"),
            ("hybrid_norm_off_pos_only", "off", "positive_only"),
        ]:
            acc, f1, train_s, inf_forward_ms, _inf_e2e = _fit_hybrid_local_readout(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                k=4,
                sigma=0.25,
                lambda_max_hz=200.0,
                homeostasis=homeostasis,
                reward_shaping=reward_shaping,
                seed=seed,
                timing_repeats=1,
                measure_end_to_end=False,
                epochs=6,
            )
            rows.append(
                {
                    "dataset": "mnist_784",
                    "condition": condition,
                    "seed": str(seed),
                    "test_accuracy": _format_float(acc),
                    "macro_f1": _format_float(f1),
                    "train_time_s": _format_float(train_s),
                    "infer_time_ms_per_sample": _format_float(inf_forward_ms),
                    "notes": "phase_openml_benchmark",
                }
            )

        s_acc, s_f1, s_train_s, s_inf_forward_ms, _s_inf_e2e, _sat_lo, _sat_hi, _margin = _fit_stdp_competitive(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            k=4,
            sigma=0.25,
            lambda_max_hz=200.0,
            homeostasis="on",
            reward_shaping="signed",
            seed=seed,
            timing_repeats=1,
            measure_end_to_end=False,
            n_exc=48,
            epochs=2,
        )
        rows.append(
            {
                "dataset": "mnist_784",
                "condition": "stdp_proxy_norm_on_signed",
                "seed": str(seed),
                "test_accuracy": _format_float(s_acc),
                "macro_f1": _format_float(s_f1),
                "train_time_s": _format_float(s_train_s),
                "infer_time_ms_per_sample": _format_float(s_inf_forward_ms),
                "notes": "phase_openml_benchmark",
            }
        )

        l_acc, l_f1, l_train_s, l_infer_ms = _eval_logreg_encoded(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            seed=seed,
        )
        rows.append(
            {
                "dataset": "mnist_784",
                "condition": "logreg_spike_encoded_rates",
                "seed": str(seed),
                "test_accuracy": _format_float(l_acc),
                "macro_f1": _format_float(l_f1),
                "train_time_s": _format_float(l_train_s),
                "infer_time_ms_per_sample": _format_float(l_infer_ms),
                "notes": "phase_openml_benchmark",
            }
        )

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
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
