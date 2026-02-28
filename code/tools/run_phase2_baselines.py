#!/usr/bin/env python3
import csv
import os
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(ROOT, "results", "baselines_raw.csv")
SEEDS = [11, 23, 37, 41, 53]
SPLIT_SEED = 2026
N_CLASSES = 10
WINDOW_S = 0.12
ENC_K = 4
ENC_SIGMA = 0.25
ENC_LAMBDA_MAX_HZ = 200.0


def _ensure_results_csv():
    if os.path.exists(RESULTS_CSV):
        return
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "seed",
                "test_accuracy",
                "macro_f1",
                "train_time_s",
                "infer_time_ms_per_sample",
                "notes",
            ],
        )
        writer.writeheader()


def _load_rows():
    _ensure_results_csv()
    with open(RESULTS_CSV, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(rows):
    fieldnames = [
        "method",
        "seed",
        "test_accuracy",
        "macro_f1",
        "train_time_s",
        "infer_time_ms_per_sample",
        "notes",
    ]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(x):
    return f"{x:.6f}"


def _split_data():
    digits = load_digits()
    x = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SPLIT_SEED
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=SPLIT_SEED
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


def _population_encode(x, k, sigma, lambda_max_hz, rng):
    centers = np.linspace(0.0, 1.0, k, dtype=np.float32)[None, None, :]
    x3 = x[:, :, None]
    rates = lambda_max_hz * np.exp(-((x3 - centers) ** 2) / (2.0 * sigma * sigma))
    lam = rates * WINDOW_S
    counts = rng.poisson(lam).astype(np.float32)
    return counts.reshape(x.shape[0], x.shape[1] * k)


def _evaluate_model(model, x_train, y_train, x_test, y_test):
    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    train_time_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = model.predict(x_test)
    infer_time_s = time.perf_counter() - t1

    acc = accuracy_score(y_test, y_pred) * 100.0
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    infer_ms = infer_time_s * 1000.0 / len(y_test)
    return acc, macro_f1, train_time_s, infer_ms


def _build_models(seed):
    logreg = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=seed,
                    n_jobs=None,
                ),
            ),
        ]
    )
    mlp = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=64,
                    learning_rate_init=1e-3,
                    max_iter=250,
                    random_state=seed,
                ),
            ),
        ]
    )
    return {
        "logreg_pixels": logreg,
        "mlp_pixels": mlp,
    }


def _upsert_row(rows, payload):
    method = payload["method"]
    seed = str(payload["seed"])
    for row in rows:
        if row.get("method") == method and str(row.get("seed", "")).strip() == seed:
            row.update(payload)
            return
    rows.append(payload)


def main():
    rows = _load_rows()
    x_train, y_train, _x_val, _y_val, x_test, y_test = _split_data()
    x_train_norm = (x_train / 16.0).clip(0.0, 1.0).astype(np.float32)
    x_test_norm = (x_test / 16.0).clip(0.0, 1.0).astype(np.float32)

    for seed in SEEDS:
        models = _build_models(seed)
        for method, model in models.items():
            acc, macro_f1, train_s, infer_ms = _evaluate_model(
                model, x_train, y_train, x_test, y_test
            )
            _upsert_row(
                rows,
                {
                    "method": method,
                    "seed": str(seed),
                    "test_accuracy": _format_metric(acc),
                    "macro_f1": _format_metric(macro_f1),
                    "train_time_s": _format_metric(train_s),
                    "infer_time_ms_per_sample": _format_metric(infer_ms),
                    "notes": "phase2_auto_baseline",
                },
            )

        enc_rng = np.random.default_rng(seed)
        x_train_enc = _population_encode(
            x_train_norm, ENC_K, ENC_SIGMA, ENC_LAMBDA_MAX_HZ, enc_rng
        )
        x_test_enc = _population_encode(
            x_test_norm, ENC_K, ENC_SIGMA, ENC_LAMBDA_MAX_HZ, enc_rng
        )
        encoded_logreg = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=seed,
                        n_jobs=None,
                    ),
                ),
            ]
        )
        acc, macro_f1, train_s, infer_ms = _evaluate_model(
            encoded_logreg, x_train_enc, y_train, x_test_enc, y_test
        )
        _upsert_row(
            rows,
            {
                "method": "logreg_spike_encoded_rates",
                "seed": str(seed),
                "test_accuracy": _format_metric(acc),
                "macro_f1": _format_metric(macro_f1),
                "train_time_s": _format_metric(train_s),
                "infer_time_ms_per_sample": _format_metric(infer_ms),
                "notes": "phase2_auto_baseline_encoded_rates",
            },
        )

    _write_rows(rows)
    print(f"Updated {RESULTS_CSV} with logreg/MLP multi-seed baselines.")


if __name__ == "__main__":
    main()
