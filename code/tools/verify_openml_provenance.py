#!/usr/bin/env python3
"""Reviewer quick-check for fixed-seed OpenML CSV integrity and headline contrasts."""

import csv
import os
from statistics import mean, stdev

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(ROOT, "data", "results", "openml_benchmark_raw.csv")
SEEDS = ["11", "23", "37", "41", "53"]
CONDITIONS = [
    "hybrid_default_norm_on_signed",
    "hybrid_best_norm_off_signed",
    "hybrid_norm_on_pos_only",
    "hybrid_norm_off_pos_only",
    "stdp_proxy_norm_on_signed",
    "logreg_spike_encoded_rates",
]


def _load_rows():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 30:
        raise SystemExit(f"Expected 30 OpenML rows, found {len(rows)}")
    return rows


def _check_completeness(rows):
    seen = {(r["condition"], r["seed"]) for r in rows}
    missing = []
    for cond in CONDITIONS:
        for seed in SEEDS:
            if (cond, seed) not in seen:
                missing.append((cond, seed))
    if missing:
        raise SystemExit(f"Missing condition-seed pairs: {missing}")


def _acc_by_cond(rows, cond):
    vals = [float(r["test_accuracy"]) for r in rows if r["condition"] == cond]
    if len(vals) != 5:
        raise SystemExit(f"Expected 5 rows for {cond}, found {len(vals)}")
    return vals


def _fmt(vals):
    return f"{mean(vals):.2f} ± {stdev(vals):.2f}"


def main():
    rows = _load_rows()
    _check_completeness(rows)

    on_signed = _acc_by_cond(rows, "hybrid_default_norm_on_signed")
    off_signed = _acc_by_cond(rows, "hybrid_best_norm_off_signed")
    on_pos = _acc_by_cond(rows, "hybrid_norm_on_pos_only")
    off_pos = _acc_by_cond(rows, "hybrid_norm_off_pos_only")
    proxy = _acc_by_cond(rows, "stdp_proxy_norm_on_signed")
    logreg = _acc_by_cond(rows, "logreg_spike_encoded_rates")

    print("OpenML CSV integrity: PASS (30 rows, complete 6x5 grid)")
    print(f"Hybrid norm-on signed: {_fmt(on_signed)}")
    print(f"Hybrid norm-off signed: {_fmt(off_signed)}")
    print(f"Hybrid norm-on pos-only: {_fmt(on_pos)}")
    print(f"Hybrid norm-off pos-only: {_fmt(off_pos)}")
    print(f"STDP proxy norm-on signed: {_fmt(proxy)}")
    print(f"LogReg spike-encoded rates: {_fmt(logreg)}")
    print(f"Delta (norm off - norm on, signed): {mean(off_signed) - mean(on_signed):.2f} pp")
    print(f"Delta (pos-only - signed, norm on): {mean(on_pos) - mean(on_signed):.2f} pp")
    print(f"Delta (pos-only - signed, norm off): {mean(off_pos) - mean(off_signed):.2f} pp")


if __name__ == "__main__":
    main()
