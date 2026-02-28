#!/usr/bin/env python3
import csv
import math
import os
import re
from statistics import mean, stdev


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "results", "openml_benchmark_raw.csv")
SEED_TEX = os.path.join(ROOT, "tables", "openml_seed_values.tex")
SUMMARY_TEX = os.path.join(ROOT, "tables", "openml_benchmark.tex")

SEEDS = ["11", "23", "37", "41", "53"]
COND_LABELS = {
    "hybrid_default_norm_on_signed": "Hybrid default (norm on, signed)",
    "hybrid_best_norm_off_signed": "Hybrid best (norm off, signed)",
    "hybrid_norm_on_pos_only": "Hybrid norm on (pos-only)",
    "hybrid_norm_off_pos_only": "Hybrid norm off (pos-only)",
    "stdp_proxy_norm_on_signed": "STDP proxy (norm on, signed)",
    "logreg_spike_encoded_rates": "LogReg (spike-encoded rates)",
}
LABEL_TO_COND = {v: k for k, v in COND_LABELS.items()}


def _load_csv():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_cond_seed = {}
    by_cond = {}
    for r in rows:
        cond = r["condition"]
        seed = r["seed"]
        acc = float(r["test_accuracy"])
        f1 = float(r["macro_f1"])
        by_cond_seed[(cond, seed)] = acc
        by_cond.setdefault(cond, {"acc": [], "f1": []})
        by_cond[cond]["acc"].append(acc)
        by_cond[cond]["f1"].append(f1)
    return rows, by_cond_seed, by_cond


def _parse_seed_table():
    txt = open(SEED_TEX, encoding="utf-8").read()
    found = {}
    for line in txt.splitlines():
        raw = line.strip()
        if "&" not in raw or not raw.endswith("\\\\"):
            continue
        parts = [p.strip().rstrip("\\").strip() for p in raw.split("&")]
        label = parts[0]
        if label not in LABEL_TO_COND:
            continue
        cond = LABEL_TO_COND[label]
        vals = parts[1:6]
        if len(vals) != 5:
            raise SystemExit(f"Seed table parse error for {label}: expected 5 seeds")
        for seed, v in zip(SEEDS, vals):
            found[(cond, seed)] = float(v)
    return found


def _parse_summary_table():
    txt = open(SUMMARY_TEX, encoding="utf-8").read()
    out = {}
    # Example: Label & 86.69 $\pm$ 1.90 & 1.67 & 0.87 $\pm$ 0.02 \\
    pat = re.compile(
        r"^\s*(?P<label>[^&]+?)\s*&\s*(?P<accm>-?\d+\.\d+)\s*\$\\pm\$\s*(?P<accs>-?\d+\.\d+)\s*&\s*(?P<ci>-?\d+\.\d+)\s*&\s*(?P<f1m>-?\d+\.\d+)\s*\$\\pm\$\s*(?P<f1s>-?\d+\.\d+)\s*\\\\\s*$"
    )
    for line in txt.splitlines():
        m = pat.match(line)
        if not m:
            continue
        label = m.group("label").strip()
        if label not in LABEL_TO_COND:
            continue
        cond = LABEL_TO_COND[label]
        out[cond] = {
            "acc_mean_2d": float(m.group("accm")),
            "acc_std_2d": float(m.group("accs")),
            "ci_2d": float(m.group("ci")),
            "f1_mean_2d": float(m.group("f1m")),
            "f1_std_2d": float(m.group("f1s")),
        }
    return out


def _ci95(vals):
    if len(vals) < 2:
        return 0.0
    return 1.96 * stdev(vals) / math.sqrt(len(vals))


def _round2(x):
    return float(f"{x:.2f}")


def main():
    rows, by_cond_seed, by_cond = _load_csv()
    if len(rows) != 30:
        raise SystemExit(f"Expected 30 OpenML rows, found {len(rows)}")

    seed_vals = _parse_seed_table()
    for cond in COND_LABELS:
        for seed in SEEDS:
            key = (cond, seed)
            if key not in seed_vals:
                raise SystemExit(f"Missing seed-table value for {key}")
            c = by_cond_seed[key]
            t = seed_vals[key]
            if abs(c - t) > 1e-6:
                raise SystemExit(
                    f"CSV↔seed table mismatch for {cond} seed {seed}: csv={c:.6f}, table={t:.6f}"
                )

    summary = _parse_summary_table()
    for cond in COND_LABELS:
        if cond not in summary:
            raise SystemExit(f"Missing summary-table row for {cond}")
        acc = by_cond[cond]["acc"]
        f1 = by_cond[cond]["f1"]
        exp = {
            "acc_mean_2d": _round2(mean(acc)),
            "acc_std_2d": _round2(stdev(acc) if len(acc) > 1 else 0.0),
            "ci_2d": _round2(_ci95(acc)),
            "f1_mean_2d": _round2(mean(f1)),
            "f1_std_2d": _round2(stdev(f1) if len(f1) > 1 else 0.0),
        }
        got = summary[cond]
        for k, vexp in exp.items():
            if abs(got[k] - vexp) > 1e-6:
                raise SystemExit(
                    f"CSV↔summary table mismatch for {cond} field {k}: expected={vexp:.2f}, got={got[k]:.2f}"
                )

    print("OpenML provenance verification passed.")


if __name__ == "__main__":
    main()
