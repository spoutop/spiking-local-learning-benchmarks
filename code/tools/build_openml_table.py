#!/usr/bin/env python3
import csv
import math
import os
from collections import defaultdict


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV = os.path.join(ROOT, "results", "openml_benchmark_raw.csv")
OUT_TEX = os.path.join(ROOT, "tables", "openml_benchmark.tex")


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _mean_std_ci(values):
    vals = [x for x in values if x is not None]
    if not vals:
        return None, None, None
    n = len(vals)
    mean = sum(vals) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in vals) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    ci95 = 1.96 * std / math.sqrt(n) if n > 0 else 0.0
    return mean, std, ci95


def _fmt_mean_std(values):
    m, s, _ = _mean_std_ci(values)
    if m is None:
        return "TBD"
    return f"{m:.2f} $\\pm$ {s:.2f}"


def _fmt_ci(values):
    _, _, c = _mean_std_ci(values)
    if c is None:
        return "TBD"
    return f"{c:.2f}"


def _read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    rows = _read_csv(IN_CSV)
    by_cond = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)

    order = [
        ("hybrid_default_norm_on_signed", "Hybrid default (norm on, signed)"),
        ("hybrid_best_norm_off_signed", "Hybrid best (norm off, signed)"),
        ("hybrid_norm_on_pos_only", "Hybrid norm on (pos-only)"),
        ("hybrid_norm_off_pos_only", "Hybrid norm off (pos-only)"),
        ("stdp_proxy_norm_on_signed", "STDP proxy (norm on, signed)"),
        ("logreg_spike_encoded_rates", "LogReg (spike-encoded rates)"),
    ]

    default_acc = [_to_float(r.get("test_accuracy")) for r in by_cond.get(order[0][0], [])]
    best_acc = [_to_float(r.get("test_accuracy")) for r in by_cond.get(order[1][0], [])]
    pos_only_acc = [_to_float(r.get("test_accuracy")) for r in by_cond.get(order[2][0], [])]
    pos_only_off_acc = [_to_float(r.get("test_accuracy")) for r in by_cond.get(order[3][0], [])]
    diff_norm = [b - a for a, b in zip(default_acc, best_acc) if a is not None and b is not None]
    diff_reward = [p - a for a, p in zip(default_acc, pos_only_acc) if a is not None and p is not None]
    diff_reward_off = [p - a for a, p in zip(best_acc, pos_only_off_acc) if a is not None and p is not None]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{OpenML external benchmark on MNIST (fixed split/seed protocol). Accuracy is reported as mean$\pm$std, with 95\% CI half-width shown separately.}")
    lines.append(r"  \scriptsize")
    lines.append(r"  \setlength{\tabcolsep}{3pt}")
    lines.append(r"  \resizebox{\linewidth}{!}{%")
    lines.append(r"  \begin{tabular}{lccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Condition & Accuracy (mean$\pm$std, \%) & 95\% CI half-width (\%) & Macro F1 \\")
    lines.append(r"    \midrule")
    for key, label in order:
        cond_rows = by_cond.get(key, [])
        acc = [_to_float(r.get("test_accuracy")) for r in cond_rows]
        f1 = [_to_float(r.get("macro_f1")) for r in cond_rows]
        lines.append(f"    {label} & {_fmt_mean_std(acc)} & {_fmt_ci(acc)} & {_fmt_mean_std(f1)} \\\\")
    if diff_norm or diff_reward:
        lines.append(r"    \midrule")
        if diff_norm:
            d_mean = sum(diff_norm) / len(diff_norm)
            lines.append(f"    $\\Delta$ (norm off $-$ norm on, signed) & {d_mean:.2f} & -- & -- \\\\")
        if diff_reward:
            d_mean = sum(diff_reward) / len(diff_reward)
            lines.append(f"    $\\Delta$ (pos-only $-$ signed, norm on) & {d_mean:.2f} & -- & -- \\\\")
        if diff_reward_off:
            d_mean = sum(diff_reward_off) / len(diff_reward_off)
            lines.append(f"    $\\Delta$ (pos-only $-$ signed, norm off) & {d_mean:.2f} & -- & -- \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}%")
    lines.append(r"  }")
    lines.append(
        r"  \vspace{1mm}\\\footnotesize{$n=5$ seeds per condition; split seed 2026; no per-split tuning. The STDP proxy row is a controlled abstraction and is not presented as a competitive MNIST SNN baseline.}"
    )
    lines.append(r"  \label{tab:openml_bench}")
    lines.append(r"\end{table}")
    lines.append("")

    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
