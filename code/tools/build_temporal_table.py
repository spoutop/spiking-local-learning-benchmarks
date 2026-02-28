#!/usr/bin/env python3
import csv
import math
import os
from collections import defaultdict


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV = os.path.join(ROOT, "results", "temporal_synthetic_raw.csv")
OUT_TEX = os.path.join(ROOT, "tables", "temporal_synthetic.tex")


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
        ("count_readout_local", "Count readout local (timing-agnostic)"),
        ("timebin_readout_local", "Time-bin readout local (timing-aware)"),
    ]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Temporal synthetic benchmark (fixed seeds). Accuracy is reported as mean$\pm$std, with 95\% CI half-width shown separately.}")
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
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}%")
    lines.append(r"  }")
    lines.append(r"  \vspace{1mm}\\\footnotesize{$n=5$ seeds per condition; synthetic temporal-order classification with fixed split seed 2026.}")
    lines.append(r"  \label{tab:temporal_synth}")
    lines.append(r"\end{table}")
    lines.append("")

    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
