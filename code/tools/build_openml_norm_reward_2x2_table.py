#!/usr/bin/env python3
import csv
import math
import os
from collections import defaultdict


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV = os.path.join(ROOT, "results", "openml_benchmark_raw.csv")
OUT_TEX = os.path.join(ROOT, "tables", "openml_norm_reward_2x2.tex")


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _stats(vals):
    clean = [v for v in vals if v is not None]
    if not clean:
        return None, None
    n = len(clean)
    mean = sum(clean) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in clean) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return mean, std


def _fmt(vals):
    m, s = _stats(vals)
    if m is None:
        return "TBD"
    return f"{m:.2f} $\\pm$ {s:.2f}"


def _read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    rows = _read_csv(IN_CSV)
    by_cond = defaultdict(list)
    for row in rows:
        by_cond[row.get("condition")].append(_to_float(row.get("test_accuracy")))

    conds = [
        ("hybrid_default_norm_on_signed", "Norm on + signed"),
        ("hybrid_norm_on_pos_only", "Norm on + pos-only"),
        ("hybrid_best_norm_off_signed", "Norm off + signed"),
        ("hybrid_norm_off_pos_only", "Norm off + pos-only"),
    ]

    on_signed = by_cond.get("hybrid_default_norm_on_signed", [])
    on_pos = by_cond.get("hybrid_norm_on_pos_only", [])
    off_signed = by_cond.get("hybrid_best_norm_off_signed", [])
    off_pos = by_cond.get("hybrid_norm_off_pos_only", [])

    d_on = [b - a for a, b in zip(on_signed, on_pos) if a is not None and b is not None]
    d_off = [b - a for a, b in zip(off_signed, off_pos) if a is not None and b is not None]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{OpenML MNIST descriptive 2$\times$2 interaction context for hybrid reward shaping and normalization regime (accuracy, \%).}")
    lines.append(r"  \scriptsize")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(r"  \begin{tabular}{lc}")
    lines.append(r"    \toprule")
    lines.append(r"    Cell & Accuracy (mean$\pm$std) \\")
    lines.append(r"    \midrule")
    for key, label in conds:
        lines.append(f"    {label} & {_fmt(by_cond.get(key, []))} \\\\")
    lines.append(r"    \midrule")
    lines.append(f"    $\\Delta$ (pos-only $-$ signed | norm on) & {_fmt(d_on)} \\\\")
    lines.append(f"    $\\Delta$ (pos-only $-$ signed | norm off) & {_fmt(d_off)} \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \vspace{1mm}\\\footnotesize{Descriptive interaction context only ($n=5$ seeds/cell); no confirmatory factorial significance claim.}")
    lines.append(r"  \label{tab:openml_norm_reward_2x2}")
    lines.append(r"\end{table}")
    lines.append("")

    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
