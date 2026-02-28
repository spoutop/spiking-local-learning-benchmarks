#!/usr/bin/env python3
import csv
import math
import os
from collections import defaultdict


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV = os.path.join(ROOT, "results", "split_robustness_raw.csv")
OUT_TEX = os.path.join(ROOT, "tables", "split_robustness.tex")
OUT_MD = os.path.join(ROOT, "split_robustness_summary.md")

SPLIT_ORDER = [2026, 2027, 2028]
COND_DEFAULT = "hybrid_default_norm_on"
COND_BEST = "hybrid_best_norm_off"


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _mean_std(values):
    clean = [x for x in values if x is not None]
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


def _fmt_mean_std(values):
    m, s = _mean_std(values)
    if m is None:
        return "TBD"
    return f"{m:.2f} $\\pm$ {s:.2f}"


def _fmt_signed(v):
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}"


def _read_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    rows = _read_rows(IN_CSV)
    by_split_cond = defaultdict(list)
    seeds_by_split_cond = defaultdict(set)
    for r in rows:
        split = int(r["split_seed"])
        cond = r["condition"]
        by_split_cond[(split, cond)].append(_to_float(r.get("test_accuracy")))
        seeds_by_split_cond[(split, cond)].add(int(r["model_seed"]))

    expected_seed_set = {11, 23, 37, 41, 53}
    split_default_means = []
    split_best_means = []
    split_deltas = []

    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"  \centering")
    latex_lines.append(
        r"  \caption{Robustness across dataset splits (no retuning). Per-split values aggregate over five model seeds; $\Delta$ denotes best minus default. Reward shaping is signed in both compared conditions; only the normalization heuristic is toggled.}"
    )
    latex_lines.append(r"  \scriptsize")
    latex_lines.append(r"  \setlength{\tabcolsep}{3pt}")
    latex_lines.append(r"  \resizebox{\linewidth}{!}{%")
    latex_lines.append(r"  \begin{tabular}{lccc}")
    latex_lines.append(r"    \toprule")
    latex_lines.append(
        r"    Split seed & Hybrid default Acc (mean$\pm$std, \%) & Hybrid best (norm off) Acc (mean$\pm$std, \%) & $\Delta$ (pp) \\"
    )
    latex_lines.append(r"    \midrule")

    md_lines = []
    md_lines.append("# Split Robustness Summary")
    md_lines.append("")
    md_lines.append("| Split | Hybrid default (mean±std) | Hybrid best norm-off (mean±std) | Δ (pp) |")
    md_lines.append("|---|---:|---:|---:|")

    for split in SPLIT_ORDER:
        default_vals = by_split_cond[(split, COND_DEFAULT)]
        best_vals = by_split_cond[(split, COND_BEST)]
        assert seeds_by_split_cond[(split, COND_DEFAULT)] == expected_seed_set, (
            f"Split {split} default seed set mismatch"
        )
        assert seeds_by_split_cond[(split, COND_BEST)] == expected_seed_set, (
            f"Split {split} best seed set mismatch"
        )
        m_def, _ = _mean_std(default_vals)
        m_best, _ = _mean_std(best_vals)
        assert m_def is not None and m_best is not None, f"Missing values for split {split}"
        delta = m_best - m_def
        split_default_means.append(m_def)
        split_best_means.append(m_best)
        split_deltas.append(delta)

        latex_lines.append(
            f"    {split} & {_fmt_mean_std(default_vals)} & {_fmt_mean_std(best_vals)} & {_fmt_signed(delta)} \\\\"
        )
        md_lines.append(
            f"| {split} | {_fmt_mean_std(default_vals).replace('$', '')} | {_fmt_mean_std(best_vals).replace('$', '')} | {_fmt_signed(delta)} |"
        )

    m_def_all, s_def_all = _mean_std(split_default_means)
    m_best_all, s_best_all = _mean_std(split_best_means)
    m_delta_all, s_delta_all = _mean_std(split_deltas)
    pos_count = sum(1 for d in split_deltas if d > 0)

    latex_lines.append(r"    \midrule")
    latex_lines.append(
        f"    Across splits & {m_def_all:.2f} $\\pm$ {s_def_all:.2f} & {m_best_all:.2f} $\\pm$ {s_best_all:.2f} & {_fmt_signed(m_delta_all)} $\\pm$ {s_delta_all:.2f} (\\(\\Delta>0\\) in {pos_count}/3) \\\\"
    )
    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"  \end{tabular}%")
    latex_lines.append(r"  }")
    latex_lines.append(
        r"  \vspace{1mm}\\\footnotesize{Per-split means are computed over $n=5$ model seeds; across-split summaries are computed over $n=3$ split-level means. Dominant-axis ablations in Table~\ref{tab:ablations} use $n=9$ seeds.}"
    )
    latex_lines.append(r"  \label{tab:split_robustness}")
    latex_lines.append(r"\end{table}")
    latex_lines.append("")

    md_lines.append("")
    md_lines.append(
        f"Across splits: default {m_def_all:.2f} ± {s_def_all:.2f}, best {m_best_all:.2f} ± {s_best_all:.2f}, delta {_fmt_signed(m_delta_all)} ± {s_delta_all:.2f} (Δ>0 in {pos_count}/3)."
    )
    md_lines.append("")
    md_lines.append(
        "Per split values aggregate over n=5 model seeds. Across-split summaries aggregate over n=3 split-level means."
    )

    _write(OUT_TEX, "\n".join(latex_lines))
    _write(OUT_MD, "\n".join(md_lines) + "\n")
    print(f"Wrote {OUT_TEX}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
