#!/usr/bin/env python3
import csv
import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV = os.path.join(ROOT, "results", "openml_benchmark_raw.csv")
OUT_TEX = os.path.join(ROOT, "tables", "openml_seed_values.tex")


def _read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    rows = _read_csv(IN_CSV)
    order = [
        "hybrid_default_norm_on_signed",
        "hybrid_best_norm_off_signed",
        "hybrid_norm_on_pos_only",
        "hybrid_norm_off_pos_only",
        "stdp_proxy_norm_on_signed",
        "logreg_spike_encoded_rates",
    ]
    grouped = {k: {} for k in order}
    for row in rows:
        cond = row.get("condition")
        if cond in grouped:
            grouped[cond][row.get("seed")] = row.get("test_accuracy")

    seeds = ["11", "23", "37", "41", "53"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{OpenML MNIST per-seed accuracy values (\%) for the six reported conditions (split seed 2026).}")
    lines.append(r"  \scriptsize")
    lines.append(r"  \setlength{\tabcolsep}{3pt}")
    lines.append(r"  \resizebox{\linewidth}{!}{%")
    lines.append(r"  \begin{tabular}{lccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Condition & seed 11 & seed 23 & seed 37 & seed 41 & seed 53 \\")
    lines.append(r"    \midrule")
    label_map = {
        "hybrid_default_norm_on_signed": "Hybrid default (norm on, signed)",
        "hybrid_best_norm_off_signed": "Hybrid best (norm off, signed)",
        "hybrid_norm_on_pos_only": "Hybrid norm on (pos-only)",
        "hybrid_norm_off_pos_only": "Hybrid norm off (pos-only)",
        "stdp_proxy_norm_on_signed": "STDP proxy (norm on, signed)",
        "logreg_spike_encoded_rates": "LogReg (spike-encoded rates)",
    }
    for cond in order:
        vals = [grouped.get(cond, {}).get(seed, "TBD") for seed in seeds]
        lines.append(
            "    " + label_map[cond] + " & " + " & ".join(vals) + r" \\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}%")
    lines.append(r"  }")
    lines.append(r"  \vspace{1mm}\\\footnotesize{Values are taken directly from \texttt{results/openml\_benchmark\_raw.csv}; no additional runs.}")
    lines.append(r"  \label{tab:openml_seed_values}")
    lines.append(r"\end{table}")
    lines.append("")

    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
