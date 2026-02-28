#!/usr/bin/env python3
import csv
import math
import os

from run_phase2_models import (
    ROBUST_ABLATION_SEEDS,
    _fit_hybrid_local_readout,
    _split_data,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABLATIONS_CSV = os.path.join(ROOT, "results", "ablations_raw.csv")
OUT_TEX = os.path.join(ROOT, "tables", "norm_reward_2x2.tex")


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _format_float(x):
    return f"{x:.6f}"


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


def _read_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _extract_existing(rows):
    on_signed = {}
    off_signed = {}
    on_pos = {}
    off_pos = {}
    for r in rows:
        factor = (r.get("factor") or "").strip()
        setting = (r.get("setting") or "").strip()
        try:
            seed = int(r.get("seed"))
        except (TypeError, ValueError):
            continue
        acc = _to_float(r.get("test_accuracy"))
        if acc is None:
            continue
        if factor == "reward_shaping" and setting == "signed":
            on_signed[seed] = acc
        elif factor == "homeostasis" and setting == "off":
            off_signed[seed] = acc
        elif factor == "norm_reward_2x2" and setting == "on_positive_only":
            on_pos[seed] = acc
        elif factor == "norm_reward_2x2" and setting == "off_positive_only":
            off_pos[seed] = acc
    return on_signed, off_signed, on_pos, off_pos


def _compute_missing(on_pos, off_pos, rows):
    needed_on = [s for s in ROBUST_ABLATION_SEEDS if s not in on_pos]
    needed_off = [s for s in ROBUST_ABLATION_SEEDS if s not in off_pos]
    if not needed_on and not needed_off:
        return on_pos, off_pos

    x_train, y_train, x_test, y_test = _split_data()
    for seed in needed_on:
        acc, f1, train_s, inf_ms, _ = _fit_hybrid_local_readout(
            x_train,
            y_train,
            x_test,
            y_test,
            k=4,
            sigma=0.25,
            lambda_max_hz=200.0,
            homeostasis="on",
            reward_shaping="positive_only",
            seed=seed,
            timing_repeats=1,
            measure_end_to_end=False,
        )
        rows.append(
            {
                "factor": "norm_reward_2x2",
                "setting": "on_positive_only",
                "seed": str(seed),
                "test_accuracy": _format_float(acc),
                "macro_f1": _format_float(f1),
                "train_time_s": _format_float(train_s),
                "infer_time_ms_per_sample": _format_float(inf_ms),
                "notes": "phase2_norm_reward_2x2",
            }
        )
        on_pos[seed] = acc
    for seed in needed_off:
        acc, f1, train_s, inf_ms, _ = _fit_hybrid_local_readout(
            x_train,
            y_train,
            x_test,
            y_test,
            k=4,
            sigma=0.25,
            lambda_max_hz=200.0,
            homeostasis="off",
            reward_shaping="positive_only",
            seed=seed,
            timing_repeats=1,
            measure_end_to_end=False,
        )
        rows.append(
            {
                "factor": "norm_reward_2x2",
                "setting": "off_positive_only",
                "seed": str(seed),
                "test_accuracy": _format_float(acc),
                "macro_f1": _format_float(f1),
                "train_time_s": _format_float(train_s),
                "infer_time_ms_per_sample": _format_float(inf_ms),
                "notes": "phase2_norm_reward_2x2",
            }
        )
        off_pos[seed] = acc
    _write_rows(
        ABLATIONS_CSV,
        rows,
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
    return on_pos, off_pos


def main():
    rows = _read_rows(ABLATIONS_CSV)
    on_signed, off_signed, on_pos, off_pos = _extract_existing(rows)
    on_pos, off_pos = _compute_missing(on_pos, off_pos, rows)

    on_signed_vals = [on_signed.get(s) for s in ROBUST_ABLATION_SEEDS]
    on_pos_vals = [on_pos.get(s) for s in ROBUST_ABLATION_SEEDS]
    off_signed_vals = [off_signed.get(s) for s in ROBUST_ABLATION_SEEDS]
    off_pos_vals = [off_pos.get(s) for s in ROBUST_ABLATION_SEEDS]

    on_diffs = [
        b - a for a, b in zip(on_signed_vals, on_pos_vals) if a is not None and b is not None
    ]
    off_diffs = [
        b - a for a, b in zip(off_signed_vals, off_pos_vals) if a is not None and b is not None
    ]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Explicit 2$\times$2 interaction summary for normalization and reward shaping on \texttt{sklearn} digits (descriptive, no factorial significance claim).}")
    lines.append(r"  \scriptsize")
    lines.append(r"  \setlength{\tabcolsep}{3pt}")
    lines.append(r"  \resizebox{\linewidth}{!}{%")
    lines.append(r"  \begin{tabular}{lccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Condition & Accuracy (mean$\pm$std, \%) & 95\% CI half-width (\%) & Notes \\")
    lines.append(r"    \midrule")
    lines.append(
        f"    norm on + reward signed & {_fmt_mean_std(on_signed_vals)} & {_fmt_ci(on_signed_vals)} & default \\\\"
    )
    lines.append(
        f"    norm on + reward pos-only & {_fmt_mean_std(on_pos_vals)} & {_fmt_ci(on_pos_vals)} & targeted run \\\\"
    )
    lines.append(
        f"    norm off + reward signed & {_fmt_mean_std(off_signed_vals)} & {_fmt_ci(off_signed_vals)} & ablation row \\\\"
    )
    lines.append(
        f"    norm off + reward pos-only & {_fmt_mean_std(off_pos_vals)} & {_fmt_ci(off_pos_vals)} & targeted run \\\\"
    )
    lines.append(r"    \midrule")
    if on_diffs:
        lines.append(f"    $\\Delta$ pos-only$-$signed (norm on) & {sum(on_diffs)/len(on_diffs):.2f} & -- & descriptive \\\\")
    if off_diffs:
        lines.append(f"    $\\Delta$ pos-only$-$signed (norm off) & {sum(off_diffs)/len(off_diffs):.2f} & -- & descriptive \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}%")
    lines.append(r"  }")
    lines.append(r"  \vspace{1mm}\\\footnotesize{$n=9$ seeds in each cell, using the dominant-axis seed set.}")
    lines.append(r"  \label{tab:norm_reward_2x2}")
    lines.append(r"\end{table}")
    lines.append("")

    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
