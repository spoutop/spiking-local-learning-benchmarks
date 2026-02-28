#!/usr/bin/env python3
import csv
import math
import os
from collections import defaultdict


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
TABLES_DIR = os.path.join(ROOT, "tables")


def _to_float(value):
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _fmt_mean_std(values, digits=2):
    clean = [v for v in values if v is not None]
    if not clean:
        return "TBD"
    mean = sum(clean) / len(clean)
    if len(clean) > 1:
        var = sum((x - mean) ** 2 for x in clean) / (len(clean) - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _ci95(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    n = len(clean)
    mean = sum(clean) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in clean) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return 1.96 * std / math.sqrt(n) if n > 0 else 0.0


def _fmt_ci(values, digits=2):
    c = _ci95(values)
    if c is None:
        return "TBD"
    return f"{c:.{digits}f}"


def _escape_latex(text):
    if text is None:
        return ""
    out = str(text)
    out = out.replace("\\", r"\textbackslash{}")
    out = out.replace("_", r"\_")
    out = out.replace("%", r"\%")
    out = out.replace("&", r"\&")
    out = out.replace("#", r"\#")
    return out


def _read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def build_baselines():
    rows = _read_csv(os.path.join(RESULTS_DIR, "baselines_raw.csv"))
    grouped = defaultdict(list)
    notes = {}
    for r in rows:
        method = (r.get("method") or "").strip()
        if not method:
            continue
        grouped[method].append(r)
        n = (r.get("notes") or "").strip()
        if n:
            notes[method] = n

    ordered_methods = [
        "logreg_pixels",
        "mlp_pixels",
        "logreg_spike_encoded_rates",
        "hybrid_encoder_local_readout",
        "pure_stdp_ei",
    ]
    display_names = {
        "logreg_pixels": "LogReg (pixels)",
        "mlp_pixels": "MLP (pixels)",
        "logreg_spike_encoded_rates": "LogReg (spike-encoded rates)",
        "hybrid_encoder_local_readout": "Hybrid local readout",
        "pure_stdp_ei": "STDP-style competitive",
    }
    note_names = {
        "phase2_auto_baseline": "auto-baseline",
        "phase2_auto_baseline_encoded_rates": "auto-encoded-baseline",
        "phase2_auto_hybrid_local_rule": "auto-hybrid",
        "phase2_auto_stdp_competitive_proxy": "auto-stdp-proxy",
    }

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Baseline comparison over seeds. Accuracy is reported as mean$\pm$std, with 95\% CI half-width shown separately.}")
    lines.append(r"  \scriptsize")
    lines.append(r"  \setlength{\tabcolsep}{3pt}")
    lines.append(r"  \resizebox{\linewidth}{!}{%")
    lines.append(r"  \begin{tabular}{lccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Method & Accuracy (mean$\pm$std, \%) & 95\% CI half-width (\%) & Macro F1 \\")
    lines.append(r"    \midrule")

    for method in ordered_methods:
        method_rows = grouped.get(method, [])
        acc_vals = [_to_float(r.get("test_accuracy")) for r in method_rows]
        f1_vals = [_to_float(r.get("macro_f1")) for r in method_rows]
        note = notes.get(method, "")
        if not note:
            note = "TBD"
        note = note_names.get(note, note)
        lines.append(
            f"    {_escape_latex(display_names.get(method, method))} & {_fmt_mean_std(acc_vals)} & {_fmt_ci(acc_vals)} & {_fmt_mean_std(f1_vals)} \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}%")
    lines.append(r"  }")
    lines.append(r"  \label{tab:baselines}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def build_ablations():
    rows = _read_csv(os.path.join(RESULTS_DIR, "ablations_raw.csv"))
    grouped = defaultdict(list)
    for r in rows:
        factor = (r.get("factor") or "").strip()
        setting = (r.get("setting") or "").strip()
        if not factor or not setting:
            continue
        grouped[(factor, setting)].append(r)

    ordered = [
        ("K", ["1", "2", "4", "6"]),
        ("sigma", ["0.15", "0.25", "0.35"]),
        ("lambda_max_hz", ["100", "150", "200", "250"]),
        ("homeostasis", ["on", "gentle", "off"]),
        ("reward_shaping", ["signed", "positive_only"]),
    ]
    factor_display = {
        "K": "K",
        "sigma": r"$\sigma$",
        "lambda_max_hz": r"$\lambda_{\max}$ (Hz)",
        "homeostasis": "norm heuristic",
        "reward_shaping": "reward",
    }
    setting_display = {
        "positive_only": "pos-only",
        "signed": "signed",
        "gentle": "gentle",
    }

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Ablation results over seeds. Accuracy is reported as mean$\pm$std, with 95\% CI half-width shown separately.}")
    lines.append(r"  \scriptsize")
    lines.append(r"  \setlength{\tabcolsep}{3pt}")
    lines.append(r"  \resizebox{\linewidth}{!}{%")
    lines.append(r"  \begin{tabular}{llcc}")
    lines.append(r"    \toprule")
    lines.append(r"    Factor & Setting & Accuracy (mean$\pm$std, \%) & 95\% CI half-width (\%) \\")
    lines.append(r"    \midrule")

    for factor, settings in ordered:
        for setting in settings:
            key = (factor, setting)
            set_rows = grouped.get(key, [])
            acc_vals = [_to_float(r.get("test_accuracy")) for r in set_rows]
            setting_label = setting_display.get(setting, setting)
            if factor in {"homeostasis", "reward_shaping"}:
                setting_label = f"{setting_label} (n=9)"
            lines.append(
                f"    {factor_display.get(factor, _escape_latex(factor))} & {_escape_latex(setting_label)} & {_fmt_mean_std(acc_vals)} & {_fmt_ci(acc_vals)} \\\\"
            )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}%")
    lines.append(r"  }")
    lines.append(
        r"  \vspace{1mm}\\"
        r"\footnotesize{Rows for normalization heuristic and reward shaping use $n=9$ seeds; all other ablation rows use $n=5$ seeds. The reward=signed row is the default configuration (normalization heuristic on), so those values match.}"
    )
    lines.append(r"  \label{tab:ablations}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main():
    baselines_tex = build_baselines()
    ablations_tex = build_ablations()
    _write(os.path.join(TABLES_DIR, "baselines.tex"), baselines_tex)
    _write(os.path.join(TABLES_DIR, "ablations.tex"), ablations_tex)
    print("Wrote tables/baselines.tex and tables/ablations.tex")


if __name__ == "__main__":
    main()
