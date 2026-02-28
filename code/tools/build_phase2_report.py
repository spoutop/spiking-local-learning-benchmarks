#!/usr/bin/env python3
import csv
import math
import os
from collections import defaultdict


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES_CSV = os.path.join(ROOT, "results", "baselines_raw.csv")
ABLATIONS_CSV = os.path.join(ROOT, "results", "ablations_raw.csv")
REPORT_MD = os.path.join(ROOT, "phase2_results_summary.md")


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


def _load_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt_triplet(values):
    m, s, c = _mean_std_ci(values)
    if m is None:
        return "TBD"
    return f"{m:.2f} +- {s:.2f} (95% CI +/- {c:.2f})"


def _sign_test_two_sided(a_vals, b_vals):
    diffs = [a - b for a, b in zip(a_vals, b_vals) if a is not None and b is not None and a != b]
    n = len(diffs)
    if n == 0:
        return None, None, None
    k_pos = sum(1 for d in diffs if d > 0)
    k_tail = min(k_pos, n - k_pos)
    # Exact binomial two-sided p-value under p=0.5
    p = 0.0
    for i in range(0, k_tail + 1):
        p += math.comb(n, i)
    p = min(1.0, 2.0 * p / (2 ** n))
    mean_diff = sum(diffs) / n
    return mean_diff, n, p


def _paired_diff_ci95(a_vals, b_vals):
    diffs = [a - b for a, b in zip(a_vals, b_vals) if a is not None and b is not None]
    if not diffs:
        return None, None, None
    n = len(diffs)
    mean_diff = sum(diffs) / n
    if n > 1:
        var = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    ci95 = 1.96 * std / math.sqrt(n) if n > 0 else 0.0
    return mean_diff, ci95, n


def _paired_cohens_dz(a_vals, b_vals):
    diffs = [a - b for a, b in zip(a_vals, b_vals) if a is not None and b is not None]
    n = len(diffs)
    if n < 2:
        return None
    mean_diff = sum(diffs) / n
    var = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std = math.sqrt(var)
    if std <= 1e-12:
        return None
    return mean_diff / std


def _cliffs_delta(a_vals, b_vals):
    paired = [(a, b) for a, b in zip(a_vals, b_vals) if a is not None and b is not None]
    n = len(paired)
    if n == 0:
        return None
    gt = sum(1 for a, b in paired if a > b)
    lt = sum(1 for a, b in paired if a < b)
    return (gt - lt) / n


def _fmt_eff(x):
    return "NA" if x is None else f"{x:.2f}"


def _extract_seeded(rows, method=None, factor=None, setting=None):
    out = {}
    for r in rows:
        if method is not None and r.get("method") != method:
            continue
        if factor is not None and r.get("factor") != factor:
            continue
        if setting is not None and r.get("setting") != setting:
            continue
        try:
            seed = int(r.get("seed"))
        except (TypeError, ValueError):
            continue
        out[seed] = _to_float(r.get("test_accuracy"))
    return out


def main():
    b_rows = _load_csv(BASELINES_CSV)
    a_rows = _load_csv(ABLATIONS_CSV)

    b_group = defaultdict(list)
    for r in b_rows:
        b_group[r["method"]].append(_to_float(r.get("test_accuracy")))

    a_group = defaultdict(list)
    for r in a_rows:
        key = (r["factor"], r["setting"])
        a_group[key].append(_to_float(r.get("test_accuracy")))

    methods = [
        "logreg_pixels",
        "mlp_pixels",
        "hybrid_encoder_local_readout",
        "pure_stdp_ei",
    ]
    ablation_order = [
        ("K", "1"),
        ("K", "2"),
        ("K", "4"),
        ("K", "6"),
        ("sigma", "0.15"),
        ("sigma", "0.25"),
        ("sigma", "0.35"),
        ("lambda_max_hz", "100"),
        ("lambda_max_hz", "150"),
        ("lambda_max_hz", "200"),
        ("lambda_max_hz", "250"),
        ("homeostasis", "on"),
        ("homeostasis", "gentle"),
        ("homeostasis", "off"),
        ("reward_shaping", "positive_only"),
        ("reward_shaping", "signed"),
    ]

    lines = []
    lines.append("# Phase 2 Results Summary")
    lines.append("")
    lines.append("## Baselines (Test Accuracy %)")
    for m in methods:
        lines.append(f"- {m}: {_fmt_triplet(b_group.get(m, []))}")
    lines.append("")
    lines.append("## Ablations (Test Accuracy %)")
    for factor, setting in ablation_order:
        key = (factor, setting)
        lines.append(f"- {factor}={setting}: {_fmt_triplet(a_group.get(key, []))}")
    lines.append("")

    lines.append("## Paired significance checks (seed-matched sign test)")
    lines.append("- Seeds are treated as the unit of replication.")
    hybrid_map = _extract_seeded(b_rows, method="hybrid_encoder_local_readout")
    stdp_map = _extract_seeded(b_rows, method="pure_stdp_ei")
    common = sorted(set(hybrid_map) & set(stdp_map))
    hybrid_vals = [hybrid_map[s] for s in common]
    stdp_vals = [stdp_map[s] for s in common]
    md, n, p = _sign_test_two_sided(hybrid_vals, stdp_vals)
    if md is not None:
        e_md, e_ci, e_n = _paired_diff_ci95(hybrid_vals, stdp_vals)
        dz = _paired_cohens_dz(hybrid_vals, stdp_vals)
        cliff = _cliffs_delta(hybrid_vals, stdp_vals)
        lines.append(
            f"- hybrid vs pure_stdp_ei: mean diff {e_md:.2f} pp, sign-test n={n}, p={p:.4f}; effect CI95 +/- {e_ci:.2f} pp (paired n={e_n}); d_z={_fmt_eff(dz)} and Cliff's delta={_fmt_eff(cliff)}"
        )

    homeo_on_map = _extract_seeded(a_rows, factor="homeostasis", setting="on")
    homeo_off_map = _extract_seeded(a_rows, factor="homeostasis", setting="off")
    common = sorted(set(homeo_on_map) & set(homeo_off_map))
    homeo_on = [homeo_on_map[s] for s in common]
    homeo_off = [homeo_off_map[s] for s in common]
    md, n, p = _sign_test_two_sided(homeo_off, homeo_on)
    if md is not None:
        e_md, e_ci, e_n = _paired_diff_ci95(homeo_off, homeo_on)
        dz = _paired_cohens_dz(homeo_off, homeo_on)
        cliff = _cliffs_delta(homeo_off, homeo_on)
        lines.append(
            f"- normalization heuristic off vs on: mean diff {e_md:.2f} pp, sign-test n={n}, p={p:.4f}; effect CI95 +/- {e_ci:.2f} pp (paired n={e_n}); d_z={_fmt_eff(dz)} and Cliff's delta={_fmt_eff(cliff)}"
        )

    reward_pos_map = _extract_seeded(a_rows, factor="reward_shaping", setting="positive_only")
    reward_signed_map = _extract_seeded(a_rows, factor="reward_shaping", setting="signed")
    common = sorted(set(reward_pos_map) & set(reward_signed_map))
    reward_pos = [reward_pos_map[s] for s in common]
    reward_signed = [reward_signed_map[s] for s in common]
    md, n, p = _sign_test_two_sided(reward_pos, reward_signed)
    if md is not None:
        e_md, e_ci, e_n = _paired_diff_ci95(reward_pos, reward_signed)
        dz = _paired_cohens_dz(reward_pos, reward_signed)
        cliff = _cliffs_delta(reward_pos, reward_signed)
        lines.append(
            f"- reward positive_only vs signed: mean diff {e_md:.2f} pp, sign-test n={n}, p={p:.4f}; effect CI95 +/- {e_ci:.2f} pp (paired n={e_n}); d_z={_fmt_eff(dz)} and Cliff's delta={_fmt_eff(cliff)}"
        )
    lines.append("")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
