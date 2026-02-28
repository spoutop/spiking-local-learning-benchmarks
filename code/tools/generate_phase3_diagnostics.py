#!/usr/bin/env python3
import csv
import os
from collections import defaultdict

# Writable local caches for matplotlib/fontconfig in restricted environments.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, ".cache")
MPL_CACHE_DIR = os.path.join(CACHE_DIR, "matplotlib")
FC_CACHE_DIR = os.path.join(CACHE_DIR, "fontconfig")
for _d in (CACHE_DIR, MPL_CACHE_DIR, FC_CACHE_DIR):
    os.makedirs(_d, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", CACHE_DIR)
os.environ.setdefault("MPLCONFIGDIR", MPL_CACHE_DIR)
os.environ.setdefault("FC_CACHEDIR", FC_CACHE_DIR)

import matplotlib.pyplot as plt
import numpy as np


ABLATIONS_CSV = os.path.join(ROOT, "results", "ablations_raw.csv")
OUT_FIG = os.path.join(ROOT, "figs", "homeostasis_diag.png")


def _load_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _extract(rows, factor, setting):
    vals = []
    for r in rows:
        if r.get("factor") != factor or r.get("setting") != setting:
            continue
        try:
            vals.append(float(r.get("test_accuracy")))
        except (TypeError, ValueError):
            continue
    return vals


def main():
    rows = _load_rows(ABLATIONS_CSV)
    on = _extract(rows, "homeostasis", "on")
    off = _extract(rows, "homeostasis", "off")
    rp = _extract(rows, "reward_shaping", "positive_only")
    rs = _extract(rows, "reward_shaping", "signed")

    groups = [on, off, rs, rp]
    labels = ["Homeo on", "Homeo off", "Signed reward", "Positive-only"]

    means = [np.mean(g) if g else np.nan for g in groups]
    stds = [np.std(g, ddof=1) if len(g) > 1 else 0.0 for g in groups]

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    x = np.arange(len(labels))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=["#4C78A8", "#F58518", "#72B7B2", "#E45756"],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
    )

    rng = np.random.default_rng(2026)
    for i, g in enumerate(groups):
        if not g:
            continue
        jitter = rng.uniform(-0.10, 0.10, size=len(g))
        ax.scatter(np.full(len(g), i) + jitter, g, s=22, color="black", alpha=0.75, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Normalization and Reward-Shaping Diagnostics")
    ax.set_ylim(75, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG)
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
