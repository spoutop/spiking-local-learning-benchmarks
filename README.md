# Code and Data Package

This repository contains the core reproducibility artifacts for the manuscript:
"Reward-Modulated Local Learning in Spiking Encoders: Controlled Benchmarks with STDP and Hybrid Rate Readouts".

## Contents
- `code/tools/`: core run scripts + OpenML integrity checker.
- `data/results/`: raw fixed-seed CSV outputs used for reported metrics.

## One-command check
From repository root, run:

```bash
python code/tools/verify_openml_provenance.py
```

Expected result:
- CSV integrity pass (`30 rows`, complete `6 conditions x 5 seeds`)
- Printed condition means/std and key deltas for quick verification.

## Raw result files
- `data/results/baselines_raw.csv`
- `data/results/ablations_raw.csv`
- `data/results/split_robustness_raw.csv`
- `data/results/temporal_synthetic_raw.csv`
- `data/results/openml_benchmark_raw.csv`
