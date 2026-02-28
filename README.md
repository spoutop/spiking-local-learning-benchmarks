# Fixed-Seed Benchmark Code and Data (Paper Companion)

This package contains code and raw result data for the manuscript:
"Reward-Modulated Local Learning in Spiking Encoders: Controlled Benchmarks with STDP and Hybrid Rate Readouts".

## Structure
- `code/tools/`: Python scripts used to run benchmarks and build tables/figures.
- `data/results/`: Raw CSV outputs used as source-of-truth for reported aggregates.
- `docs/`: Minimal metadata and submission notes.

## Recommended quick checks
From this folder root:

```bash
python code/tools/verify_openml_provenance.py
```

## Notes
- Results are fixed-seed and intended for reproducibility/audit workflows.
- Aggregate numbers should be recomputed from CSV files in `data/results/`.
