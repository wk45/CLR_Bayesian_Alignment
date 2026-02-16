# CLR Bayesian Alignment

This repository contains a standalone implementation of the CLR Bayesian alignment workflow,
focused on **Section 2.4.2, Algorithm 3**.

Change history is maintained in `UPDATE_LOG.md`.

## Manuscript Status

The related manuscript is currently **under review at CSDA**.

This repository provides implementation and reproducible experiments, but does not include
full manuscript text.

## Repository Contents

- `src/alg3/inference.py`: Algorithm 3/4 MCMC core and wrappers
- `src/alg3/modeling.py`: covariance and data helpers
- `src/alg3/clr.py`: pure-Python CLR transform utilities
- `data/data_st.npz`: sample dataset for smoke/benchmark tests
- `scripts/run_alg3_smoke.py`: minimal execution check
- `scripts/run_scale_bench.py`: scale benchmark (`--full`, `--stress`)
- `tests/test_alg3_run.py`: regression tests for core runtime behavior

## Installation

```bash
python -m pip install -e .
```

Optional acceleration:

```bash
python -m pip install numba
```

## Quick Start

```bash
python scripts/run_alg3_smoke.py
python scripts/run_scale_bench.py --full
python scripts/run_scale_bench.py --stress
```
