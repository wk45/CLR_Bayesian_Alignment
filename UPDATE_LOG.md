# Update Log

This file is the single running log for project updates.
Rule: after each meaningful change, append a new dated entry with:
- what changed
- files touched
- validation run (tests/smoke/compile)

## 2026-02-16

### Project setup and repo
- Created and used `.venv`.
- Added `requirements.txt` flow and installed dependencies.
- Initialized git repository and pushed to `https://github.com/wk45/CLR_Bayesian_Alignment`.

### Runtime/test scaffolding
- Added alg3 smoke test and test script execution path.
- Main test file: `tests/test_alg3_run.py`.
- Smoke script: `scripts/run_alg3_smoke.py`.

### CLR module path cleanup
- Stopped using fallback naming and switched to `clr.py`.
- Added `src/alg3/clr.py`.
- Removed `src/alg3/clr_fallback.py`.
- Updated imports in `src/alg3/modeling.py`.

### Typing and notation alignment
- Added function-level type annotations broadly in `src/alg3/inference.py` and related code.
- Aligned notation to paper/`sig_est01` convention across code and outputs (`hInv -> h`, `gamma_hInv -> gamma_h`, `hInv_mean -> h_bar`, `GammaInv -> Gamma`, `Acc_hInv -> Acc_h`, `Acc_rate_hInv -> Acc_rate_h`).

### Output/data structure cleanup
- Removed list-style output mode (`as_lists` path).
- Standardized outputs to numpy arrays.

### Performance and semantic fixes (burn/collect_traces)
- Fixed burn-in storage semantics; stored length now follows post-burn and thinning logic.
- Fixed trace collection semantics; when `collect_traces=False`, heavy traces (`G/A/Alpha/C/F/Gamma`) are not stored.
- Kept `SSE` trace available for diagnostics.
- Applied to both alg3 and alg4 cores/wrappers in `src/alg3/inference.py`.

### Validation done
- `python -m py_compile src/alg3/inference.py tests/test_alg3_run.py`
- `python tests/test_alg3_run.py`
- `python scripts/run_alg3_smoke.py`
- Additional manual check for alg4 with `collect_traces=False`.

## 2026-02-16 (this task)

### Logging policy added
- Created this `UPDATE_LOG.md` as the always-updated project change log.
- Added a pointer in `README.md` so the log location is visible.

## 2026-02-16 (extended testing)

### More tests executed
- Ran compile check:
- `python -m py_compile src/alg3/inference.py src/alg3/modeling.py src/alg3/clr.py tests/test_alg3_run.py scripts/run_alg3_smoke.py`
- Ran unit tests:
- `python -m unittest discover -s tests -v` (all passed)
- Ran smoke test:
- `python scripts/run_alg3_smoke.py` (passed)
- Ran extended matrix validation (custom script):
- 10 cases across alg3/alg4 with `burn/thin/collect_traces` combinations.
- Included edge cases: `burn > K`, `thin <= 0`.
- Verified expected stored sample length and trace tensor shapes for each case.
- Verified `collect_traces=False` returns empty heavy traces while keeping `SSE`.
- Ran determinism check (custom script):
- Same seed + same input gives identical `SSE/G/Gamma/F/Beta_h/Beta_g` for both alg3 and alg4.

## 2026-02-16 (scale test)

### Scale benchmark added and executed
- Added `scripts/run_scale_bench.py` for repeatable scale testing (quick mode + `--full`).
- Updated `README.md` to include the new script.

### Full benchmark results (`python scripts/run_scale_bench.py --full`)
- Medium (`N=20, T=120, K=200, burn=50, thin=2`):
- alg3 `collect=True`: 1.997s, `G(75,120)`, `Gamma/F(75,20,120)`, est trace ~2.84 MB
- alg3 `collect=False`: 1.980s, `G/Gamma/F` empty, `SSE(75,)`
- alg4 `collect=True`: 2.062s, `G(75,120)`, `Gamma/F(75,20,120)`, est trace ~2.84 MB
- alg4 `collect=False`: 2.011s, `G/Gamma/F` empty, `SSE(75,)`
- Large (`N=40, T=160, K=400, burn=100, thin=2`):
- alg3 `collect=True`: 10.171s, `G(150,160)`, `Gamma/F(150,40,160)`, est trace ~14.93 MB
- alg3 `collect=False`: 10.429s, `G/Gamma/F` empty, `SSE(150,)`
- alg4 `collect=True`: 10.895s, `G(150,160)`, `Gamma/F(150,40,160)`, est trace ~14.93 MB
- alg4 `collect=False`: 11.450s, `G/Gamma/F` empty, `SSE(150,)`

### Notes
- Shape/length semantics are correct in all scale cases.
- `collect_traces=False` reliably removes heavy trace outputs.
- Runtime gain from disabling traces is case-dependent; the main guaranteed benefit is memory reduction.

## 2026-02-16 (stress-scale execution)

### Stress option and run
- Added `--stress` option to `scripts/run_scale_bench.py`.
- Stress profile: `N=80, T=200, K=800, burn=200, thin=2` (`n_stored=300`).
- Estimated trace memory at `collect_traces=True`: ~74.07 MB.

### Measured results (manual stress runs)
- alg3 `collect=True`: 62.190s, `G(300,200)`, `Gamma/F(300,80,200)`, `SSE(300,)`
- alg3 `collect=False`: 51.896s, `G/Gamma/F` empty, `SSE(300,)`
- alg4 `collect=True`: 58.255s, `G(300,200)`, `Gamma/F(300,80,200)`, `SSE(300,)`
- alg4 `collect=False`: 64.117s, `G/Gamma/F` empty, `SSE(300,)`

### Observation
- Stress scale에서도 shape/length semantics는 정상 동작.
- `collect_traces=False`는 메모리 절감은 확실하지만, 시간 개선은 알고리즘/실행 상황에 따라 달라짐.

## 2026-02-16 (template notebook conversion)

### Notebook creation
- Created `temaplate.ipynb` from `template.py` using `# %%` / `# %% [markdown]` cell markers.
- Preserved markdown/code cell order and notebook metadata (`python3` kernelspec).
- Validated generated notebook JSON structure.

## 2026-02-16 (README language and positioning update)

### README rewrite
- Rewrote `README.md` in English.
- Updated project naming to `CLR Bayesian Alignment`.
- Explicitly stated scope as **Section 2.4.2, Algorithm 3**.
- Added manuscript status note: currently under review at CSDA.
- Clarified that the repository provides implementation/reproducibility material, not full manuscript text.

## 2026-02-16 (repo readiness audit)

### Distribution checks
- Build check passed:
- `python -m build --sdist --wheel`
- Metadata/package check passed:
- `python -m twine check dist/*`

### Current assessment
- Research execution quality is strong (runtime semantics, stress tests, reproducibility checks).
- Repo-level distribution readiness is close but not complete:
- CI pipeline is not yet configured.
- Packaging/release hygiene (artifact ignore policy, release workflow) still needs finalization.

## 2026-02-16 (G restore path in template)

### G post-processing alignment
- Re-checked `section2.4.1_st_v3.ipynb` logic for template restoration.
- Confirmed package already restores template signal as:
- trace level: `G_old = G + Alpha / mean(C)`
- final state: `g_old = g + alpha / mean(c)`
- Updated `../template.ipynb` plotting cell to prioritize restored outputs:
- `G_old` trace mean first
- fallback to `G` raw trace mean
- fallback to final `g_old` (or `g`)

### Validation run
- Quick smoke run verified keys/shapes:
- `collect_traces=True` -> `G_old` shape `(15, 150)`, `g_old` shape `(150,)`
- `collect_traces=False` -> `G_old` empty, `g_old` available `(150,)`
- Notebook JSON integrity check passed for `../template.ipynb`.

## 2026-02-16 (warping-direction consistency + alignment view)

### Root-cause check
- Verified that synthetic data generation and inference used opposite-looking interpolation forms.
- Inference model uses observation-space fit:
- `f_i(t) = a_i + c_i * g(gamma_i^{-1}(t))` implemented as `interp(t, gamma_i, g)`.

### Fix applied
- Updated dataset generator to match inference definition.
- File: `scripts/generate_data_st.py`
- Change: `f_true_i` now uses `np.interp(t, gamma_i, g_true)` (previously `np.interp(gamma_i, t, g_true)`).
- Regenerated `data/data_st.npz`.

### Template notebook visualization update
- Updated parent notebook `../template.ipynb` (cell 13) to reduce interpretation ambiguity:
- Kept observation-space fitted panel (`y` vs `F_mean`).
- Added template-space alignment check panel:
- aligns each curve by posterior `gamma_mean` with `np.interp(gamma_mean[i], t, y[i])`
- applies posterior mean `(a_i, c_i)` correction
- overlays aligned mean against estimated template `G`.

### Validation run
- `python scripts/generate_data_st.py`
- `python -m unittest discover -s tests -v` (all passed)
- smoke check for updated plotting logic: shapes/correlation validated without errors.
