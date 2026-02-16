# alg3_standalone

`section2.4.1_st_v3.ipynb`에서 사용한 Alg3 MCMC 코드를 독립 실행 가능 형태로 복사한 폴더입니다.

변경 이력은 `UPDATE_LOG.md`에 누적합니다.

## Included

- `src/alg3/inference.py`: Alg3/Alg4 MCMC core
- `src/alg3/modeling.py`: covariance/data helper
- `src/alg3/clr.py`: pure Python CLR transforms
- `data/data_st.npz`: smoke test용 데이터 복사본
- `scripts/run_alg3_smoke.py`: 최소 실행 검증 스크립트
- `scripts/run_scale_bench.py`: 스케일(중간/대형) 벤치 스크립트

## Quick start

```bash
cd alg3_standalone
python -m pip install -e .
python scripts/run_alg3_smoke.py
python scripts/run_scale_bench.py --full
python scripts/run_scale_bench.py --stress
```

`numba`가 있으면 가속 가능하며, 없으면 pure Python 경로로 동작합니다.
