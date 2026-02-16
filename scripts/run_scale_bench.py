from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alg3.inference import run_alg3_chain, run_alg4_chain


def build_scaled_data(base_t: np.ndarray, base_y: np.ndarray, num_obs: int, num_points: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(float(base_t.min()), float(base_t.max()), num_points)
    y = np.zeros((num_obs, num_points), dtype=np.float64)
    for i in range(num_obs):
        src = base_y[i % base_y.shape[0]]
        y[i] = np.interp(t, base_t, src) + 0.005 * (i + 1)
    return t, y


def estimate_trace_mb(n_stored: int, num_obs: int, num_points: int, collect_traces: bool) -> float:
    sse_bytes = n_stored * 8
    if not collect_traces:
        return sse_bytes / (1024.0**2)
    g_bytes = n_stored * num_points * 8
    gamma_bytes = n_stored * num_obs * num_points * 8
    f_bytes = n_stored * num_obs * num_points * 8
    a_bytes = n_stored * num_obs * 8
    c_bytes = n_stored * num_obs * 8
    alpha_bytes = n_stored * 8
    total = sse_bytes + g_bytes + gamma_bytes + f_bytes + a_bytes + c_bytes + alpha_bytes
    return total / (1024.0**2)


def expected_stored(K: int, burn: int, thin: int) -> int:
    thin = max(int(thin), 1)
    burn = min(max(int(burn), 0), int(K))
    n_kept = int(K) - burn
    if n_kept <= 0:
        return 0
    return (n_kept + thin - 1) // thin


def run_one_case(
    algo: str,
    y: np.ndarray,
    t: np.ndarray,
    s2y: float,
    K: int,
    burn: int,
    thin: int,
    collect_traces: bool,
    seed: int,
) -> dict[str, Any]:
    fn = run_alg3_chain if algo == "alg3" else run_alg4_chain
    kwargs: dict[str, Any] = {
        "y_local": y,
        "t": t,
        "ell_h": 5.0,
        "nu_h": 1.0,
        "ell_g": 0.01,
        "nu_g": 1.0,
        "cov_kernel_h": "exp",
        "cov_kernel_g": "se",
        "K": K,
        "burn": burn,
        "thin": thin,
        "collect_traces": collect_traces,
        "s2y": s2y,
        "beta_h_choices": np.array([0.05, 0.1], dtype=float),
        "beta_g_choices": np.array([0.005, 0.01], dtype=float),
        "beta_adapt": False,
        "a_hyper": {"mean": 0.0, "var": 0.1},
        "c_hyper": {"shape": 10.0, "scale": 0.1},
        "seed": seed,
        "display_plots": False,
        "verbose": False,
    }
    if algo == "alg4":
        kwargs["trim_frac"] = 0.1

    t0 = time.perf_counter()
    out = fn(**kwargs)
    elapsed = time.perf_counter() - t0

    G = np.asarray(out["G"])
    Gamma = np.asarray(out["Gamma"])
    F = np.asarray(out["F"])
    SSE = np.asarray(out["SSE"])

    return {
        "algo": algo,
        "K": K,
        "burn": burn,
        "thin": thin,
        "collect_traces": collect_traces,
        "elapsed_s": elapsed,
        "G_shape": G.shape,
        "Gamma_shape": Gamma.shape,
        "F_shape": F.shape,
        "SSE_shape": SSE.shape,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale benchmark for alg3/alg4.")
    parser.add_argument("--full", action="store_true", help="Run larger scale cases in addition to quick cases.")
    parser.add_argument("--stress", action="store_true", help="Include stress scale cases (may take longer).")
    args = parser.parse_args()

    data = np.load(ROOT / "data" / "data_st.npz", allow_pickle=True)
    base_t = np.asarray(data["t"][:80], dtype=np.float64)
    base_y = np.asarray(data["y"][:8, :80], dtype=np.float64)
    s2y = float(data["sig_y0"]) / 2.0

    scales: list[tuple[str, int, int, int, int]] = [
        ("medium", 20, 120, 200, 50),
    ]
    if args.full:
        scales.append(("large", 40, 160, 400, 100))
    if args.stress:
        scales.append(("stress", 80, 200, 800, 200))

    print("Scale benchmark start")
    print("label algo collect K burn thin N T n_stored est_trace_mb elapsed_s")

    for label, n_obs, n_points, K, burn in scales:
        t, y = build_scaled_data(base_t, base_y, n_obs, n_points)
        for algo in ("alg3", "alg4"):
            for collect in (True, False):
                thin = 2
                n_stored = expected_stored(K, burn, thin)
                result = run_one_case(
                    algo=algo,
                    y=y,
                    t=t,
                    s2y=s2y,
                    K=K,
                    burn=burn,
                    thin=thin,
                    collect_traces=collect,
                    seed=11,
                )
                est_mb = estimate_trace_mb(n_stored, n_obs, n_points, collect)
                print(
                    f"{label} {algo} {collect} {K} {burn} {thin} {n_obs} {n_points} "
                    f"{n_stored} {est_mb:.2f} {result['elapsed_s']:.3f}"
                )
                print(
                    f" shapes G={result['G_shape']} Gamma={result['Gamma_shape']} "
                    f"F={result['F_shape']} SSE={result['SSE_shape']}"
                )


if __name__ == "__main__":
    main()
