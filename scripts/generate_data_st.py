from __future__ import annotations

from pathlib import Path

import numpy as np


def _count_extrema(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.diff(signal)
    peaks = np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1
    valleys = np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1
    return peaks, valleys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_path = root / "data" / "data_st.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seed = 20260216
    rng = np.random.default_rng(seed)

    num_obs = 10
    num_points = 200
    t = np.linspace(0.0, 1.0, num_points, dtype=np.float64)

    # Peaky/valley-rich underlying signal for clearer visual demos.
    g_true = (
        1.30 * np.sin(2.0 * np.pi * (1.15 * t + 0.10))
        + 0.70 * np.sin(2.0 * np.pi * (3.70 * t + 0.62))
        - 0.45 * np.cos(2.0 * np.pi * (5.90 * t + 0.18))
    )
    g_true = g_true - np.mean(g_true)
    g_true = g_true / (np.std(g_true) + 1e-12)

    peaks, valleys = _count_extrema(g_true)
    if len(peaks) < 3 or len(valleys) < 3:
        raise RuntimeError("g_true is not peaky enough; regeneration aborted.")

    a_true = rng.normal(loc=0.0, scale=0.08, size=num_obs).astype(np.float64)
    c_true = rng.lognormal(mean=0.0, sigma=0.12, size=num_obs).astype(np.float64)

    sig_y0 = 0.05
    f_true = np.zeros((num_obs, num_points), dtype=np.float64)
    y = np.zeros((num_obs, num_points), dtype=np.float64)
    gamma_all = np.zeros((num_obs, num_points), dtype=np.float64)

    for i in range(num_obs):
        amp = 0.12 + 0.02 * (i % 5)
        phase = 0.35 * i
        gamma_raw = t + amp * np.sin(2.0 * np.pi * t + phase) / (2.0 * np.pi)
        gamma_raw = np.clip(gamma_raw, 0.0, 1.0)
        gamma_i = np.maximum.accumulate(gamma_raw)
        gamma_i = (gamma_i - gamma_i[0]) / (gamma_i[-1] - gamma_i[0] + 1e-12)
        gamma_all[i] = gamma_i

        f_true_i = a_true[i] + c_true[i] * np.interp(gamma_i, t, g_true)
        f_true[i] = f_true_i
        y[i] = f_true_i + rng.normal(loc=0.0, scale=sig_y0, size=num_points)

    ys = (y - np.mean(y)) / (np.std(y) + 1e-12)
    gamma0 = np.mean(gamma_all, axis=0)
    gamma0 = (gamma0 - gamma0[0]) / (gamma0[-1] - gamma0[0] + 1e-12)

    np.savez_compressed(
        out_path,
        t=t,
        y=y,
        ys=ys,
        g_true=g_true,
        f_true=f_true,
        gamma0=gamma0,
        a_true=a_true,
        c_true=c_true,
        sig_y0=np.float64(sig_y0),
        num_obs=np.int64(num_obs),
        num_points=np.int64(num_points),
        seed=np.int64(seed),
    )

    print(f"Wrote: {out_path}")
    print(f"g_true extrema: peaks={len(peaks)}, valleys={len(valleys)}")
    print(f"y shape: {y.shape}, t shape: {t.shape}, sig_y0={sig_y0}")


if __name__ == "__main__":
    main()
