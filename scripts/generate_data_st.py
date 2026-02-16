from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

# Match section2.4.1_st_data.ipynb defaults (v3 dataset recipe).
DATA_SEED = 42
NUM_OBS = 10
NUM_POINTS = 200
SIG_Y0 = 0.2


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_path = root / "data" / "data_st.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from alg3.modeling import generate_random_warpings

    np.random.seed(DATA_SEED)

    num_obs = int(NUM_OBS)
    num_points = int(NUM_POINTS)
    t = np.linspace(0.0, 1.0, num_points, dtype=np.float64)

    mu1, sigma1, h1 = 0.06, 0.2, 1.5
    mu2, sigma2, h2 = 0.94, 0.2, 1.5
    mu3, sigma3, h3 = 0.5, 0.15, 1.2
    bump1 = h1 * np.exp(-((t - mu1) ** 2) / (2 * sigma1**2))
    bump2 = h2 * np.exp(-((t - mu2) ** 2) / (2 * sigma2**2))
    bump3 = h3 * np.exp(-((t - mu3) ** 2) / (2 * sigma3**2))
    g_true = bump1 + bump2 + bump3
    g_true = g_true - g_true[0]

    # section2.4.1 v3 notation:
    # gamma0 is used directly in f_i(t) = g_true(gamma0_i(t)).
    gamma0, _ = generate_random_warpings(t, -1.5, 1.5, num_obs)
    f_warp = np.array([np.interp(gamma0[i], t, g_true) for i in range(num_obs)], dtype=np.float64)

    sig_y0 = float(SIG_Y0)
    a = np.random.randn(num_obs) * 0.1
    a_true = a - np.mean(a)
    c = np.random.gamma(shape=1e1, scale=1e-1, size=num_obs)
    c_true = c / np.mean(c)

    f_true = a_true[:, None] + c_true[:, None] * f_warp
    epsilon = np.random.randn(num_obs, num_points)
    epsilon -= np.mean(epsilon, axis=0)[None, :]
    y = f_true + sig_y0 * epsilon
    ys = y.copy()

    np.savez_compressed(
        out_path,
        t=t,
        y=y,
        ys=ys,
        g_true=g_true,
        f_true=f_true,
        gamma0=gamma0,
        # Keep both names for notation compatibility.
        # a_true/c_true: existing key names
        # ai_true/ci_true: explicit indexed notation aliases
        a_true=a_true,
        c_true=c_true,
        ai_true=a_true,
        ci_true=c_true,
        sig_y0=np.float64(sig_y0),
        num_obs=np.int64(num_obs),
        num_points=np.int64(num_points),
        seed=np.int64(DATA_SEED),
    )

    print(f"Wrote: {out_path}")
    print(f"y shape: {y.shape}, t shape: {t.shape}, sig_y0={sig_y0}")


if __name__ == "__main__":
    main()
