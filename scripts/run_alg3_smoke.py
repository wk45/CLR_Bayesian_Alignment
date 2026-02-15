from pathlib import Path
import numpy as np

from alg3 import run_alg3_chain


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "data_st.npz"
    data = np.load(data_path, allow_pickle=True)

    # Small slice for quick standalone smoke execution.
    t = data["t"][:80]
    y = data["y"][:8, :80]
    sig_y0 = float(data["sig_y0"])

    out = run_alg3_chain(
        y_local=y,
        t=t,
        ell_h=5.0,
        nu_h=1.0,
        ell_g=0.01,
        nu_g=1.0,
        cov_kernel_h="exp",
        cov_kernel_g="se",
        K=50,
        s2y=sig_y0 / 2.0,
        beta_h_choices=np.array([0.05, 0.1, 0.15], dtype=float),
        beta_g_choices=np.array([0.005, 0.01, 0.02], dtype=float),
        beta_adapt=False,
        a_hyper={"mean": 0.0, "var": 0.1},
        c_hyper={"shape": 10.0, "scale": 0.1},
        seed=2,
        burn=10,
        thin=1,
        collect_traces=True,
        display_plots=False,
        verbose=False,
        use_python=True,
    )

    g = np.asarray(out["G"])
    gam = np.asarray(out["GammaInv"])
    print("Alg3 smoke run complete")
    print("G shape:", g.shape)
    print("Gamma shape:", gam.shape)
    print("Acc_h_mean:", out.get("Acc_h_mean"))
    print("Acc_g_mean:", out.get("Acc_g_mean"))


if __name__ == "__main__":
    main()
