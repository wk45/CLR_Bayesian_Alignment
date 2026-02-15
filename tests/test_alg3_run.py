from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alg3 import run_alg3_chain


class TestAlg3Run(unittest.TestCase):
    def test_alg3_smoke_run(self):
        data_path = ROOT / "data" / "data_st.npz"
        data = np.load(data_path, allow_pickle=True)

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
            K=30,
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
        gamma_inv = np.asarray(out["GammaInv"])

        self.assertEqual(g.shape, (30, 80))
        self.assertEqual(gamma_inv.shape, (30, 8, 80))
        self.assertTrue(np.isfinite(out["Acc_h_mean"]))
        self.assertTrue(np.isfinite(out["Acc_g_mean"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
