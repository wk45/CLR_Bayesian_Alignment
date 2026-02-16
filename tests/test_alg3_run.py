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
    def _run_chain(self, *, K: int, burn: int, thin: int, collect_traces: bool) -> dict[str, object]:
        data_path = ROOT / "data" / "data_st.npz"
        data = np.load(data_path, allow_pickle=True)

        t = data["t"][:80]
        y = data["y"][:8, :80]
        sig_y0 = float(data["sig_y0"])

        return run_alg3_chain(
            y_local=y,
            t=t,
            ell_h=5.0,
            nu_h=1.0,
            ell_g=0.01,
            nu_g=1.0,
            cov_kernel_h="exp",
            cov_kernel_g="se",
            K=K,
            s2y=sig_y0 / 2.0,
            beta_h_choices=np.array([0.05, 0.1, 0.15], dtype=float),
            beta_g_choices=np.array([0.005, 0.01, 0.02], dtype=float),
            beta_adapt=False,
            a_hyper={"mean": 0.0, "var": 0.1},
            c_hyper={"shape": 10.0, "scale": 0.1},
            seed=2,
            burn=burn,
            thin=thin,
            collect_traces=collect_traces,
            display_plots=False,
            verbose=False,
        )

    def test_alg3_burn_and_thin_affect_trace_length(self) -> None:
        out = self._run_chain(K=30, burn=10, thin=1, collect_traces=True)

        g = np.asarray(out["G"])
        gamma = np.asarray(out["Gamma"])
        sse = np.asarray(out["SSE"])

        self.assertIsInstance(out["G"], np.ndarray)
        self.assertIsInstance(out["Gamma"], np.ndarray)
        self.assertEqual(g.shape, (20, 80))
        self.assertEqual(gamma.shape, (20, 8, 80))
        self.assertEqual(sse.shape, (20,))
        self.assertTrue(np.isfinite(out["Acc_h_mean"]))
        self.assertTrue(np.isfinite(out["Acc_g_mean"]))

    def test_alg3_collect_traces_false_omits_heavy_traces(self) -> None:
        out = self._run_chain(K=24, burn=8, thin=4, collect_traces=False)

        g = np.asarray(out["G"])
        gamma = np.asarray(out["Gamma"])
        f = np.asarray(out["F"])
        sse = np.asarray(out["SSE"])

        self.assertEqual(g.shape, (0, 80))
        self.assertEqual(gamma.shape, (0, 8, 80))
        self.assertEqual(f.shape, (0, 8, 80))
        self.assertEqual(sse.shape, (4,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
