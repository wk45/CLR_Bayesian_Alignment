import numpy as np

# Pure-Python fallback for CLR transforms (mirrors CLR.pyx).

_EPS = 1e-12


def _cumtrapz_1d(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(y, dtype=float)
    for i in range(1, len(x)):
        out[i] = out[i - 1] + (y[i] + y[i - 1]) * (x[i] - x[i - 1]) * 0.5
    return out


def Psi_inv(h, t):
    h = np.asarray(h, dtype=float)
    t = np.asarray(t, dtype=float)
    exp_h = np.exp(h)
    num = _cumtrapz_1d(exp_h, t)
    denom = num[-1]
    if denom <= 0:
        denom = _EPS
    return num / denom


def Psi(gamma, t):
    gamma = np.asarray(gamma, dtype=float)
    t = np.asarray(t, dtype=float)
    gamma_dot = np.gradient(gamma, t)
    gamma_dot = np.maximum(gamma_dot, _EPS)
    log_gamma_dot = np.log(gamma_dot)
    # CLR centering: subtract a scalar integral mean so that int Psi(gamma)(t) dt = 0.
    if hasattr(np, "trapezoid"):
        mean_log = np.trapezoid(log_gamma_dot, t)
    else:
        mean_log = np.trapz(log_gamma_dot, t)
    return log_gamma_dot - mean_log


def Psi_batch(gamma_all, t):
    gamma_all = np.asarray(gamma_all, dtype=float)
    t = np.asarray(t, dtype=float)
    num_obs, n = gamma_all.shape
    h_all = np.zeros((num_obs, n), dtype=float)
    for i in range(num_obs):
        h_all[i] = Psi(gamma_all[i], t)
    return h_all


def Psi_inv_batch(h_all, t):
    h_all = np.asarray(h_all, dtype=float)
    t = np.asarray(t, dtype=float)
    num_obs, n = h_all.shape
    gamma_all = np.zeros((num_obs, n), dtype=float)
    for i in range(num_obs):
        gamma_all[i] = Psi_inv(h_all[i], t)
    return gamma_all
