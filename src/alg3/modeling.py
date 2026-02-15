import numpy as np
from .clr_fallback import Psi, Psi_inv


def compute_covariance(t, v, l, cov_kernel_type):
    if cov_kernel_type == "se":
        C = v * np.exp(- (t[:, None] - t[None, :])**2 / l)
    elif cov_kernel_type == "exp":
        C = v * np.exp(- np.abs(t[:, None] - t[None, :]) / l)
    else:
        raise ValueError(f"Unknown kernel: {cov_kernel_type}")
    L = np.linalg.cholesky(C + 1e-12 * np.eye(len(t)))
    return C, L


def get_ell(rho, N):
    return -1 / ((N - 1) * np.log(rho))


def jeffreys_prior_logpdf(x):
    if x <= 0:
        return -np.inf
    return -np.log(x)


def log_acceptance_ratio(y, f_current, f_proposal, s2y):
    if len(y.shape) == 1:
        f_current = f_current.reshape(1, -1)
        f_proposal = f_proposal.reshape(1, -1)
        y = y.reshape(1, -1)

    N = y.shape[0]
    residual_current = np.array([y[i] - f_current[i] for i in range(N)])
    sse_current = np.sum([residual_current[i] ** 2 for i in range(N)])

    residual_proposal = np.array([y[i] - f_proposal[i] for i in range(N)])
    sse_proposal = np.sum([residual_proposal[i] ** 2 for i in range(N)])

    return -0.5 * (sse_proposal - sse_current) / (s2y**2)


def log_acceptance_ratio_ci(y, f_current, f_proposal, c_current, c_proposal, s2y):
    if len(y.shape) == 1:
        f_current = f_current.reshape(1, -1)
        f_proposal = f_proposal.reshape(1, -1)
        y = y.reshape(1, -1)

    N = y.shape[0]
    residual_current = np.array([y[i] - f_current[i] for i in range(N)])
    residual_proposal = np.array([y[i] - f_proposal[i] for i in range(N)])

    sse_current = np.sum([residual_current[i] ** 2 for i in range(N)])
    sse_proposal = np.sum([residual_proposal[i] ** 2 for i in range(N)])

    return -0.5 * (sse_proposal - sse_current) / (s2y**2) + np.sum(np.log(c_proposal)) - np.sum(np.log(c_current))


def generate_random_warpings(t, a_lower_bound, a_upper_bound, num_obs):
    a = np.random.uniform(a_lower_bound, a_upper_bound, num_obs)
    gamma = np.array([(np.exp(a[i] * t) - 1) / (np.exp(a[i]) - 1) for i in range(num_obs)])
    gamma = np.array([(gamma[i] - np.min(gamma[i]))/(np.max(gamma[i])-np.min(gamma[i])) for i in range(num_obs)])

    gamma_inv = np.array([np.interp(t, gamma[i], t) for i in range(num_obs)])
    hInv = np.array([Psi(gamma_inv[i], t) for i in range(num_obs)])
    hInv_bar = np.mean(hInv, axis=0)
    hInv = hInv - hInv_bar
    gamma_inv = np.array([Psi_inv(hInv[i], t) for i in range(num_obs)])
    gamma = np.array([np.interp(t, gamma_inv[i], t) for i in range(num_obs)])
    h = np.array([Psi(gamma[i], t) for i in range(num_obs)])
    return gamma, h


def simulate_dataset(num_obs_local, num_points_local, sig_y_local):
    t_local = np.linspace(0, 1, num_points_local)
    mu1, sigma1, h1 = 0.06, 0.2, 1.5
    mu2, sigma2, h2 = 0.94, 0.2, 1.5
    mu3, sigma3, h3 = 0.5, 0.15, 1.2
    bump1 = h1 * np.exp(-((t_local - mu1) ** 2) / (2 * sigma1**2))
    bump2 = h2 * np.exp(-((t_local - mu2) ** 2) / (2 * sigma2**2))
    bump3 = h3 * np.exp(-((t_local - mu3) ** 2) / (2 * sigma3**2))
    g_true_local = bump1 + bump2 + bump3
    g_true_local = g_true_local - g_true_local[0]

    gamma0_local, _ = generate_random_warpings(t_local, -1.5, 1.5, num_obs_local)
    f_local = np.array([np.interp(gamma0_local[i], t_local, g_true_local) for i in range(num_obs_local)])

    a = np.random.randn(num_obs_local) * 0.1
    a_true_local = a - np.mean(a)
    c = np.random.gamma(shape=1e1, scale=1e-1, size=num_obs_local)
    c_true_local = c / np.mean(c)

    f_local = a_true_local[:, None] + c_true_local[:, None] * f_local
    eps = np.random.randn(num_obs_local, num_points_local)
    eps -= np.mean(eps, axis=0)[None, :]
    y_local = f_local + sig_y_local * eps
    return t_local, y_local, f_local, g_true_local, gamma0_local, c_true_local, a_true_local


def simulate_dataset_sbc(cfg, num_points_local, num_obs_local, sig_y_local):
    t_local = np.linspace(0, 1, num_points_local)

    Kg_local, Lg_local = compute_covariance(t_local, cfg["nu_g"], cfg["ell_g"], cfg["cov_kernel_g"])
    g_true_local = Lg_local @ np.random.randn(num_points_local)
    g_true_local = g_true_local - g_true_local[0]

    Kh_local, Lh_local = compute_covariance(t_local, cfg["nu_h"], cfg["ell_h"], cfg["cov_kernel_h"])
    gamma0_local = []
    for _ in range(num_obs_local):
        hInv = Lh_local @ np.random.randn(num_points_local)
        hInv = hInv - np.trapezoid(hInv, t_local)
        gamma_i = Psi_inv(hInv, t_local)
        gamma0_local.append(gamma_i)
    gamma0_local = np.array(gamma0_local)

    a_true_local = np.random.randn(num_obs_local) * np.sqrt(cfg["a_hyper"]["var"])
    a_true_local = a_true_local - np.mean(a_true_local)
    c = np.random.gamma(shape=cfg["c_hyper"]["shape"], scale=cfg["c_hyper"]["scale"], size=num_obs_local)
    c_true_local = c / np.mean(c)

    f_local = np.array([np.interp(gamma0_local[i], t_local, g_true_local) for i in range(num_obs_local)])
    f_local = a_true_local[:, None] + c_true_local[:, None] * f_local

    eps = np.random.randn(num_obs_local, num_points_local)
    eps -= np.mean(eps, axis=0)[None, :]
    y_local = f_local + sig_y_local * eps
    return t_local, y_local, f_local, g_true_local, gamma0_local, c_true_local, a_true_local
