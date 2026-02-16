import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
try:
    from numba import jit
except Exception:
    def jit(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            func.py_func = func
            return func
        return _decorator

from .modeling import compute_covariance


import json
import hashlib


def _hash_array(arr: ArrayLike) -> str:
    arr = np.asarray(arr)
    h = hashlib.sha256()
    h.update(arr.tobytes())
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    return h.hexdigest()


def make_cache_key(params: Mapping[str, Any], y: ArrayLike, t: ArrayLike) -> str:
    payload: dict[str, Any] = dict(params)
    payload["y_hash"] = _hash_array(y)
    payload["t_hash"] = _hash_array(t)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def save_cache(path: str | Path, params: Mapping[str, Any], arrays: Mapping[str, ArrayLike]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params_json = json.dumps(params, sort_keys=True)
    np.savez_compressed(path, params_json=params_json, **arrays)
    return path


def load_cache(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    params_json = data["params_json"]
    if hasattr(params_json, "item"):
        params_json = params_json.item()
    params = json.loads(str(params_json))
    arrays: dict[str, NDArray[np.float64]] = {k: data[k] for k in data.files if k != "params_json"}
    return {"params": params, "arrays": arrays, "path": str(path)}


@jit(nopython=True)
def numba_trapz_1d(y: NDArray[np.float64], dt: float) -> float:
    s = 0.0
    n = y.shape[0]
    for i in range(n-1):
        s += 0.5 * (y[i] + y[i+1]) * dt
    return s

@jit(nopython=True)
def numba_cumtrapz_1d(y: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
    n = y.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[0] = 0.0
    s = 0.0
    for i in range(n-1):
        s += 0.5 * (y[i] + y[i+1]) * dt
        out[i+1] = s
    return out

@jit(nopython=True)
def numba_interp(x: NDArray[np.float64], xp: NDArray[np.float64], fp: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.interp(x, xp, fp)

@jit(nopython=True)
def numba_grad_1d(y: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
    n = y.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 1:
        out[0] = 0.0
        return out
    out[0] = (y[1] - y[0]) / dt
    out[n-1] = (y[n-1] - y[n-2]) / dt
    for i in range(1, n-1):
        out[i] = (y[i+1] - y[i-1]) / (2.0 * dt)
    return out

@jit(nopython=True)
def Psi_inv_numba(h: NDArray[np.float64], t: NDArray[np.float64]) -> NDArray[np.float64]:
    dt = t[1] - t[0]
    exp_h = np.exp(h)
    gam = numba_cumtrapz_1d(exp_h, dt)
    scale = gam[-1]
    if scale < 1e-12: scale = 1.0
    return gam / scale

@jit(nopython=True)
def log_acceptance_ratio_numba(
    y_row: NDArray[np.float64],
    f_curr: NDArray[np.float64],
    f_prop: NDArray[np.float64],
    s2y: float,
) -> float:
    n = y_row.shape[0]
    sse_curr = 0.0
    sse_prop = 0.0
    for i in range(n):
        sse_curr += (y_row[i] - f_curr[i])**2
        sse_prop += (y_row[i] - f_prop[i])**2
    return -0.5 * (sse_prop - sse_curr) / (s2y**2)

@jit(nopython=True)
def log_prior_gamma_pdf(x: float, shape: float, scale: float) -> float:
    # Log PDF of Gamma(shape, scale)
    if x <= 0: return -1e10
    return (shape - 1.0) * np.log(x) - x / scale

@jit(nopython=True)
def sse_row_numba(y_row: NDArray[np.float64], f_row: NDArray[np.float64]) -> float:
    sse = 0.0
    n = y_row.shape[0]
    for i in range(n):
        diff = y_row[i] - f_row[i]
        sse += diff * diff
    return sse

@jit(nopython=True)
def run_mcmc_alg3_numba(
    y: NDArray[np.float64],
    t: NDArray[np.float64],
    Lh: NDArray[np.float64],
    Lg: NDArray[np.float64],
    h: NDArray[np.float64],
    gamma_h: NDArray[np.float64],
    h_bar: NDArray[np.float64],
    g: NDArray[np.float64],
    a: NDArray[np.float64],
    c: NDArray[np.float64],
    f: NDArray[np.float64],
    alpha: float,
    K: int,
    s2y: float,
    beta_h: float,
    beta_g: float,
    beta_h_choices: NDArray[np.float64],
    beta_g_choices: NDArray[np.float64],
    use_beta_choices_h: int,
    use_beta_choices_g: int,
    beta_h_fixed_prob: float,
    beta_g_fixed_prob: float,
    beta_fixed: float,
    c_hyper_shape: float,
    c_hyper_scale: float,
    a_hyper_mean: float,
    a_hyper_var: float,
    seed: int,
    thin: int,
    burn_iter: int,
    collect_traces: int,
    update_s2y: bool,
    s2y_shape: float,
    s2y_scale: float,
    debug_check: int,
    debug_every: int,
    debug_tol: float,
    progress_every: int,
    no_ca: int,
    adapt_beta: int,
    n_adapt: int,
    adapt_window: int,
    acc_target_h: float,
    acc_target_g: float,
    c_h: float,
    c_g: float,
) -> tuple[Any, ...]:
    np.random.seed(seed)
    num_obs, num_points = y.shape
    dt = t[1] - t[0]
    beta_min = 1e-4
    eps_beta = 1e-6
    beta_max = 0.9
    # Keep initial betas within bounds (hard clip).
    beta_h = min(max(beta_h, beta_min), beta_max)
    beta_g = min(max(beta_g, beta_min), beta_max)
    sqrt_beta_h = np.sqrt(1 - beta_h**2)
    sqrt_beta_g = np.sqrt(1 - beta_g**2)

    if thin < 1:
        thin = 1
    if burn_iter < 0:
        burn_iter = 0
    if burn_iter > K:
        burn_iter = K
    n_kept = K - burn_iter
    n_stored = 0
    if n_kept > 0:
        n_stored = (n_kept + thin - 1) // thin
    if n_stored < 0:
        n_stored = 0
    n_trace_stored = n_stored if collect_traces != 0 else 0
    G_store = np.zeros((n_trace_stored, num_points), dtype=np.float64)
    A_store = np.zeros((n_trace_stored, num_obs, 1), dtype=np.float64)
    Alpha_store = np.zeros((n_trace_stored,), dtype=np.float64)
    C_store = np.zeros((n_trace_stored, num_obs, 1), dtype=np.float64)
    F_store = np.zeros((n_trace_stored, num_obs, num_points), dtype=np.float64)
    Gamma_store = np.zeros((n_trace_stored, num_obs, num_points), dtype=np.float64)
    SSE_store = np.zeros(n_stored, dtype=np.float64)
    Beta_h_trace = np.zeros(K, dtype=np.float64)
    Beta_g_trace = np.zeros(K, dtype=np.float64)
    Acc_h_rate_trace = np.zeros(K, dtype=np.float64)
    Acc_g_trace = np.zeros(K, dtype=np.float64)
    Acc_c_rate_trace = np.zeros(K, dtype=np.float64)

    indices = np.arange(num_obs)
    store_idx = 0
    n_beta_h = beta_h_choices.shape[0] if beta_h_choices is not None else 0
    n_beta_g = beta_g_choices.shape[0] if beta_g_choices is not None else 0
    beta_h_counts = np.zeros(n_beta_h + 1, dtype=np.float64)
    beta_h_accepts = np.zeros(n_beta_h + 1, dtype=np.float64)
    beta_g_counts = np.zeros(n_beta_g + 1, dtype=np.float64)
    beta_g_accepts = np.zeros(n_beta_g + 1, dtype=np.float64)

    # Robbins-Monro logit-shift adaptation (warm-up only)
    do_adapt = (adapt_beta != 0) and ((use_beta_choices_h != 0 and n_beta_h > 0) or (use_beta_choices_g != 0 and n_beta_g > 0))
    if adapt_window < 1:
        adapt_window = 1
    h_hist = np.zeros(adapt_window, dtype=np.float64)
    g_hist = np.zeros(adapt_window, dtype=np.float64)
    h_hist_sum = 0.0
    g_hist_sum = 0.0
    h_hist_idx = 0
    g_hist_idx = 0
    h_hist_count = 0
    g_hist_count = 0
    k_h = 0
    k_g = 0
    s_h = 0.0
    s_g = 0.0
    logit_h_base = np.zeros(n_beta_h, dtype=np.float64)
    logit_g_base = np.zeros(n_beta_g, dtype=np.float64)
    if do_adapt:
        for i in range(n_beta_h):
            x = beta_h_choices[i]
            if x < 1e-6:
                x = 1e-6
            if x > 1.0 - 1e-6:
                x = 1.0 - 1e-6
            logit_h_base[i] = np.log(x / (1.0 - x))
        for i in range(n_beta_g):
            x = beta_g_choices[i]
            if x < 1e-6:
                x = 1e-6
            if x > 1.0 - 1e-6:
                x = 1.0 - 1e-6
            logit_g_base[i] = np.log(x / (1.0 - x))
    sse_i = np.zeros(num_obs, dtype=np.float64)
    sse_total = 0.0
    for i in range(num_obs):
        sse_i[i] = sse_row_numba(y[i], f[i])
        sse_total += sse_i[i]

    for k_global in range(K):
        # Fisher-Yates shuffle
        for i in range(num_obs-1, 0, -1):
            j = np.random.randint(0, i+1)
            tmp = indices[i]
            indices[i] = indices[j]
            indices[j] = tmp

        beta_h_iter_sum = 0.0
        acc_accept_g = False
        h_accept_count = 0
        c_accept_count = 0

        for idx in range(num_obs):
            i = indices[idx]
            if use_beta_choices_h != 0 and beta_h_choices.shape[0] > 0:
                if np.random.uniform(0.0, 1.0) < beta_h_fixed_prob:
                    beta_hi = beta_fixed
                    beta_idx = n_beta_h
                else:
                    beta_idx = np.random.randint(0, beta_h_choices.shape[0])
                    beta_hi = beta_h_choices[beta_idx]
                if beta_hi >= 1.0:
                    beta_hi = 1.0 - eps_beta
                if beta_hi <= 0.0:
                    beta_hi = eps_beta
                sqrt_beta_h_i = np.sqrt(1 - beta_hi**2)
            else:
                beta_hi = beta_h
                sqrt_beta_h_i = sqrt_beta_h
                beta_idx = n_beta_h
            beta_h_iter_sum += beta_hi
            beta_h_counts[beta_idx] += 1.0

            z = np.random.standard_normal(num_points)
            xi = Lh @ z
            xi -= numba_trapz_1d(xi, dt)

            h_prime = h_bar + sqrt_beta_h_i * (h[i] - h_bar) + beta_hi * xi
            h_prime -= numba_trapz_1d(h_prime, dt)
            gamma_h_prime = Psi_inv_numba(h_prime, t)

            f_prime = alpha + a[i] + c[i] * numba_interp(t, gamma_h_prime, g)
            log_alpha = log_acceptance_ratio_numba(y[i], f[i], f_prime, s2y)
            if log_alpha > 0.0:
                log_alpha = 0.0
            accept_h = False
            if np.log(np.random.uniform(0.0, 1.0)) < log_alpha:
                h[i] = h_prime
                gamma_h[i] = gamma_h_prime
                f[i] = f_prime
                accept_h = True
                h_accept_count += 1
                beta_h_accepts[beta_idx] += 1.0
                sse_new = sse_row_numba(y[i], f[i])
                sse_total += sse_new - sse_i[i]
                sse_i[i] = sse_new
            if do_adapt and k_h < n_adapt:
                acc_val = 1.0 if accept_h else 0.0
                if h_hist_count < adapt_window:
                    h_hist_count += 1
                    h_hist_sum += acc_val
                    h_hist[h_hist_idx] = acc_val
                else:
                    h_hist_sum += acc_val - h_hist[h_hist_idx]
                    h_hist[h_hist_idx] = acc_val
                h_hist_idx += 1
                if h_hist_idx >= adapt_window:
                    h_hist_idx = 0
                k_h += 1

            if no_ca == 0:
                # c update (log-normal)
                c_prime = c[i] * np.exp(np.random.standard_normal() * 0.1)
                f_prime = alpha + a[i] + c_prime * numba_interp(t, gamma_h[i], g)
                log_alpha1 = log_acceptance_ratio_numba(y[i], f[i], f_prime, s2y)
                log_prior_prop = log_prior_gamma_pdf(c_prime, c_hyper_shape, c_hyper_scale) + np.log(c_prime)
                log_prior_curr = log_prior_gamma_pdf(c[i], c_hyper_shape, c_hyper_scale) + np.log(c[i])
                log_alpha = log_alpha1 + (log_prior_prop - log_prior_curr)
                if log_alpha > 0.0:
                    log_alpha = 0.0
                if np.log(np.random.uniform(0.0, 1.0)) < log_alpha:
                    c[i] = c_prime
                    f[i] = f_prime
                    c_accept_count += 1
                    sse_new = sse_row_numba(y[i], f[i])
                    sse_total += sse_new - sse_i[i]
                    sse_i[i] = sse_new

                # a update (Gibbs)
                ai_var_post = 1.0 / (1.0 / a_hyper_var + num_points / (s2y**2))
                ai_mean_post = ai_var_post * (
                    a_hyper_mean / a_hyper_var + np.sum(y[i] - (f[i] - a[i])) / (s2y**2)
                )
                ftemp = f[i] - a[i]
                a[i] = np.random.standard_normal() * np.sqrt(ai_var_post) + ai_mean_post
                f[i] = ftemp + a[i]
                sse_new = sse_row_numba(y[i], f[i])
                sse_total += sse_new - sse_i[i]
                sse_i[i] = sse_new

        # centering
        for j in range(num_points):
            h_bar[j] = 0.0
        for i in range(num_obs):
            for j in range(num_points):
                h_bar[j] += h[i, j]
        for j in range(num_points):
            h_bar[j] /= num_obs

        exp_mean = np.exp(h_bar)
        temp_in = numba_cumtrapz_1d(exp_mean, dt)
        scale = numba_trapz_1d(exp_mean, dt)
        if scale < 1e-12:
            scale = 1.0
        for j in range(num_points):
            temp_in[j] = temp_in[j] / scale

        minv = temp_in[0]
        maxv = temp_in[0]
        for j in range(1, num_points):
            if temp_in[j] < minv:
                minv = temp_in[j]
            if temp_in[j] > maxv:
                maxv = temp_in[j]
        denom = maxv - minv
        if denom < 1e-12:
            denom = 1.0
        gamma0_hat = (temp_in - minv) / denom
        gamma0Inv_hat = numba_interp(t, gamma0_hat, t)

        g = numba_interp(gamma0Inv_hat, t, g)
        for i in range(num_obs):
            gamma_h[i] = numba_interp(gamma0Inv_hat, t, gamma_h[i])
            gamma_h_dot = numba_grad_1d(gamma_h[i], dt)
            for j in range(num_points):
                if gamma_h_dot[j] < 1e-12:
                    gamma_h_dot[j] = 1e-12
            log_gamma_h_dot = np.log(gamma_h_dot)
            h[i] = log_gamma_h_dot - numba_trapz_1d(log_gamma_h_dot, dt)
            g_warped = numba_interp(t, gamma_h[i], g)
            f[i] = alpha + a[i] + c[i] * g_warped
            sse_i[i] = sse_row_numba(y[i], f[i])
        sse_total = 0.0
        for i in range(num_obs):
            sse_total += sse_i[i]

        # g update
        if use_beta_choices_g != 0 and beta_g_choices.shape[0] > 0:
            if np.random.uniform(0.0, 1.0) < beta_g_fixed_prob:
                beta_gi = beta_fixed
                beta_g_idx = n_beta_g
            else:
                beta_g_idx = np.random.randint(0, beta_g_choices.shape[0])
                beta_gi = beta_g_choices[beta_g_idx]
            if beta_gi >= 1.0:
                beta_gi = 1.0 - eps_beta
            if beta_gi <= 0.0:
                beta_gi = eps_beta
            sqrt_beta_g_i = np.sqrt(1 - beta_gi**2)
        else:
            beta_gi = beta_g
            sqrt_beta_g_i = sqrt_beta_g
            beta_g_idx = n_beta_g
        beta_g_counts[beta_g_idx] += 1.0
        z = np.random.standard_normal(num_points)
        xi = Lg @ z
        g_prime = sqrt_beta_g_i * g + beta_gi * xi

        log_alpha_total = 0.0
        for i in range(num_obs):
            g_warped_p = numba_interp(t, gamma_h[i], g_prime)
            f_p_i = alpha + a[i] + c[i] * g_warped_p
            log_alpha_total += log_acceptance_ratio_numba(y[i], f[i], f_p_i, s2y)
        if log_alpha_total > 0.0:
            log_alpha_total = 0.0
        if np.log(np.random.uniform(0.0, 1.0)) < log_alpha_total:
            g = g_prime
            acc_accept_g = True
            beta_g_accepts[beta_g_idx] += 1.0
            sse_total = 0.0
            for i in range(num_obs):
                g_warped = numba_interp(t, gamma_h[i], g)
                f[i] = alpha + a[i] + c[i] * g_warped
                sse_i[i] = sse_row_numba(y[i], f[i])
                sse_total += sse_i[i]
        if do_adapt and k_g < n_adapt:
            acc_val = 1.0 if acc_accept_g else 0.0
            if g_hist_count < adapt_window:
                g_hist_count += 1
                g_hist_sum += acc_val
                g_hist[g_hist_idx] = acc_val
            else:
                g_hist_sum += acc_val - g_hist[g_hist_idx]
                g_hist[g_hist_idx] = acc_val
            g_hist_idx += 1
            if g_hist_idx >= adapt_window:
                g_hist_idx = 0
            k_g += 1

        if do_adapt and k_global < n_adapt:
            if h_hist_count > 0 and n_beta_h > 0:
                acc_hat_h = h_hist_sum / h_hist_count
                eta_h = c_h / np.sqrt(max(1.0, float(k_h)))
                s_h += eta_h * (acc_hat_h - acc_target_h)
                for i in range(n_beta_h):
                    z = logit_h_base[i] + s_h
                    u = 1.0 / (1.0 + np.exp(-z))
                    # soft cap: smoothly map u∈(0,1) into [beta_min, beta_max]
                    u = (1.0 - np.exp(-u)) / (1.0 - np.exp(-1.0))
                    beta_h_choices[i] = beta_min + (beta_max - beta_min) * u
            if g_hist_count > 0 and n_beta_g > 0:
                acc_hat_g = g_hist_sum / g_hist_count
                eta_g = c_g / np.sqrt(max(1.0, float(k_g)))
                s_g += eta_g * (acc_hat_g - acc_target_g)
                for i in range(n_beta_g):
                    z = logit_g_base[i] + s_g
                    u = 1.0 / (1.0 + np.exp(-z))
                    # soft cap: smoothly map u∈(0,1) into [beta_min, beta_max]
                    u = (1.0 - np.exp(-u)) / (1.0 - np.exp(-1.0))
                    beta_g_choices[i] = beta_min + (beta_max - beta_min) * u

        # alpha update (global intercept)
        prec_prior_alpha = 1.0 / a_hyper_var
        prec_like_alpha = (num_obs * num_points) / (s2y**2)
        var_post_alpha = 1.0 / (prec_prior_alpha + prec_like_alpha)
        sum_resid_alpha = 0.0
        for i in range(num_obs):
            for j in range(num_points):
                sum_resid_alpha += (y[i, j] - (f[i, j] - alpha))
        mean_post_alpha = var_post_alpha * ((a_hyper_mean * prec_prior_alpha) + (sum_resid_alpha / (s2y**2)))
        alpha_old = alpha
        alpha = np.random.normal(mean_post_alpha, np.sqrt(var_post_alpha))
        delta_alpha = alpha - alpha_old
        if delta_alpha != 0.0:
            sse_total = 0.0
            for i in range(num_obs):
                for j in range(num_points):
                    f[i, j] += delta_alpha
                sse_i[i] = sse_row_numba(y[i], f[i])
                sse_total += sse_i[i]

        # projection: scale -> mean(g)=0 -> mean(a)=0
        eps = 1e-12
        log_c_mean = 0.0
        for i in range(num_obs):
            val = c[i]
            if val < eps:
                val = eps
            log_c_mean += np.log(val)
        log_c_mean /= num_obs
        c_mean = np.exp(log_c_mean)
        if not np.isfinite(c_mean) or c_mean <= 0:
            c_mean = 1.0
        for i in range(num_obs):
            c[i] /= c_mean
        for j in range(num_points):
            g[j] = g[j] * c_mean

        g_mean = 0.0
        for j in range(num_points):
            g_mean += g[j]
        g_mean /= num_points
        for j in range(num_points):
            g[j] -= g_mean
        for i in range(num_obs):
            a[i] += c[i] * g_mean

        a_mean = 0.0
        for i in range(num_obs):
            a_mean += a[i]
        a_mean /= num_obs
        for i in range(num_obs):
            a[i] -= a_mean
        alpha += a_mean

        Acc_h_rate_trace[k_global] = h_accept_count / num_obs
        Acc_g_trace[k_global] = 1.0 if acc_accept_g else 0.0
        Acc_c_rate_trace[k_global] = c_accept_count / num_obs

        if update_s2y:
            # numba path: keep fixed when update_s2y=False; skip sampling for now
            s2y = s2y

        Beta_h_trace[k_global] = beta_h_iter_sum / num_obs
        Beta_g_trace[k_global] = beta_gi

        if debug_check != 0 and debug_every > 0:
            if (k_global + 1) % debug_every == 0:
                max_diff = 0.0
                for i in range(num_obs):
                    g_warped = numba_interp(t, gamma_h[i], g)
                    for j in range(num_points):
                        f_ref = alpha + a[i] + c[i] * g_warped[j]
                        diff = f[i, j] - f_ref
                        if diff < 0.0:
                            diff = -diff
                        if diff > max_diff:
                            max_diff = diff
                if max_diff > debug_tol:
                    print("[Alg3 Debug] f mismatch max diff", max_diff)
        if progress_every > 0 and (k_global + 1) % progress_every == 0:
            print("[Alg3] Iter", k_global + 1, "/", K)

        if k_global >= burn_iter and ((k_global - burn_iter) % thin == 0):
            if store_idx < n_stored:
                if collect_traces != 0:
                    G_store[store_idx] = g
                    for i in range(num_obs):
                        A_store[store_idx, i, 0] = a[i]
                        C_store[store_idx, i, 0] = c[i]
                    Alpha_store[store_idx] = alpha
                    F_store[store_idx] = f
                    Gamma_store[store_idx] = gamma_h
                SSE_store[store_idx] = sse_total
                store_idx += 1

    return (G_store, A_store, Alpha_store, C_store, F_store, Gamma_store, SSE_store,
            a, alpha, c, g, f, h, gamma_h, h_bar,
            Beta_h_trace, Beta_g_trace, Acc_h_rate_trace, Acc_g_trace, s2y,
            beta_h, beta_g, beta_h_counts, beta_h_accepts, beta_g_counts, beta_g_accepts, Acc_c_rate_trace,
            s_h, s_g, beta_h_choices, beta_g_choices)


@jit(nopython=True)
def run_mcmc_alg4_numba(
    y: NDArray[np.float64],
    t: NDArray[np.float64],
    Lh: NDArray[np.float64],
    Lg: NDArray[np.float64],
    h: NDArray[np.float64],
    gamma_h: NDArray[np.float64],
    h_bar: NDArray[np.float64],
    g: NDArray[np.float64],
    a: NDArray[np.float64],
    c: NDArray[np.float64],
    f: NDArray[np.float64],
    alpha: float,
    K: int,
    s2y: float,
    beta_h: float,
    beta_g: float,
    beta_h_choices: NDArray[np.float64],
    beta_g_choices: NDArray[np.float64],
    use_beta_choices_h: int,
    use_beta_choices_g: int,
    beta_h_fixed_prob: float,
    beta_g_fixed_prob: float,
    beta_fixed: float,
    c_hyper_shape: float,
    c_hyper_scale: float,
    a_hyper_mean: float,
    a_hyper_var: float,
    seed: int,
    thin: int,
    update_s2y: bool,
    s2y_shape: float,
    s2y_scale: float,
    debug_check: int,
    debug_every: int,
    debug_tol: float,
    progress_every: int,
    no_ca: int,
    adapt_beta: int,
    n_adapt: int,
    adapt_window: int,
    acc_target_h: float,
    acc_target_g: float,
    c_h: float,
    c_g: float,
    trim_frac: float,
    burn_iter: int,
    collect_traces: int,
) -> tuple[Any, ...]:
    np.random.seed(seed)
    num_obs, num_points = y.shape
    dt = t[1] - t[0]
    beta_min = 1e-4
    eps_beta = 1e-6
    beta_max = 0.9
    beta_h = min(max(beta_h, beta_min), beta_max)
    beta_g = min(max(beta_g, beta_min), beta_max)
    sqrt_beta_h = np.sqrt(1 - beta_h**2)
    sqrt_beta_g = np.sqrt(1 - beta_g**2)

    if thin < 1:
        thin = 1
    if burn_iter < 0:
        burn_iter = 0
    if burn_iter > K:
        burn_iter = K
    n_kept = K - burn_iter
    n_stored = 0
    if n_kept > 0:
        n_stored = (n_kept + thin - 1) // thin
    if n_stored < 0:
        n_stored = 0
    n_trace_stored = n_stored if collect_traces != 0 else 0
    G_store = np.zeros((n_trace_stored, num_points), dtype=np.float64)
    A_store = np.zeros((n_trace_stored, num_obs, 1), dtype=np.float64)
    Alpha_store = np.zeros((n_trace_stored,), dtype=np.float64)
    C_store = np.zeros((n_trace_stored, num_obs, 1), dtype=np.float64)
    F_store = np.zeros((n_trace_stored, num_obs, num_points), dtype=np.float64)
    Gamma_store = np.zeros((n_trace_stored, num_obs, num_points), dtype=np.float64)
    SSE_store = np.zeros(n_stored, dtype=np.float64)
    Beta_h_trace = np.zeros(K, dtype=np.float64)
    Beta_g_trace = np.zeros(K, dtype=np.float64)
    Acc_h_rate_trace = np.zeros(K, dtype=np.float64)
    Acc_g_trace = np.zeros(K, dtype=np.float64)
    Acc_c_rate_trace = np.zeros(K, dtype=np.float64)

    indices = np.arange(num_obs)
    store_idx = 0
    n_beta_h = beta_h_choices.shape[0] if beta_h_choices is not None else 0
    n_beta_g = beta_g_choices.shape[0] if beta_g_choices is not None else 0
    beta_h_counts = np.zeros(n_beta_h + 1, dtype=np.float64)
    beta_h_accepts = np.zeros(n_beta_h + 1, dtype=np.float64)
    beta_g_counts = np.zeros(n_beta_g + 1, dtype=np.float64)
    beta_g_accepts = np.zeros(n_beta_g + 1, dtype=np.float64)

    do_adapt = (adapt_beta != 0) and ((use_beta_choices_h != 0 and n_beta_h > 0) or (use_beta_choices_g != 0 and n_beta_g > 0))
    if adapt_window < 1:
        adapt_window = 1
    h_hist = np.zeros(adapt_window, dtype=np.float64)
    g_hist = np.zeros(adapt_window, dtype=np.float64)
    h_hist_sum = 0.0
    g_hist_sum = 0.0
    h_hist_idx = 0
    g_hist_idx = 0
    h_hist_count = 0
    g_hist_count = 0
    k_h = 0
    k_g = 0
    s_h = 0.0
    s_g = 0.0
    logit_h_base = np.zeros(n_beta_h, dtype=np.float64)
    logit_g_base = np.zeros(n_beta_g, dtype=np.float64)
    if do_adapt:
        for i in range(n_beta_h):
            x = beta_h_choices[i]
            if x < 1e-6:
                x = 1e-6
            if x > 1.0 - 1e-6:
                x = 1.0 - 1e-6
            logit_h_base[i] = np.log(x / (1.0 - x))
        for i in range(n_beta_g):
            x = beta_g_choices[i]
            if x < 1e-6:
                x = 1e-6
            if x > 1.0 - 1e-6:
                x = 1.0 - 1e-6
            logit_g_base[i] = np.log(x / (1.0 - x))

    sse_i = np.zeros(num_obs, dtype=np.float64)
    sse_total = 0.0
    for i in range(num_obs):
        sse_i[i] = sse_row_numba(y[i], f[i])
        sse_total += sse_i[i]

    # trimmed-mean workspace
    h_mean = np.zeros(num_points, dtype=np.float64)
    h_trim = np.zeros(num_points, dtype=np.float64)
    dist = np.zeros(num_obs, dtype=np.float64)
    idx_sort = np.zeros(num_obs, dtype=np.int64)

    for k_global in range(K):
        for i in range(num_obs-1, 0, -1):
            j = np.random.randint(0, i+1)
            tmp = indices[i]
            indices[i] = indices[j]
            indices[j] = tmp

        beta_h_iter_sum = 0.0
        acc_accept_g = False
        h_accept_count = 0
        c_accept_count = 0

        for idx in range(num_obs):
            i = indices[idx]
            if use_beta_choices_h != 0 and beta_h_choices.shape[0] > 0:
                if np.random.uniform(0.0, 1.0) < beta_h_fixed_prob:
                    beta_hi = beta_fixed
                    beta_idx = n_beta_h
                else:
                    beta_idx = np.random.randint(0, beta_h_choices.shape[0])
                    beta_hi = beta_h_choices[beta_idx]
                if beta_hi >= 1.0:
                    beta_hi = 1.0 - eps_beta
                if beta_hi <= 0.0:
                    beta_hi = eps_beta
                sqrt_beta_h_i = np.sqrt(1 - beta_hi**2)
            else:
                beta_hi = beta_h
                sqrt_beta_h_i = sqrt_beta_h
                beta_idx = n_beta_h
            beta_h_iter_sum += beta_hi
            beta_h_counts[beta_idx] += 1.0

            z = np.random.standard_normal(num_points)
            xi = Lh @ z
            xi -= numba_trapz_1d(xi, dt)
            h_prime = h_bar + sqrt_beta_h_i * (h[i] - h_bar) + beta_hi * xi
            h_prime -= numba_trapz_1d(h_prime, dt)
            gamma_h_prime = Psi_inv_numba(h_prime, t)

            f_prime = alpha + a[i] + c[i] * numba_interp(t, gamma_h_prime, g)
            log_alpha = log_acceptance_ratio_numba(y[i], f[i], f_prime, s2y)
            if log_alpha > 0.0:
                log_alpha = 0.0
            accept_h = False
            if np.log(np.random.uniform(0.0, 1.0)) < log_alpha:
                h[i] = h_prime
                gamma_h[i] = gamma_h_prime
                f[i] = f_prime
                accept_h = True
                h_accept_count += 1
                beta_h_accepts[beta_idx] += 1.0
                sse_new = sse_row_numba(y[i], f[i])
                sse_total += sse_new - sse_i[i]
                sse_i[i] = sse_new
            if do_adapt and k_h < n_adapt:
                acc_val = 1.0 if accept_h else 0.0
                if h_hist_count < adapt_window:
                    h_hist_count += 1
                    h_hist_sum += acc_val
                    h_hist[h_hist_idx] = acc_val
                else:
                    h_hist_sum += acc_val - h_hist[h_hist_idx]
                    h_hist[h_hist_idx] = acc_val
                h_hist_idx += 1
                if h_hist_idx >= adapt_window:
                    h_hist_idx = 0
                k_h += 1

            if no_ca == 0:
                c_prime = c[i] * np.exp(np.random.standard_normal() * 0.1)
                f_prime = alpha + a[i] + c_prime * numba_interp(t, gamma_h[i], g)
                log_alpha1 = log_acceptance_ratio_numba(y[i], f[i], f_prime, s2y)
                log_prior_prop = log_prior_gamma_pdf(c_prime, c_hyper_shape, c_hyper_scale) + np.log(c_prime)
                log_prior_curr = log_prior_gamma_pdf(c[i], c_hyper_shape, c_hyper_scale) + np.log(c[i])
                log_alpha = log_alpha1 + (log_prior_prop - log_prior_curr)
                if log_alpha > 0.0:
                    log_alpha = 0.0
                if np.log(np.random.uniform(0.0, 1.0)) < log_alpha:
                    c[i] = c_prime
                    f[i] = f_prime
                    c_accept_count += 1
                    sse_new = sse_row_numba(y[i], f[i])
                    sse_total += sse_new - sse_i[i]
                    sse_i[i] = sse_new

                ai_var_post = 1.0 / (1.0 / a_hyper_var + num_points / (s2y**2))
                ai_mean_post = ai_var_post * (
                    a_hyper_mean / a_hyper_var + np.sum(y[i] - (f[i] - a[i])) / (s2y**2)
                )
                ftemp = f[i] - a[i]
                a[i] = np.random.standard_normal() * np.sqrt(ai_var_post) + ai_mean_post
                f[i] = ftemp + a[i]
                sse_new = sse_row_numba(y[i], f[i])
                sse_total += sse_new - sse_i[i]
                sse_i[i] = sse_new

        if burn_iter > 0 and k_global < burn_iter:
            # trimmed-mean centering for h_bar (burn-in only)
            for j in range(num_points):
                h_mean[j] = 0.0
                for i in range(num_obs):
                    h_mean[j] += h[i, j]
                h_mean[j] /= num_obs
            for i in range(num_obs):
                dist2 = 0.0
                for j in range(num_points):
                    diff = h[i, j] - h_mean[j]
                    dist2 += diff * diff
                dist[i] = dist2 * dt
                idx_sort[i] = i
            for i in range(num_obs - 1):
                min_j = i
                min_v = dist[idx_sort[i]]
                for j in range(i + 1, num_obs):
                    v = dist[idx_sort[j]]
                    if v < min_v:
                        min_v = v
                        min_j = j
                if min_j != i:
                    tmp = idx_sort[i]
                    idx_sort[i] = idx_sort[min_j]
                    idx_sort[min_j] = tmp
            trim = int(trim_frac * num_obs)
            start = 0
            end = num_obs - trim
            if end <= start:
                start = 0
                end = num_obs
            for j in range(num_points):
                h_trim[j] = 0.0
                for ii in range(start, end):
                    h_trim[j] += h[idx_sort[ii], j]
                h_trim[j] /= (end - start)
            h_trim -= numba_trapz_1d(h_trim, dt)
            for j in range(num_points):
                h_bar[j] = h_trim[j]
        else:
            # mean centering after burn-in
            for j in range(num_points):
                h_bar[j] = 0.0
            for i in range(num_obs):
                for j in range(num_points):
                    h_bar[j] += h[i, j]
            for j in range(num_points):
                h_bar[j] /= num_obs

        exp_mean = np.exp(h_bar)
        temp_in = numba_cumtrapz_1d(exp_mean, dt)
        scale = numba_trapz_1d(exp_mean, dt)
        if scale < 1e-12:
            scale = 1.0
        for j in range(num_points):
            temp_in[j] = temp_in[j] / scale

        minv = temp_in[0]
        maxv = temp_in[0]
        for j in range(1, num_points):
            if temp_in[j] < minv:
                minv = temp_in[j]
            if temp_in[j] > maxv:
                maxv = temp_in[j]
        denom = maxv - minv
        if denom < 1e-12:
            denom = 1.0
        gamma0_hat = (temp_in - minv) / denom
        gamma0Inv_hat = numba_interp(t, gamma0_hat, t)

        g = numba_interp(gamma0Inv_hat, t, g)
        for i in range(num_obs):
            gamma_h[i] = numba_interp(gamma0Inv_hat, t, gamma_h[i])
            gamma_h_dot = numba_grad_1d(gamma_h[i], dt)
            for j in range(num_points):
                if gamma_h_dot[j] < 1e-12:
                    gamma_h_dot[j] = 1e-12
            log_gamma_h_dot = np.log(gamma_h_dot)
            h[i] = log_gamma_h_dot - numba_trapz_1d(log_gamma_h_dot, dt)
            g_warped = numba_interp(t, gamma_h[i], g)
            f[i] = alpha + a[i] + c[i] * g_warped
            sse_i[i] = sse_row_numba(y[i], f[i])
        sse_total = 0.0
        for i in range(num_obs):
            sse_total += sse_i[i]

        if use_beta_choices_g != 0 and beta_g_choices.shape[0] > 0:
            if np.random.uniform(0.0, 1.0) < beta_g_fixed_prob:
                beta_gi = beta_fixed
                beta_g_idx = n_beta_g
            else:
                beta_g_idx = np.random.randint(0, beta_g_choices.shape[0])
                beta_gi = beta_g_choices[beta_g_idx]
            if beta_gi >= 1.0:
                beta_gi = 1.0 - eps_beta
            if beta_gi <= 0.0:
                beta_gi = eps_beta
            sqrt_beta_g_i = np.sqrt(1 - beta_gi**2)
        else:
            beta_gi = beta_g
            sqrt_beta_g_i = sqrt_beta_g
            beta_g_idx = n_beta_g
        beta_g_counts[beta_g_idx] += 1.0
        z = np.random.standard_normal(num_points)
        xi = Lg @ z
        g_prime = sqrt_beta_g_i * g + beta_gi * xi

        log_alpha_total = 0.0
        for i in range(num_obs):
            g_warped_p = numba_interp(t, gamma_h[i], g_prime)
            f_p_i = alpha + a[i] + c[i] * g_warped_p
            log_alpha_total += log_acceptance_ratio_numba(y[i], f[i], f_p_i, s2y)
        if log_alpha_total > 0.0:
            log_alpha_total = 0.0
        if np.log(np.random.uniform(0.0, 1.0)) < log_alpha_total:
            g = g_prime
            acc_accept_g = True
            beta_g_accepts[beta_g_idx] += 1.0
            sse_total = 0.0
            for i in range(num_obs):
                g_warped = numba_interp(t, gamma_h[i], g)
                f[i] = alpha + a[i] + c[i] * g_warped
                sse_i[i] = sse_row_numba(y[i], f[i])
                sse_total += sse_i[i]
        if do_adapt and k_g < n_adapt:
            acc_val = 1.0 if acc_accept_g else 0.0
            if g_hist_count < adapt_window:
                g_hist_count += 1
                g_hist_sum += acc_val
                g_hist[g_hist_idx] = acc_val
            else:
                g_hist_sum += acc_val - g_hist[g_hist_idx]
                g_hist[g_hist_idx] = acc_val
            g_hist_idx += 1
            if g_hist_idx >= adapt_window:
                g_hist_idx = 0
            k_g += 1

        if do_adapt and k_global < n_adapt:
            if h_hist_count > 0 and n_beta_h > 0:
                acc_hat_h = h_hist_sum / h_hist_count
                eta_h = c_h / np.sqrt(max(1.0, float(k_h)))
                s_h += eta_h * (acc_hat_h - acc_target_h)
                for i in range(n_beta_h):
                    z = logit_h_base[i] + s_h
                    u = 1.0 / (1.0 + np.exp(-z))
                    u = (1.0 - np.exp(-u)) / (1.0 - np.exp(-1.0))
                    beta_h_choices[i] = beta_min + (beta_max - beta_min) * u
            if g_hist_count > 0 and n_beta_g > 0:
                acc_hat_g = g_hist_sum / g_hist_count
                eta_g = c_g / np.sqrt(max(1.0, float(k_g)))
                s_g += eta_g * (acc_hat_g - acc_target_g)
                for i in range(n_beta_g):
                    z = logit_g_base[i] + s_g
                    u = 1.0 / (1.0 + np.exp(-z))
                    u = (1.0 - np.exp(-u)) / (1.0 - np.exp(-1.0))
                    beta_g_choices[i] = beta_min + (beta_max - beta_min) * u

        prec_prior_alpha = 1.0 / a_hyper_var
        prec_like_alpha = (num_obs * num_points) / (s2y**2)
        var_post_alpha = 1.0 / (prec_prior_alpha + prec_like_alpha)
        sum_resid_alpha = 0.0
        for i in range(num_obs):
            for j in range(num_points):
                sum_resid_alpha += (y[i, j] - (f[i, j] - alpha))
        mean_post_alpha = var_post_alpha * ((a_hyper_mean * prec_prior_alpha) + (sum_resid_alpha / (s2y**2)))
        alpha_old = alpha
        alpha = np.random.normal(mean_post_alpha, np.sqrt(var_post_alpha))
        delta_alpha = alpha - alpha_old
        if delta_alpha != 0.0:
            sse_total = 0.0
            for i in range(num_obs):
                for j in range(num_points):
                    f[i, j] += delta_alpha
                sse_i[i] = sse_row_numba(y[i], f[i])
                sse_total += sse_i[i]

        eps = 1e-12
        log_c_mean = 0.0
        for i in range(num_obs):
            val = c[i]
            if val < eps:
                val = eps
            log_c_mean += np.log(val)
        log_c_mean /= num_obs
        c_mean = np.exp(log_c_mean)
        if not np.isfinite(c_mean) or c_mean <= 0:
            c_mean = 1.0
        for i in range(num_obs):
            c[i] /= c_mean
        for j in range(num_points):
            g[j] = g[j] * c_mean

        g_mean = 0.0
        for j in range(num_points):
            g_mean += g[j]
        g_mean /= num_points
        for j in range(num_points):
            g[j] -= g_mean
        for i in range(num_obs):
            a[i] += c[i] * g_mean

        a_mean = 0.0
        for i in range(num_obs):
            a_mean += a[i]
        a_mean /= num_obs
        for i in range(num_obs):
            a[i] -= a_mean
        alpha += a_mean

        Acc_h_rate_trace[k_global] = h_accept_count / num_obs
        Acc_g_trace[k_global] = 1.0 if acc_accept_g else 0.0
        Acc_c_rate_trace[k_global] = c_accept_count / num_obs

        if update_s2y:
            s2y = s2y

        Beta_h_trace[k_global] = beta_h_iter_sum / num_obs
        Beta_g_trace[k_global] = beta_gi

        if debug_check != 0 and debug_every > 0:
            if (k_global + 1) % debug_every == 0:
                max_diff = 0.0
                for i in range(num_obs):
                    g_warped = numba_interp(t, gamma_h[i], g)
                    for j in range(num_points):
                        f_ref = alpha + a[i] + c[i] * g_warped[j]
                        diff = f[i, j] - f_ref
                        if diff < 0.0:
                            diff = -diff
                        if diff > max_diff:
                            max_diff = diff
                if max_diff > debug_tol:
                    print("[Alg4 Debug] f mismatch max diff", max_diff)
        if progress_every > 0 and (k_global + 1) % progress_every == 0:
            print("[Alg4] Iter", k_global + 1, "/", K)

        if k_global >= burn_iter and ((k_global - burn_iter) % thin == 0):
            if store_idx < n_stored:
                if collect_traces != 0:
                    G_store[store_idx] = g
                    for i in range(num_obs):
                        A_store[store_idx, i, 0] = a[i]
                        C_store[store_idx, i, 0] = c[i]
                    Alpha_store[store_idx] = alpha
                    F_store[store_idx] = f
                    Gamma_store[store_idx] = gamma_h
                SSE_store[store_idx] = sse_total
                store_idx += 1

    return (G_store, A_store, Alpha_store, C_store, F_store, Gamma_store, SSE_store,
            a, alpha, c, g, f, h, gamma_h, h_bar,
            Beta_h_trace, Beta_g_trace, Acc_h_rate_trace, Acc_g_trace, s2y,
            beta_h, beta_g, beta_h_counts, beta_h_accepts, beta_g_counts, beta_g_accepts, Acc_c_rate_trace,
            s_h, s_g, beta_h_choices, beta_g_choices)


def _recover_g_old_from_traces(
    G_trace: NDArray[np.float64],
    C_trace: NDArray[np.float64],
    Alpha_trace: NDArray[np.float64] | None,
    min_c: float = 1e-12,
) -> NDArray[np.float64]:
    if G_trace.size == 0:
        return G_trace
    c_vals = C_trace
    if c_vals.ndim == 3:
        c_vals = c_vals[..., 0]
    c_bar = np.mean(c_vals, axis=1)
    c_bar = np.maximum(c_bar, min_c)
    if Alpha_trace is None or Alpha_trace.size == 0:
        return G_trace
    return G_trace + (Alpha_trace / c_bar)[:, None]



def run_alg3_chain(
    y_local: ArrayLike,
    t: ArrayLike,
    ell_h: float,
    nu_h: float,
    ell_g: float,
    nu_g: float,
    cov_kernel_h: str | None = None,
    cov_kernel_g: str | None = None,
    K: int = 20000,
    dispFreq: int = 5000,
    s2y: float = 0.1,
    beta_h: float = 0.1,
    beta_g: float = 0.1,
    beta_h_choices: ArrayLike | None = None,
    beta_g_choices: ArrayLike | None = None,
    beta_h_fixed_prob: float = 0.1,
    beta_g_fixed_prob: float = 0.02,
    beta_fixed: float = 0.4,
    update_s2y: bool = False,
    s2y_hyper: Mapping[str, float] | None = None,
    a_hyper: Mapping[str, float] | None = None,
    c_hyper: Mapping[str, float] | None = None,
    seed: int = 2,
    burn: int = 0,
    thin: int = 1,
    collect_traces: bool = True,
    display_plots: bool = True,
    y_raw_local: ArrayLike | None = None,
    Y_GLOBAL_MEAN: float = 0.0,
    Y_GLOBAL_STD: float = 1.0,
    g_true: ArrayLike | None = None,
    h_init: ArrayLike | None = None,
    gamma_h_init: ArrayLike | None = None,
    g_init: ArrayLike | None = None,
    a_init: ArrayLike | None = None,
    c_init: ArrayLike | None = None,
    f_init: ArrayLike | None = None,
    use_python: bool = False,
    debug_check: bool = False,
    debug_every: int = 1000,
    debug_tol: float = 1e-6,
    progress_every: int = 0,
    no_ca: bool = False,
    beta_adapt: bool = False,
    beta_n_adapt: int = 0,
    beta_window: int = 50,
    beta_acc_target_h: float = 0.25,
    beta_acc_target_g: float = 0.25,
    beta_c_h: float = 0.15,
    beta_c_g: float = 0.15,
    verbose: bool = True,
) -> dict[str, Any]:
    y_local = np.asarray(y_local, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    K = int(K)
    thin = int(thin)
    if thin < 1:
        thin = 1
    burn = int(burn)
    if burn < 0:
        burn = 0
    if burn > K:
        burn = K
    s2y = float(s2y)
    num_obs, num_points = y_local.shape

    if a_hyper is None:
        a_hyper = {"mean": 0.0, "var": 1000.0}
    if c_hyper is None:
        c_hyper = {"shape": 10.0, "scale": 0.1}

    if beta_h_choices is None or len(beta_h_choices) == 0:
        r_local = np.array([0.999999, 0.99999, 0.9999, 0.999, 0.998, 0.995, 0.99, 0.985, 0.98], dtype=np.float64)
        beta_local = np.sqrt(1.0 - r_local**2)
        beta_h_choices = np.concatenate([beta_local, np.array([0.4], dtype=np.float64)])
    if beta_g_choices is None or len(beta_g_choices) == 0:
        r_local = np.array([0.999999, 0.99999, 0.9999, 0.999, 0.998, 0.995, 0.99, 0.985, 0.98], dtype=np.float64)
        beta_local = np.sqrt(1.0 - r_local**2)
        beta_g_choices = np.concatenate([beta_local, np.array([0.4], dtype=np.float64)])
    use_beta_choices_h = True
    use_beta_choices_g = True

    if update_s2y:
        print("[Alg3] update_s2y=True not supported in numba core; keeping s2y fixed.")

    kernel_h = cov_kernel_h if cov_kernel_h is not None else "se"
    kernel_g = cov_kernel_g if cov_kernel_g is not None else "se"
    if verbose:
        print(f"{'='*30} MCMC Hyperparameters {'='*30}")
        print(f" > Data: Shape={y_local.shape}, Range=[{t[0]:.2f}, {t[-1]:.2f}]")
        print(f" > MCMC Config: K={K}, Burn={burn}, Thin={thin}, Seed={seed}")
        print(f" > Noise: s2y={s2y:.5f} (Fixed)")
        print(f" > Kernel h: type={kernel_h}, ell_h={ell_h:.4f}, nu_h={nu_h:.4f}")
        print(f" > Kernel g: type={kernel_g}, ell_g={ell_g:.4f}, nu_g={nu_g:.4f}")
        print(f" > Prior a: N(mean={a_hyper['mean']}, var={a_hyper['var']})")
        print(f" > Prior c: Gamma(shape={c_hyper['shape']}, scale={c_hyper['scale']})")
        print(f" > No c/a: {bool(no_ca)}")
        if beta_adapt:
            print(f" > Beta sampling: random choices (RM adapt enabled)")
            print(f" > Beta adapt: n_adapt={beta_n_adapt}, window={beta_window}, target_h={beta_acc_target_h}, target_g={beta_acc_target_g}")
        else:
            print(f" > Beta sampling: random choices (adaptive disabled)")
        print(f"{'='*80}")
    try:
        _, Lh = compute_covariance(t, nu_h, ell_h, kernel_h)
        _, Lg = compute_covariance(t, nu_g, ell_g, kernel_g)
    except NameError:
        print("[Warning] compute_covariance not found. Using simple RBF.")
        from scipy.linalg import cholesky
        dists = (t[:, None] - t[None, :])**2
        Ch = nu_h * np.exp(-dists/(ell_h+1e-9)) + 1e-6*np.eye(len(t))
        Lh = cholesky(Ch, lower=True)
        Cg = nu_g * np.exp(-dists/(ell_g+1e-9)) + 1e-6*np.eye(len(t))
        Lg = cholesky(Cg, lower=True)

    Lh = Lh.astype(np.float64)
    Lg = Lg.astype(np.float64)

    if h_init is None:
        h = np.zeros((num_obs, num_points), dtype=np.float64)
    else:
        h = np.asarray(h_init, dtype=np.float64).copy()

    if g_init is None:
        g = np.zeros(num_points, dtype=np.float64)
    else:
        g = np.asarray(g_init, dtype=np.float64).copy()

    if a_init is None:
        a = np.zeros(num_obs, dtype=np.float64)
    else:
        a = np.asarray(a_init, dtype=np.float64).reshape(-1).copy()

    if c_init is None:
        c = np.ones(num_obs, dtype=np.float64)
    else:
        c = np.asarray(c_init, dtype=np.float64).reshape(-1).copy()

    alpha = 0.0

    if gamma_h_init is None:
        gamma_h = np.tile(t, (num_obs, 1)).astype(np.float64)
        if h_init is not None:
            for i in range(num_obs):
                gamma_h[i] = Psi_inv_numba(h[i], t)
    else:
        gamma_h = np.asarray(gamma_h_init, dtype=np.float64).copy()

    h_bar = np.zeros(num_points, dtype=np.float64)
    f = np.zeros((num_obs, num_points), dtype=np.float64)

    if no_ca:
        for i in range(num_obs):
            a[i] = 0.0
            c[i] = 1.0
    if f_init is None:
        for i in range(num_obs):
            f[i] = alpha + a[i] + c[i] * np.interp(t, gamma_h[i], g)
    else:
        f = np.asarray(f_init, dtype=np.float64).copy()

    start_time = time.time()

    core_fn = run_mcmc_alg3_numba.py_func if use_python else run_mcmc_alg3_numba
    res = core_fn(
        y_local, t, Lh, Lg,
        h, gamma_h, h_bar, g, a, c, f, alpha,
        K,
        s2y, float(beta_h), float(beta_g),
        np.asarray(beta_h_choices if use_beta_choices_h else np.empty(0), dtype=np.float64),
        np.asarray(beta_g_choices if use_beta_choices_g else np.empty(0), dtype=np.float64),
        1 if use_beta_choices_h else 0,
        1 if use_beta_choices_g else 0,
        float(beta_h_fixed_prob), float(beta_g_fixed_prob), float(beta_fixed),
        float(c_hyper["shape"]), float(c_hyper["scale"]),
        float(a_hyper["mean"]), float(a_hyper["var"]),
        seed, thin, burn, 1 if collect_traces else 0, update_s2y,
        float(s2y_hyper["shape"]) if s2y_hyper is not None else 0.0,
        float(s2y_hyper["scale"]) if s2y_hyper is not None else 0.0,
        1 if debug_check else 0, int(debug_every), float(debug_tol), int(progress_every),
        1 if no_ca else 0,
        1 if beta_adapt else 0, int(beta_n_adapt), int(beta_window),
        float(beta_acc_target_h), float(beta_acc_target_g), float(beta_c_h), float(beta_c_g),
    )

    (G_final, A_final, Alpha_final, C_final, F_final, Gamma_final, SSE_final,
     a, alpha, c, g, f, h, gamma_h, h_bar,
     Beta_h_final, Beta_g_final, Acc_h_final, Acc_g_final, s2y,
     beta_h, beta_g, beta_h_counts, beta_h_accepts, beta_g_counts, beta_g_accepts, Acc_c_final,
     beta_shift_h, beta_shift_g, beta_h_adapted, beta_g_adapted) = res

    if display_plots and verbose:
        elapsed = time.time() - start_time
        print(f"[Alg3] Iter [{K}/{K}] | Elapsed: {elapsed:.1f}s")

    if SSE_final.size > 0:
        LogLikihood = -0.5 * SSE_final / (s2y**2)
        S2y_list = np.full(SSE_final.shape, s2y, dtype=np.float64)
    else:
        LogLikihood = np.empty(0, dtype=np.float64)
        S2y_list = np.empty(0, dtype=np.float64)

    if collect_traces:
        G_old = _recover_g_old_from_traces(G_final, C_final, Alpha_final)
    else:
        G_old = np.empty(0, dtype=np.float64)

    if Acc_h_final.size > 0:
        acc_h_list = Acc_h_final
        acc_h_mean = float(np.mean(Acc_h_final))
    else:
        acc_h_list = np.ones(SSE_final.shape, dtype=np.float64)
        acc_h_mean = float("nan")
    if Acc_g_final.size > 0:
        acc_g_list = Acc_g_final
        acc_g_mean = float(np.mean(Acc_g_final))
    else:
        acc_g_list = np.ones(SSE_final.shape, dtype=np.float64)
        acc_g_mean = float("nan")
    if Acc_c_final.size > 0:
        acc_c_list = Acc_c_final
        acc_c_mean = float(np.mean(Acc_c_final))
    else:
        acc_c_list = np.ones(SSE_final.shape, dtype=np.float64)
        acc_c_mean = float("nan")
    acc_dummy = np.ones(SSE_final.shape, dtype=np.float64)

    c_bar = float(np.mean(c)) if c.size > 0 else 1.0
    if c_bar <= 0:
        c_bar = 1.0
    g_old_final = g + alpha / c_bar

    beta_h_choices_list = np.asarray(beta_h_choices, dtype=np.float64) if beta_h_choices is not None else np.empty(0, dtype=np.float64)
    beta_g_choices_list = np.asarray(beta_g_choices, dtype=np.float64) if beta_g_choices is not None else np.empty(0, dtype=np.float64)
    beta_h_adapted_list = np.asarray(beta_h_adapted, dtype=np.float64) if beta_h_adapted is not None else np.empty(0, dtype=np.float64)
    beta_g_adapted_list = np.asarray(beta_g_adapted, dtype=np.float64) if beta_g_adapted is not None else np.empty(0, dtype=np.float64)
    beta_h_counts_list = np.asarray(beta_h_counts, dtype=np.float64) if beta_h_counts is not None else np.empty(0, dtype=np.float64)
    beta_h_accepts_list = np.asarray(beta_h_accepts, dtype=np.float64) if beta_h_accepts is not None else np.empty(0, dtype=np.float64)
    beta_g_counts_list = np.asarray(beta_g_counts, dtype=np.float64) if beta_g_counts is not None else np.empty(0, dtype=np.float64)
    beta_g_accepts_list = np.asarray(beta_g_accepts, dtype=np.float64) if beta_g_accepts is not None else np.empty(0, dtype=np.float64)

    beta_h_accept_rate = np.divide(
        beta_h_accepts_list,
        beta_h_counts_list,
        out=np.full(beta_h_accepts_list.shape, np.nan, dtype=np.float64),
        where=beta_h_counts_list != 0,
    )
    beta_g_accept_rate = np.divide(
        beta_g_accepts_list,
        beta_g_counts_list,
        out=np.full(beta_g_accepts_list.shape, np.nan, dtype=np.float64),
        where=beta_g_counts_list != 0,
    )

    res = {
        "gamma_h": gamma_h, "g": g, "g_old": g_old_final, "a": a, "c": c, "f": f,
        "Gamma": Gamma_final,
        "G": G_final, "G_old": G_old, "G_raw": G_final,
        "A": A_final, "Alpha": Alpha_final, "C": C_final, "F": F_final,
        "SSE": SSE_final, "LogLikihood": LogLikihood, "S2y": S2y_list,
        "Acc_h": acc_h_list, "Acc_rate_h": acc_h_list,
        "Acc_g": acc_g_list, "Acc_c": acc_c_list, "Acc_a": acc_dummy,
        "Acc_h_mean": acc_h_mean, "Acc_g_mean": acc_g_mean, "Acc_c_mean": acc_c_mean,
        "Beta_h": Beta_h_final, "Beta_g": Beta_g_final,
        "final_s2y": s2y,
        "Beta_h_choices": beta_h_choices_list,
        "Beta_g_choices": beta_g_choices_list,
        "Beta_h_adapted": beta_h_adapted_list,
        "Beta_g_adapted": beta_g_adapted_list,
        "Beta_shift_h": float(beta_shift_h),
        "Beta_shift_g": float(beta_shift_g),
        "Beta_adapt": bool(beta_adapt),
        "Beta_n_adapt": int(beta_n_adapt),
        "Beta_window": int(beta_window),
        "Beta_acc_target_h": float(beta_acc_target_h),
        "Beta_acc_target_g": float(beta_acc_target_g),
        "Beta_h_counts": beta_h_counts_list,
        "Beta_h_accepts": beta_h_accepts_list,
        "Beta_g_counts": beta_g_counts_list,
        "Beta_g_accepts": beta_g_accepts_list,
        "Beta_h_accept_rate": beta_h_accept_rate,
        "Beta_g_accept_rate": beta_g_accept_rate,
    }

    if verbose:
        print("[Alg3 Numba] MCMC Done.")
    return res


def run_alg4_chain(
    y_local: ArrayLike,
    t: ArrayLike,
    ell_h: float,
    nu_h: float,
    ell_g: float,
    nu_g: float,
    cov_kernel_h: str | None = None,
    cov_kernel_g: str | None = None,
    K: int = 20000,
    dispFreq: int = 5000,
    s2y: float = 0.1,
    beta_h: float = 0.1,
    beta_g: float = 0.1,
    beta_h_choices: ArrayLike | None = None,
    beta_g_choices: ArrayLike | None = None,
    beta_h_fixed_prob: float = 0.1,
    beta_g_fixed_prob: float = 0.02,
    beta_fixed: float = 0.4,
    update_s2y: bool = False,
    s2y_hyper: Mapping[str, float] | None = None,
    a_hyper: Mapping[str, float] | None = None,
    c_hyper: Mapping[str, float] | None = None,
    seed: int = 2,
    burn: int = 0,
    thin: int = 1,
    collect_traces: bool = True,
    display_plots: bool = True,
    y_raw_local: ArrayLike | None = None,
    Y_GLOBAL_MEAN: float = 0.0,
    Y_GLOBAL_STD: float = 1.0,
    g_true: ArrayLike | None = None,
    h_init: ArrayLike | None = None,
    gamma_h_init: ArrayLike | None = None,
    g_init: ArrayLike | None = None,
    a_init: ArrayLike | None = None,
    c_init: ArrayLike | None = None,
    f_init: ArrayLike | None = None,
    use_python: bool = False,
    debug_check: bool = False,
    debug_every: int = 1000,
    debug_tol: float = 1e-6,
    progress_every: int = 0,
    no_ca: bool = False,
    beta_adapt: bool = False,
    beta_n_adapt: int = 0,
    beta_window: int = 50,
    beta_acc_target_h: float = 0.25,
    beta_acc_target_g: float = 0.25,
    beta_c_h: float = 0.15,
    beta_c_g: float = 0.15,
    trim_frac: float = 0.1,
    verbose: bool = True,
) -> dict[str, Any]:
    y_local = np.asarray(y_local, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    K = int(K)
    thin = int(thin)
    if thin < 1:
        thin = 1
    burn = int(burn)
    if burn < 0:
        burn = 0
    if burn > K:
        burn = K
    s2y = float(s2y)
    num_obs, num_points = y_local.shape

    if a_hyper is None:
        a_hyper = {"mean": 0.0, "var": 1000.0}
    if c_hyper is None:
        c_hyper = {"shape": 10.0, "scale": 0.1}

    if beta_h_choices is None or len(beta_h_choices) == 0:
        r_local = np.array([0.999999, 0.99999, 0.9999, 0.999, 0.998, 0.995, 0.99, 0.985, 0.98], dtype=np.float64)
        beta_local = np.sqrt(1.0 - r_local**2)
        beta_h_choices = np.concatenate([beta_local, np.array([0.4], dtype=np.float64)])
    if beta_g_choices is None or len(beta_g_choices) == 0:
        r_local = np.array([0.999999, 0.99999, 0.9999, 0.999, 0.998, 0.995, 0.99, 0.985, 0.98], dtype=np.float64)
        beta_local = np.sqrt(1.0 - r_local**2)
        beta_g_choices = np.concatenate([beta_local, np.array([0.4], dtype=np.float64)])
    use_beta_choices_h = True
    use_beta_choices_g = True

    if update_s2y:
        print("[Alg4] update_s2y=True not supported in numba core; keeping s2y fixed.")

    kernel_h = cov_kernel_h if cov_kernel_h is not None else "se"
    kernel_g = cov_kernel_g if cov_kernel_g is not None else "se"
    if verbose:
        print(f"{'='*30} MCMC Hyperparameters {'='*30}")
        print(f" > Data: Shape={y_local.shape}, Range=[{t[0]:.2f}, {t[-1]:.2f}]")
        print(f" > MCMC Config: K={K}, Burn={burn}, Thin={thin}, Seed={seed}")
        print(f" > Noise: s2y={s2y:.5f} (Fixed)")
        print(f" > Kernel h: type={kernel_h}, ell_h={ell_h:.4f}, nu_h={nu_h:.4f}")
        print(f" > Kernel g: type={kernel_g}, ell_g={ell_g:.4f}, nu_g={nu_g:.4f}")
        print(f" > Prior a: N(mean={a_hyper['mean']}, var={a_hyper['var']})")
        print(f" > Prior c: Gamma(shape={c_hyper['shape']}, scale={c_hyper['scale']})")
        print(f" > No c/a: {bool(no_ca)}")
        if beta_adapt:
            print(f" > Beta sampling: random choices (RM adapt enabled)")
            print(f" > Beta adapt: n_adapt={beta_n_adapt}, window={beta_window}, target_h={beta_acc_target_h}, target_g={beta_acc_target_g}")
        else:
            print(f" > Beta sampling: random choices (adaptive disabled)")
        print(f" > Trim frac (h mean): {trim_frac}")
        print(f"{'='*80}")
    try:
        _, Lh = compute_covariance(t, nu_h, ell_h, kernel_h)
        _, Lg = compute_covariance(t, nu_g, ell_g, kernel_g)
    except NameError:
        print("[Warning] compute_covariance not found. Using simple RBF.")
        from scipy.linalg import cholesky
        dists = (t[:, None] - t[None, :])**2
        Ch = nu_h * np.exp(-dists/(ell_h+1e-9)) + 1e-6*np.eye(len(t))
        Lh = cholesky(Ch, lower=True)
        Cg = nu_g * np.exp(-dists/(ell_g+1e-9)) + 1e-6*np.eye(len(t))
        Lg = cholesky(Cg, lower=True)

    Lh = Lh.astype(np.float64)
    Lg = Lg.astype(np.float64)

    if h_init is None:
        h = np.zeros((num_obs, num_points), dtype=np.float64)
    else:
        h = np.asarray(h_init, dtype=np.float64).copy()

    if g_init is None:
        g = np.zeros(num_points, dtype=np.float64)
    else:
        g = np.asarray(g_init, dtype=np.float64).copy()

    if a_init is None:
        a = np.zeros(num_obs, dtype=np.float64)
    else:
        a = np.asarray(a_init, dtype=np.float64).reshape(-1).copy()

    if c_init is None:
        c = np.ones(num_obs, dtype=np.float64)
    else:
        c = np.asarray(c_init, dtype=np.float64).reshape(-1).copy()

    alpha = 0.0

    if gamma_h_init is None:
        gamma_h = np.tile(t, (num_obs, 1)).astype(np.float64)
        if h_init is not None:
            for i in range(num_obs):
                gamma_h[i] = Psi_inv_numba(h[i], t)
    else:
        gamma_h = np.asarray(gamma_h_init, dtype=np.float64).copy()

    h_bar = np.zeros(num_points, dtype=np.float64)
    f = np.zeros((num_obs, num_points), dtype=np.float64)

    if no_ca:
        for i in range(num_obs):
            a[i] = 0.0
            c[i] = 1.0
    if f_init is None:
        for i in range(num_obs):
            f[i] = alpha + a[i] + c[i] * np.interp(t, gamma_h[i], g)
    else:
        f = np.asarray(f_init, dtype=np.float64).copy()

    start_time = time.time()

    core_fn = run_mcmc_alg4_numba.py_func if use_python else run_mcmc_alg4_numba
    res = core_fn(
        y_local, t, Lh, Lg,
        h, gamma_h, h_bar, g, a, c, f, alpha,
        K,
        s2y, float(beta_h), float(beta_g),
        np.asarray(beta_h_choices if use_beta_choices_h else np.empty(0), dtype=np.float64),
        np.asarray(beta_g_choices if use_beta_choices_g else np.empty(0), dtype=np.float64),
        1 if use_beta_choices_h else 0,
        1 if use_beta_choices_g else 0,
        float(beta_h_fixed_prob), float(beta_g_fixed_prob), float(beta_fixed),
        float(c_hyper["shape"]), float(c_hyper["scale"]),
        float(a_hyper["mean"]), float(a_hyper["var"]),
        seed, thin, update_s2y,
        float(s2y_hyper["shape"]) if s2y_hyper is not None else 0.0,
        float(s2y_hyper["scale"]) if s2y_hyper is not None else 0.0,
        1 if debug_check else 0, int(debug_every), float(debug_tol), int(progress_every),
        1 if no_ca else 0,
        1 if beta_adapt else 0, int(beta_n_adapt), int(beta_window),
        float(beta_acc_target_h), float(beta_acc_target_g), float(beta_c_h), float(beta_c_g),
        float(trim_frac), burn, 1 if collect_traces else 0,
    )

    (G_final, A_final, Alpha_final, C_final, F_final, Gamma_final, SSE_final,
     a, alpha, c, g, f, h, gamma_h, h_bar,
     Beta_h_final, Beta_g_final, Acc_h_final, Acc_g_final, s2y,
     beta_h, beta_g, beta_h_counts, beta_h_accepts, beta_g_counts, beta_g_accepts, Acc_c_final,
     beta_shift_h, beta_shift_g, beta_h_adapted, beta_g_adapted) = res

    if display_plots and verbose:
        elapsed = time.time() - start_time
        print(f"[Alg4] Iter [{K}/{K}] | Elapsed: {elapsed:.1f}s")

    if SSE_final.size > 0:
        LogLikihood = -0.5 * SSE_final / (s2y**2)
        S2y_list = np.full(SSE_final.shape, s2y, dtype=np.float64)
    else:
        LogLikihood = np.empty(0, dtype=np.float64)
        S2y_list = np.empty(0, dtype=np.float64)

    if collect_traces:
        G_old = _recover_g_old_from_traces(G_final, C_final, Alpha_final)
    else:
        G_old = np.empty(0, dtype=np.float64)

    if Acc_h_final.size > 0:
        acc_h_list = Acc_h_final
        acc_h_mean = float(np.mean(Acc_h_final))
    else:
        acc_h_list = np.ones(SSE_final.shape, dtype=np.float64)
        acc_h_mean = float("nan")
    if Acc_g_final.size > 0:
        acc_g_list = Acc_g_final
        acc_g_mean = float(np.mean(Acc_g_final))
    else:
        acc_g_list = np.ones(SSE_final.shape, dtype=np.float64)
        acc_g_mean = float("nan")
    if Acc_c_final.size > 0:
        acc_c_list = Acc_c_final
        acc_c_mean = float(np.mean(Acc_c_final))
    else:
        acc_c_list = np.ones(SSE_final.shape, dtype=np.float64)
        acc_c_mean = float("nan")
    acc_dummy = np.ones(SSE_final.shape, dtype=np.float64)

    c_bar = float(np.mean(c)) if c.size > 0 else 1.0
    g_old_final = g + (alpha / c_bar) if c_bar > 0 else g

    beta_h_choices_list = np.asarray(beta_h_choices, dtype=np.float64) if beta_h_choices is not None else np.empty(0, dtype=np.float64)
    beta_g_choices_list = np.asarray(beta_g_choices, dtype=np.float64) if beta_g_choices is not None else np.empty(0, dtype=np.float64)
    beta_h_adapted_list = np.asarray(beta_h_adapted, dtype=np.float64) if beta_h_adapted is not None else np.empty(0, dtype=np.float64)
    beta_g_adapted_list = np.asarray(beta_g_adapted, dtype=np.float64) if beta_g_adapted is not None else np.empty(0, dtype=np.float64)
    beta_h_counts_list = np.asarray(beta_h_counts, dtype=np.float64) if beta_h_counts is not None else np.empty(0, dtype=np.float64)
    beta_h_accepts_list = np.asarray(beta_h_accepts, dtype=np.float64) if beta_h_accepts is not None else np.empty(0, dtype=np.float64)
    beta_g_counts_list = np.asarray(beta_g_counts, dtype=np.float64) if beta_g_counts is not None else np.empty(0, dtype=np.float64)
    beta_g_accepts_list = np.asarray(beta_g_accepts, dtype=np.float64) if beta_g_accepts is not None else np.empty(0, dtype=np.float64)

    beta_h_accept_rate = np.divide(
        beta_h_accepts_list,
        beta_h_counts_list,
        out=np.full(beta_h_accepts_list.shape, np.nan, dtype=np.float64),
        where=beta_h_counts_list != 0,
    )
    beta_g_accept_rate = np.divide(
        beta_g_accepts_list,
        beta_g_counts_list,
        out=np.full(beta_g_accepts_list.shape, np.nan, dtype=np.float64),
        where=beta_g_counts_list != 0,
    )

    res = {
        "gamma_h": gamma_h, "g": g, "g_old": g_old_final, "a": a, "c": c, "f": f,
        "Gamma": Gamma_final,
        "G": G_final, "G_old": G_old, "G_raw": G_final,
        "A": A_final, "Alpha": Alpha_final, "C": C_final, "F": F_final,
        "SSE": SSE_final, "LogLikihood": LogLikihood, "S2y": S2y_list,
        "Acc_h": acc_h_list, "Acc_rate_h": acc_h_list,
        "Acc_g": acc_g_list, "Acc_c": acc_c_list, "Acc_a": acc_dummy,
        "Acc_h_mean": acc_h_mean, "Acc_g_mean": acc_g_mean, "Acc_c_mean": acc_c_mean,
        "Beta_h": Beta_h_final, "Beta_g": Beta_g_final,
        "final_s2y": s2y,
        "Beta_h_choices": beta_h_choices_list,
        "Beta_g_choices": beta_g_choices_list,
        "Beta_h_adapted": beta_h_adapted_list,
        "Beta_g_adapted": beta_g_adapted_list,
        "Beta_shift_h": float(beta_shift_h),
        "Beta_shift_g": float(beta_shift_g),
        "Beta_adapt": bool(beta_adapt),
        "Beta_n_adapt": int(beta_n_adapt),
        "Beta_window": int(beta_window),
        "Beta_acc_target_h": float(beta_acc_target_h),
        "Beta_acc_target_g": float(beta_acc_target_g),
        "Beta_h_counts": beta_h_counts_list,
        "Beta_h_accepts": beta_h_accepts_list,
        "Beta_g_counts": beta_g_counts_list,
        "Beta_g_accepts": beta_g_accepts_list,
        "Beta_h_accept_rate": beta_h_accept_rate,
        "Beta_g_accept_rate": beta_g_accept_rate,
    }

    if verbose:
        print("[Alg4 Numba] MCMC Done.")
    return res
