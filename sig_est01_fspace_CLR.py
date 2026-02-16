# conda env create --name bayes_reg -f environment.yml

import numpy as np
from scipy.integrate import cumulative_trapezoid

from matplotlib import rcParams
rcParams.update({
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "sans-serif",  
        "font.size": 10,  
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 2.5,
        # "figure.dpi": 300,
        "figure.figsize": (3.5, 2.5),
        "savefig.bbox": "tight", 
        "savefig.format": "pdf",
        "legend.frameon": False,
        "savefig.pad_inches": 0.02,
    }
)
# import importlib
# import functions.fn_GPerror_mult  # Import the module

# importlib.reload(functions.fn_GPerror_mult)  # Reload the module
# from functions.fn_GPerror_mult import *  # Re-import updated functions

def log_acceptance_ratio(y, f_current, f_proposal, sigma_y):
    if len(y.shape) == 1:
        f_current = f_current.reshape(1, -1)
        f_proposal = f_proposal.reshape(1, -1)
        y = y.reshape(1, -1)

    N = y.shape[0]
    residual_current = np.array([y[i] - f_current[i] for i in range(N)])
    sse_current = np.sum([residual_current[i] ** 2 for i in range(N)])
    
    residual_proposal = np.array([y[i] - f_proposal[i] for i in range(N)])
    sse_proposal = np.sum([residual_proposal[i] ** 2 for i in range(N)])
    
    return -0.5 * (sse_proposal - sse_current) / (sigma_y**2)


def compute_covariance(t, v, l, cov_kernel_type):
    if cov_kernel_type == "se":
        C = v * np.exp(- (t[:,None]-t[None,:])**2 / l)
    elif cov_kernel_type == "exp":
        C = v * np.exp(- np.abs(t[:,None]-t[None,:]) / l)
    
    L = np.linalg.cholesky(C + 1e-12 * np.eye(len(t)))
    return C, L
             
def clr(gamma, t):
    dgam = np.gradient(gamma, 1/(len(t)-1))
    return np.log(dgam) - np.trapezoid(np.log(dgam),t)

def clr_inv(h, t):
    e = np.exp(h)
    gam = cumulative_trapezoid(e, t, initial=0)
    return gam/gam[-1]

def generate_random_warpings(t, a_lower_bound, a_upper_bound, num_obs):
        
	a = np.random.uniform(a_lower_bound, a_upper_bound, num_obs)
	
	gamma = np.array([(np.exp(a[i] * t) - 1) / (np.exp(a[i]) - 1) for i in range(num_obs)])
	gamma = np.array([(gamma[i] - np.min(gamma[i]))/(np.max(gamma[i])-np.min(gamma[i])) for i in range(num_obs)])

	gamma_inv = np.array([np.interp(t, gamma[i], t) for i in range(num_obs)])
	hInv = np.array([clr(gamma_inv[i], t) for i in range(num_obs)])
	hInv_bar = np.mean(hInv, axis=0)
	hInv = hInv - hInv_bar

	gamma_inv = np.array([clr_inv(hInv[i], t) for i in range(num_obs)])
	gamma = np.array([np.interp(t, gamma_inv[i], t) for i in range(num_obs)])
	h = np.array([clr(gamma[i], t) for i in range(num_obs)])
	return gamma, h

#%% Multiple Signal Data

import matplotlib.pyplot as plt
np.random.seed(42)

# Parameters
num_points = 200
t = np.linspace(0, 1, num_points)
num_obs = 10

mu = np.array([0.06, 0.94, 0.5])
sigma = np.array([0.2, 0.2, 0.15])
height = np.array([1.5, 1.5, 1.2])

# Dummy random draw to preserve RNG state for reproducibility.
# Do NOT remove: required to reproduce the exact samples from the original submission.
_ = np.random.uniform(-3, 3, num_obs)


# g is the sum of three bumps
g_true = np.sum([height[i] * np.exp(-((t - mu[i]) ** 2) / (2 * sigma[i]**2)) 
                for i in range(len(mu))], axis=0)
g_true = g_true - g_true[0]

gam_true, _ = generate_random_warpings(t, -1.5, 1.5, num_obs)

# true additive noise level
sig_y0 = 0.2

# transition effect
a = np.random.randn(num_obs)*0.1
a_true = a - np.mean(a)

# scale effect
c = np.random.gamma(shape=10, scale=0.1, size=num_obs)
c_true = c / np.mean(c)

# signal generation
f_true = np.array([np.interp(gam,t,g_true) for gam in gam_true])
f_true = a_true[:,None] + c_true[:,None] * f_true

# additive noise
epsilon = np.random.randn(num_obs, num_points)
epsilon -= np.mean(epsilon, axis=0)[None,:]

y_org = f_true + sig_y0 * epsilon


# Plot all generated functions

# Noisy Raw Signals
plt.figure(figsize=(3.5,2.5))
plt.plot(t, y_org.T, linewidth=1.5, alpha = 0.8)
plt.grid(True, linestyle=":", alpha=0.3)
plt.tight_layout()

# save axis format
ax = plt.gca()
xticks = ax.get_xticks(); yticks = ax.get_yticks()
xlim = ax.get_xlim(); ylim = ax.get_ylim()
plt.show()

# Noise-free Representation of Raw Signals
plt.figure(figsize=(3.5,2.5))
plt.plot(t, f_true.T, linewidth=2.5, alpha = 0.8)
plt.grid(True, linestyle=":", alpha=0.3)
ax = plt.gca()
ax.set_xticks(xticks); ax.set_yticks(yticks)
ax.set_xlim(xlim); ax.set_ylim(ylim)
plt.tight_layout()
plt.show()

# %% Noise-free FR

# Here, noise-free signals are inputs of FR
import fdasrsf as fs

obj_fr_noisefree = fs.fdawarp(f_true.T, t)
obj_fr_noisefree.srsf_align(verbose=0, lam=0)

fn_fr_noisefree = obj_fr_noisefree.fn.T
gam_fr_noisefree = obj_fr_noisefree.gam.T
yn_fr_noisefree = np.array(
        [np.interp(gam_fr_noisefree[i], t, y_org[i]) 
        for i in range(num_obs)])


plt.figure()
plt.plot(t, fn_fr_noisefree.T, linewidth=2.5, alpha = 0.8)
plt.grid(True, linestyle=":", alpha=0.3)
ax = plt.gca()
ax.set_xticks(xticks); ax.set_yticks(yticks)
ax.set_xlim(xlim); ax.set_ylim(ylim)
plt.title("Noise-free FR (FR on Noise-free signals)")
plt.show()


plt.figure(figsize=(2.5, 2.5))
plt.plot(t, t, color="grey", linewidth=1, alpha=0.3, linestyle=":")
plt.plot(t, gam_fr_noisefree.T, linewidth=2.5)
plt.grid(True, linestyle=":", alpha=0.3)
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Noise-free FR Warpings")
plt.show()

# %% FR on Noisy Signals

obj_y_fr = fs.fdawarp(y_org.T, t)
obj_y_fr.srsf_align(verbose=0, lam=0)

yn_fr = obj_y_fr.fn.T
gam_fr = obj_y_fr.gam.T
fn_fr = np.array([np.interp(gam_fr[i], t, f_true[i]) for i in range(num_obs)])

plt.figure()  
plt.plot(t, yn_fr.T, linewidth=1.5, alpha = 0.8)
plt.grid(True, linestyle=":", alpha=0.3)
ax = plt.gca()
ax.set_xticks(xticks); ax.set_yticks(yticks)
ax.set_xlim(xlim); ax.set_ylim(ylim)
plt.title("FR Alignment")
plt.show()

plt.figure()  
plt.plot(t, fn_fr.T, linewidth=2.5, alpha = 0.8)
plt.grid(True, linestyle=":", alpha=0.3)
ax = plt.gca()
ax.set_xticks(xticks);ax.set_yticks(yticks)
ax.set_xlim(xlim); ax.set_ylim(ylim)
plt.title("FR Noise-free Representation")
plt.show()

plt.figure(figsize=(2.5, 2.5))
plt.plot(t, t, color="grey", linewidth=1, alpha=0.3, linestyle=":")
plt.plot(t, gam_fr.T, linewidth=2.5)
plt.grid(True, linestyle=":", alpha=0.3)
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.title("FR Warpings")
plt.show()
# %% Our method

## Options
diagnostic_plot = True
print(f'diagnostic_plot = {diagnostic_plot}')

K = 50000; dispFreq = 5000


## Initialization ##

sigma_y = sig_y0/2


# initialize translation factor and set hyperparams
a = np.zeros((num_obs, 1))
a_hyper = {"mean": 0, "var": 1000}

# initialize scale factor and set hyperparams
c = np.ones((num_obs, 1))
c_hyper = {"shape": 1, "scale": 1}
u = np.log(c)
ui_prime = np.zeros(num_obs)

f = a + c * np.zeros((num_obs, num_points))
g = np.zeros(num_points)
y = y_org.copy()

# Hyperparams for template function Kg
cov_kernel_g = "se"
nu_g = 1; ell_g = 1e-2
Kg, Lg = compute_covariance(t, nu_g, ell_g, cov_kernel_g)  

beta_g_choices = np.concatenate(([0.2], np.linspace(0.001, 0.1, 9)))

# Hyperparams for Kh
cov_kernel_h = "exp"
nu_h = 1; ell_h = 5
Kh, Lh = compute_covariance(t, nu_h, ell_h, cov_kernel_h)  

beta_h_choices = np.concatenate(([0.4], np.linspace(0.01, 0.1, 9)))

### time-warping hi and gamma_i (No info)
h = np.zeros((num_obs, num_points))
h_mean = np.zeros(num_points)
gamma_h = np.tile(t, (num_obs, 1))
gamma_hInv = np.array([np.interp(t, gam, t) for gam in gamma_h])




import time

## -------------------- ##
##  MCMC f-space + CLR  ##
## -------------------- ##

np.random.seed(2)

# initialize nu_q and ell_q with very small values
if diagnostic_plot:
    # acc_rate_h = np.zeros(num_obs, dtype=bool)
    # acc_rate_c = np.zeros(num_obs, dtype=bool)
    beta_h = np.zeros(num_obs)
    
    Acc_g, Acc_h, Acc_c, Acc_rate_h, Acc_rate_c = [], [], [], [], []
    F = []; SSE = []; Beta_h = []

loglik = 0
Gamma = []
G = []
Log_likelihood = []
C, A = [], []

for k in range(K):
    rand = np.random.permutation(np.arange(num_obs))
    
    if diagnostic_plot:
        acc_accept_h = np.zeros(num_obs, dtype=bool)
        acc_accept_c = np.zeros(num_obs, dtype=bool)
        acc_accept_g = False

    for i in rand:
        

        beta_hi = np.random.choice(beta_h_choices)

        z = np.random.randn(num_points)
        xi = Lh @ z
        xi -= np.trapezoid(xi, t)
        
        h_prime = (
            h_mean + np.sqrt(1 - beta_hi**2) * (h[i] - h_mean) + beta_hi * xi
        )
        h_prime -= np.trapezoid(h_prime, t)
        gamma_h_prime = clr_inv(h_prime, t)
        
        f_prime = a[i] + c[i] * np.interp(t, gamma_h_prime, g)

        log_alpha_h = log_acceptance_ratio(y[i], f[i], f_prime, sigma_y)
        log_alpha_h = np.min([0, log_alpha_h])
        

        if np.log(np.random.uniform(0, 1)) < log_alpha_h:
            h[i] = h_prime
            gamma_h[i] = gamma_h_prime
            f[i] = f_prime
            if diagnostic_plot:
                acc_accept_h[i] = True
                
        ######################## ci ############################
        
        c_prime = c[i] * np.exp(np.random.randn() * 0.1)  # Equivalent to log-based update
        f_prime = a[i] + c_prime * np.interp(t, gamma_h[i], g)
        
        log_alpha1 = log_acceptance_ratio(y[i], f[i], f_prime, sigma_y)
                
        log_prior_proposal = (c_hyper["shape"] - 1) * np.log(c_prime) - c_hyper["scale"] * c_prime + np.log(c_prime)
        log_prior_current = (c_hyper["shape"] - 1) * np.log(c[i]) - c_hyper["scale"] * c[i] + np.log(c[i])
        
        log_alpha2 = log_prior_proposal - log_prior_current
        
        log_alpha_c = log_alpha1 + log_alpha2
        log_alpha_c = min(0, log_alpha_c)

        # if np.log(np.random.uniform(0, 1)) < log_acceptance_ratio:
        if np.log(np.random.uniform(0, 1)) < log_alpha_c:
            c[i] = c_prime
            f[i] = f_prime
            
            if diagnostic_plot:
                acc_accept_c[i] = True
            
        ########################### ei ############################        
        ai_var_post = 1 / (1/a_hyper["var"] + num_points/(sigma_y**2))
        ai_mean_post = ai_var_post * (a_hyper["mean"]/a_hyper["var"] + np.sum(y[i] - (f[i] - a[i])) / (sigma_y**2))
        
        ftemp = f[i] - a[i]
        a[i] = np.random.randn() * np.sqrt(ai_var_post) + ai_mean_post
        f[i] = ftemp + a[i]

        
    ########################### scaling ############################       
    # scale back so that mean of c to be 1
    c_mean = np.mean(c)
    a_mean = np.mean(a)
    c = c / c_mean
    a = a - a_mean
    g = g * c_mean + a_mean
    
    ########################### centering ############################       
    # Let γ̄ = 1/n Σᵢ hᵢ,
    # (𝑔 ∘ γ̂ᵢ⁻¹) = (𝑔 ∘ γ̄⁻¹) ∘ (γ̂ᵢ ∘ γ̄⁻¹)⁻¹
    mean_h = np.mean(h, axis=0) # defines 𝜓(γ̄)
    
    gamma0_bar = clr_inv(mean_h, t) # γ̄
    gamma0barInv = np.interp(t, gamma0_bar, t) # γ̄⁻¹
    
    g = np.interp(gamma0barInv, t, g) #  ← (𝑔 ∘ γ̄⁻¹)
    
    # γ̂ᵢ ← (γ̂ᵢ ∘ γ̄⁻¹)
    gamma_h = np.array([np.interp(gamma0barInv, t, gam) for gam in gamma_h])  
    gamma_hInv = np.array([np.interp(t, gam , t) for gam in gamma_h])
    h = np.array([clr(gamma_h[i], t) for i in range(num_obs)])
    
    ########################### g ###########################
    beta_g = np.random.choice(beta_g_choices)
    z = np.random.randn(num_points)
    xi = Lg @ z
    
    g_prime = np.sqrt(1 - beta_g**2) * g + beta_g * xi

    # 𝑔 ∘ γ̄⁻¹
    f = np.array([a[i] + c[i] * np.interp(t, gamma_h[i], g) for i in range(num_obs)])
    f_prime = np.array([a[i] + c[i] * np.interp(t, gamma_h[i], g_prime) for i in range(num_obs)])

    log_alpha = log_acceptance_ratio(y, f, f_prime, sigma_y)
    log_alpha = np.min([0, log_alpha])
    
    if np.log(np.random.uniform(0, 1)) < log_alpha:
        g = g_prime
        if diagnostic_plot:
            acc_accept_g = True
    
    sse_i = np.array([np.sum((y[i] - f[i]) ** 2) for i in range(num_obs)])
    sse = np.sum(sse_i)

    loglik = -0.5 * sse / (sigma_y**2)

    # Collect the results
    G.append(g.copy())
    Gamma.append(gamma_h.copy())
    Log_likelihood.append(loglik)
    C.append(c.copy())
    A.append(a.copy())
    
    if diagnostic_plot == False:
        if k == 0:
            print(f"[Iters {k+1}] loglikelihood: {loglik}")
            start_time = time.time()

        if (k + 1) % dispFreq == 0:
            end_time = time.time()
            print(
                f"[Iters {k+1}] loglikelihood: {loglik}     Time taken: {end_time-start_time:.2f} sec"
            )
            if (k + 1) < k * K // dispFreq:
                start_time = time.time()
    else:
        Acc_h.append(np.array(acc_accept_h))
        Acc_c.append(np.array(acc_accept_c))
        Acc_g.append(np.array(acc_accept_g))
        F.append(f.copy())
        SSE.append(sse)
        Beta_h.append(beta_hi)
        
        acc_h_formatted = ", ".join(f"{x:.2f}" for x in np.mean(Acc_h[-1000:], axis=0))

        if ((k+1) % dispFreq == 0) and (k > 0):
            print(f"[Iters {k+1}] \n"
                    f"SSE={SSE[k]:.2f}, sigma_y={sigma_y:.2f},  \n"
                    f"q:[nu={nu_g:.2g}, ell={ell_g:.2g}], \n"
                    f"ACC Rates: \n"
                    f"(h)=[{acc_h_formatted}], "
                    f"(g)={np.mean(Acc_g):.1g}\n")
                
            chain_f = np.array(F[-dispFreq:])
            chain_f_mean = np.mean(chain_f, axis=0)
            chain_f_lower = np.percentile(chain_f, 2.5, axis=0)
            chain_f_upper = np.percentile(chain_f, 97.5, axis=0)
            
            chain_g = np.array(G[-dispFreq:])
            chain_g_mean = np.mean(chain_g, axis=0)
            chain_g_lower = np.percentile(chain_g, 2.5, axis=0)
            chain_g_upper = np.percentile(chain_g, 97.5, axis=0)
            
            plt.figure(figsize=(15,3*2/5))
            for i in range(num_obs) if num_obs < 6 else range(6):
                plt.subplot(1,6,i+1)
                plt.plot(t,y[i],'k', label=f'y_{i+1}', linewidth=1)
                plt.fill_between(t, chain_f_lower[i], chain_f_upper[i], color='c', alpha=0.8)
                plt.plot(t, chain_f_mean[i], 'b', label=f'f_{i+1}', linewidth=2)
                plt.title(f'f_{i+1}')
            plt.show()

            plt.figure(figsize=(3*2,3*2/5))
            plt.subplot(1,2,1)
            plt.plot(t,chain_g_mean,'m', label='g')
            plt.fill_between(t, chain_g_lower, chain_g_upper, color='m', alpha = 0.8)
            plt.subplot(1,2,2)
            plt.plot(SSE)
            plt.show()            

# %% 

K = np.shape(Gamma)[0]
burnin = int(4 * K / 5); iter_end = K
thinning = int((K - burnin) / 100)

dispIdx = np.arange(burnin, iter_end, thinning)

num_burnins = len(dispIdx)

Gamma = np.array(Gamma)
Log_likelihood = np.array(Log_likelihood)
SSE = np.array(SSE)

t_dense = np.linspace(0, 1, 2 * num_points)  # 201 points
Gamma_dense = np.array([[
    np.interp(t_dense, t, Gamma[k, i, :])
    for i in range(num_obs)]
    for k in range(burnin, iter_end, thinning)
])

f_true_dense = np.array([
    np.interp(t_dense, t, f_true[i])
    for i in range(num_obs)
])
y_dense = np.array([
    np.interp(t_dense, t, y_org[i]) 
    for i in range(num_obs)
])


yn_b = np.array([
    [
         np.interp(Gamma_dense[k,i], t_dense, y_dense[i]) 
         for i in range(num_obs)
    ] 
    for k in range(num_burnins)
])

fn_b = np.array([
    [
         np.interp(Gamma_dense[k,i], t_dense, f_true_dense[i])
         for i in range(num_obs)
    ]
    for k in range(num_burnins)
])

GammaInv_burnin = np.array([
    [
         np.interp(t, Gamma[k,i], t)
         for i in range(num_obs)
    ]
    for k in range(burnin, iter_end, thinning)
])

yn_b_mean = np.mean(yn_b, axis=0)
yn_b_lower = np.percentile(yn_b, 2.5, axis=0)
yn_b_upper = np.percentile(yn_b, 97.5, axis=0)

fn_b_mean = np.mean(fn_b, axis=0)
fn_b_lower = np.percentile(fn_b, 2.5, axis=0)
fn_b_upper = np.percentile(fn_b, 97.5, axis=0)

gamInv_mean = np.mean(GammaInv_burnin, axis=0)
gamInv_lower = np.percentile(GammaInv_burnin, 2.5, axis=0)
gamInv_upper = np.percentile(GammaInv_burnin, 97.5, axis=0)

plt.figure()  
plt.plot(t_dense, yn_b_mean.T, linewidth=1.5, alpha=0.8)
plt.grid(True, linestyle=":", alpha=0.3)
ax = plt.gca()
ax.set_xticks(xticks); ax.set_yticks(yticks)
ax.set_xlim(xlim); ax.set_ylim(ylim)
plt.title("Our Method Alignment")
plt.show()

plt.figure()  
plt.plot(t_dense, fn_b_mean.T, linewidth=2.5, alpha=0.8)
plt.grid(True, linestyle=":", alpha=0.3)
ax = plt.gca()
ax.set_xticks(xticks); ax.set_yticks(yticks)
ax.set_xlim(xlim); ax.set_ylim(ylim)
plt.title("Our Method Noise-free Representation")
plt.show()

plt.figure(figsize=(2.5, 2.5))
plt.plot(t, t, color="grey", linewidth=1, alpha=0.3, linestyle=":")
plt.plot(t, gamInv_mean.T, linewidth=2.5)
plt.grid(True, linestyle=":", alpha=0.3)
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Our Method Warpings")
plt.show()

#%% Likelihood

import matplotlib.ticker as mtick

plt.plot(Log_likelihood, 'k', linewidth=2.5)
plt.xlabel("MCMC iteration")
plt.ylabel("Log-likelihood")
plt.gca().xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
plt.gca().yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(3,4))

plt.grid(True, linestyle=":", alpha=0.3)
plt.xticks([0, 10000, 20000, 30000, 40000, 50000])
plt.ticklabel_format(style='sci', axis='x', scilimits=(3,4))

plt.show()

#%% Posterior Samples for scale (c) and translation (e)

# Posterior distribution of scales
C = np.array(C); A = np.array(A)

samples_c = C[dispIdx, :, 0]
num_samples, num_obs = samples_c.shape
x = np.arange(num_obs)

c_lower = np.percentile(samples_c, 2.5, axis=0)
c_upper = np.percentile(samples_c, 97.5, axis=0)

cap_width = 0.2

plt.figure()
for i in range(num_obs):
    plt.plot([x[i], x[i]], [c_lower[i], c_upper[i]], color='black', linewidth=1)
    plt.hlines(c_lower[i], x[i] - cap_width, x[i] + cap_width, color='black', linewidth=1)
    plt.hlines(c_upper[i], x[i] - cap_width, x[i] + cap_width, color='black', linewidth=1)

plt.hlines(c_true, x - cap_width, x + cap_width, color='red', linewidth=2, label='True $c_i$')

plt.xticks(x, [fr"$c_{{{i+1}}}$" for i in range(num_obs)])
plt.xlabel("Observation index");plt.ylabel("Parameter value")
plt.grid(True, linestyle=':', alpha=0.3)
plt.tight_layout()
plt.show()

# Posterior distribution of translation
samples_a = A[dispIdx, :, 0]
a_lower = np.percentile(samples_a, 2.5, axis=0)
a_upper = np.percentile(samples_a, 97.5, axis=0)

plt.figure()
for i in range(num_obs):
    plt.plot([x[i], x[i]], [a_lower[i], a_upper[i]], color='black', linewidth=1)
    plt.hlines(a_lower[i], x[i] - cap_width, x[i] + cap_width, color='black', linewidth=1)
    plt.hlines(a_upper[i], x[i] - cap_width, x[i] + cap_width, color='black', linewidth=1)

plt.hlines(a_true, x - cap_width, x + cap_width, color='red', linewidth=2, label='True $c_i$')
plt.xticks(x, [fr"$e_{{{i+1}}}$" for i in range(num_obs)])
plt.xlabel("Observation index");plt.ylabel("Parameter value")
plt.grid(True, linestyle=':', alpha=0.3)
plt.tight_layout()
plt.show()

#%% Estimation

G = np.array(G)
g_mean = np.mean(G[dispIdx], axis=0)
g_lower = np.percentile(G[dispIdx], 2.5, axis=0)
g_upper = np.percentile(G[dispIdx], 97.5, axis=0)

plt.figure(figsize=(3.5, 2.8))
plt.fill_between(t, g_lower, g_upper, color='navy', alpha=0.3, label='95% CI', edgecolor=None)
plt.plot([],[],'navy', linewidth=1.5, alpha=1, label='$g$ (posterior mean)')
plt.plot([],[],'r--', linewidth=2, alpha=1, label='$g$ (true)')
plt.plot(t, g_true, 'r--', linewidth=3, alpha=1)
plt.plot(t, g_mean, 'navy', linewidth=1.5, alpha=1)
plt.grid(True, linestyle=":", alpha=0.3)
plt.legend(loc='upper left', fontsize=11, bbox_to_anchor=(-0.1, 1.16), ncol = 2, columnspacing=0.4)
        #    handletextpad = 0.1, columnspacing = 0.5, 
        #    handlelength = 0.6)

ax = plt.gca()
ax.set_ylim([-0.25,0.2])
ax.set_xticks([0,0.25,0.5,0.75,1])
ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
ax.set_yticks([-0.2, -0.1, 0, 0.1])
plt.tight_layout()
plt.show()

#%% Alignment Check

def compute_ls(f, f_aligned, t):
    
    N, _ = f.shape
    ls_values = []

    for i in range(N):
        # Leave-one-out mean
        idx = np.arange(N) != i
        f_mean = np.mean(f[idx], axis=0)
        f_aligned_mean = np.mean(f_aligned[idx], axis=0)

        # Differences
        diff_aligned = f_aligned[i] - f_aligned_mean
        diff_original = f[i] - f_mean

        # Numerator integral
        num = np.trapezoid(diff_aligned**2, t)
        # Denominator integral
        denom = np.trapezoid(diff_original**2, t)

        # To avoid division by zero
        if denom == 0:
            raise ValueError(f"Denominator integral is zero for i={i}")

        ratio = num / denom
        ls_values.append(ratio)

    # Average over i
    ls = np.mean(ls_values)
    return ls

def compute_pc(f, f_aligned):
    
    N, _ = f.shape
    numerator_sum = 0.0
    denominator_sum = 0.0

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            # Pearson correlation between aligned functions i and j
            cc_aligned = np.corrcoef(f_aligned[i], f_aligned[j])[0,1]
            # Pearson correlation between original functions i and j
            cc_original = np.corrcoef(f[i], f[j])[0,1]

            numerator_sum += cc_aligned
            denominator_sum += cc_original

    pc = numerator_sum / denominator_sum
    return pc

def compute_sls(f, f_aligned, t):
    
    N, _ = f.shape
    sls_values = []

    # Compute derivatives along the time axis
    f_deriv = np.gradient(f, t, axis=1)
    f_aligned_deriv = np.gradient(f_aligned, t, axis=1)

    for i in range(N):
        # Leave-one-out means
        idx = np.arange(N) != i
        f_mean_deriv = np.mean(f_deriv[idx], axis=0)
        f_aligned_mean_deriv = np.mean(f_aligned_deriv[idx], axis=0)

        # Differences
        diff_aligned = f_aligned_deriv[i] - f_aligned_mean_deriv
        diff_original = f_deriv[i] - f_mean_deriv

        # Numerator and denominator integrals
        num = np.trapezoid(diff_aligned**2, t)
        denom = np.trapezoid(diff_original**2, t)

        ratio = num / denom
        sls_values.append(ratio)

    # Average over all i
    sls = np.mean(sls_values)
    return sls


print(f"Least Squares = {compute_ls(f_true_dense, fn_b_mean, t_dense)}")
print(f"Pairwise Corr = {compute_pc(f_true_dense, fn_b_mean)}")
print(f"Sobolev Least Squares = {compute_sls(f_true_dense, fn_b_mean, t_dense)}")


