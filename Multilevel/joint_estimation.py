import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.special import gammaln

# <sigma v> fit (Julian Table 1) (mF,T) coefficients. units are 1e-10 cm^3/s, T in nK
fit = {'a0': 1.946e-1, 'a1': 1.001e-2, 'a2': -1.542e-2,
       'b1': -7.342e-5, 'b2': 1.495e-8,
       'c1': 2.116e-7, 'c2': 5.173e-6, 'c3': -1.381e-9, 'c4': -6.899e-10}

def sigv(m, T):
    # <sigma v> function
    return (fit["a0"] + fit["a1"] * m + fit["a2"] * m**2
            + (fit["b1"] + fit["c1"] * m + fit["c2"] * m**2) * T
            + (fit["b2"] + fit["c3"] * m + fit["c4"] * m**2) * T**2)

def rate(mF, T, Nb):
    # Spin-exchange rates out of state mF (Hz).
    # This is the product of <sigma v> and the mean density of the Rb bath, here taken as Nb T^-3/2
    return 0.159 *sigv(mF,T)*1e-10 * Nb * (T/1e9) ** (-1.5)

def build_rate_matrix(T, Nb):
    # Build rate matrix describing the (classical) population dynamics of the 7 mF levels
    Gamma = np.zeros((7, 7)) # index i <-> mF = 3 - i
    for i in range(6):
        r = rate(3 - i, T, Nb)
        Gamma[i, i] = -r
        Gamma[i + 1, i] = r
    return Gamma

pop_init = np.zeros(7); pop_init[0] = 1.0 # Initialize in the ground (mF=3) state

# Discretise parameter values
thetas = np.linspace(500, 1500, 501) # Temperature hypothesis grid (nK)
Nbs = np.linspace(2e3, 10e3, 501) # Bath atom hypothesis grid (no. atoms)
taus = np.linspace(0, 1.2, 101) # Interaction time (s)

# Solve the classical rate equations using the Bateman equations. This can be found on Wikipedia, but Claude wrote this bit.
def populations_bateman(thetas, Nbs, taus):
    # q[a, c, b, i]: T = thetas[a], Nb = Nbs[c], tau = taus[b], level i (mF = 3 - i)
    thetas, Nbs, taus = (np.asarray(x, float) for x in (thetas, Nbs, taus))
    mF = 3 - np.arange(6)
    lam = rate(mF[None, None, :], thetas[:, None, None], Nbs[None, :, None])  # (NT, NN, 6)
    diff = lam[..., :, None] - lam[..., None, :]
    diff[..., np.arange(6), np.arange(6)] = 1.0
    P = np.cumprod(diff, axis=-2)
    N = np.concatenate([np.ones(lam.shape[:-1] + (1,)),
                        np.cumprod(lam[..., :-1], axis=-1)], axis=-1)
    C = np.tril(N[..., :, None] / P)
    E = np.exp(-lam[..., None, :] * taus[None, None, :, None])   # (NT, NN, Ntau, 6)
    q = np.empty(lam.shape[:-1] + (len(taus), 7))
    q[..., :6] = np.einsum('acik,acbk->acbi', C, E)
    q[..., 6] = 1.0 - q[..., :6].sum(-1)
    return q

# Precompute the populations of the 7 levels for all parameter values and interation times.
q = populations_bateman(thetas, Nbs, taus) # Has dimensions (len(thetas), len(Nbs), len(taus), 7)

mu = 30 # Mean Cs atom number
T_true, Nb_true = 975.0, 6.6e3 # True T and Nb of Rb bath

rng = np.random.default_rng(42)
q_true = populations_bateman([T_true], [Nb_true], taus)[0, 0] # (Ntau, 7)
q_true = np.clip(q_true, 0, None) # Stop floating-point numerical issues

# Poisson log-likelihood
def log_likelihood(n, q, mu):
    # log p(n | theta) for every theta. q is the vector q_{mF}(theta, tau).
    nu = np.maximum(mu * q, 1e-300) # avoid log(0) when a level is empty
    return n * np.log(nu) - nu - gammaln(n + 1)

# Define a 2d integration function over the T,Nb parameter grids. Uses the numpy trapezium rule
def trapz(f, thetas, Nbs):
    return np.trapezoid(np.trapezoid(f, Nbs, axis=-1), thetas, axis=-1)

# Prior choices
def log_prior_1d(x, type, x0, sigma):
    # Log prior density in x (up to a constant which is normalised later).
    # x0 = centre, sigma = width in the units of x.
    if type == "flat":
        return np.zeros_like(x)
    elif type == "gauss":
        return -0.5 * ((x - x0) / sigma)**2
    elif type == "lognormal":
        s = sigma / x0 # relative width
        return -0.5 * (np.log(x / x0) / s)**2 - np.log(x) # the -log(x) is the 1/x Jacobian

type_T,  T0,  sigma_T  = "flat", 1000.0, 150.0 # For a flat prior T0 and sigma_T are not used. They are just needed for the function
#type_Nb,  Nb0,  sigma_Nb  = "flat", 6e6, 200
type_Nb, Nb0, sigma_Nb = "gauss", 6e3 , 1000
#type_Nb, Nb0, sigma_Nb = "gauss", Nb_true + rng.normal(0, 500.0),1000.0 # Random prior mean
#type_Nb, Nb0, sigma_Nb = "lognormal", Nb_true, 500

# Because the prior factorises, build each 1D log prior and add them as a column plus a row.
log_prior = (log_prior_1d(thetas, type_T, T0, sigma_T)[:, None]
             + log_prior_1d(Nbs, type_Nb, Nb0, sigma_Nb)[None, :])

# Calculates the posterior
def posterior(log_post, thetas, Nbs):
    p = np.exp(log_post - log_post.max()) # Subtract max before exp. This makes computing the evidence easier
    return p / trapz(p, thetas, Nbs)

def estimators(post, thetas, Nbs):
    lT, lN = np.log(thetas)[:, None], np.log(Nbs)
    m = np.array([trapz(post*lT, thetas, Nbs), trapz(post*lN, thetas, Nbs)])
    dT, dN = lT - m[0], lN - m[1]
    cov = np.array([[trapz(post*dT*dT, thetas, Nbs), trapz(post*dT*dN, thetas, Nbs)],
                    [trapz(post*dT*dN, thetas, Nbs), trapz(post*dN*dN, thetas, Nbs)]])
    return np.exp(m), cov # Returns (T_est, Nb_est), Cov[ln T, ln Nb]

# Per-setting Fisher matrix in (ln T, ln Nb) at the true point: F[b, i] is 2x2
h = 1e-4
q_p = populations_bateman([T_true*np.exp(h)], [Nb_true], taus)[0, 0]
q_m = populations_bateman([T_true*np.exp(-h)], [Nb_true], taus)[0, 0]
dq_T = (q_p - q_m) / (2*h)  # dq/dlnT
q_p = populations_bateman([T_true], [Nb_true*np.exp(h)], taus)[0, 0]
q_m = populations_bateman([T_true], [Nb_true*np.exp(-h)], taus)[0, 0]
dq_N = (q_p - q_m) / (2*h) # dq/dlnNb
g = np.stack([dq_T, dq_N], axis=-1) # (Ntau, 7, 2)
F_set = mu * g[..., :, None] * g[..., None, :] / np.maximum(q_true, 1e-12)[..., None, None] # Poisson Fisher
F_set[q_true < 1e-8] = 0.0 # Reduce round-off error

def prior_fisher_1d(x, type, x0, sigma):
    # Fisher information of the prior with respect to ln x (for the log-error CRB)
    if type == "flat":
        return 0.0                                  # see caveat below
    lx = np.log(x)
    log_p = log_prior_1d(x, type, x0, sigma) + lx   # density in ln x
    p = np.exp(log_p - log_p.max()); p /= np.trapezoid(p, lx)
    score = np.gradient(log_p, lx)
    return np.trapezoid(p * score**2, lx)

F_prior = np.diag([prior_fisher_1d(thetas, type_T, T0, sigma_T),
                   prior_fisher_1d(Nbs, type_Nb, Nb0, sigma_Nb)])

# Define the scan protocol
#settings = [(5, 0), (50,6)]   # (tau index, level index)
settings = [(tau_index, level_index) for level_index in range(7) for tau_index in range(len(taus))] # Scan over all interaction times and states

# Main loop
log_post=log_prior.copy()
est_list, cov_list, F_list = [], [], []
F_cum = F_prior.copy()
for k in range(1000):
    b, i = settings[k % len(settings)] # Pick the mF level and interaction time
    n = rng.poisson(mu * q_true[b, i]) # Data generation by sampling from the likelihood
    log_post += log_likelihood(n, q[:, :, b, i], mu) # Update the (log) posterior
    post = posterior(log_post, thetas, Nbs)
    est, cov = estimators(post, thetas, Nbs) # Calculate estimators and covariances from posterior
    F_cum = F_cum + F_set[b, i] # Calculate cumulative fisher info

    # Store the estimator, covariance, and cumulative fisher info
    est_list.append(est)
    cov_list.append(cov)
    F_list.append(F_cum)

est_list, cov_list = np.array(est_list), np.array(cov_list)
k = np.arange(1, len(est_list) + 1)
deltaT  = est_list[:, 0] * np.sqrt(cov_list[:, 0, 0]) # Absolute T error (nK)
deltaNb = est_list[:, 1] * np.sqrt(cov_list[:, 1, 1]) # Absolute Nb error (atoms)

### Plot T estimate (with error bars)
plt.plot(k, est_list[:, 0], color="C3", label="T Bayes estimate")
plt.fill_between(k, est_list[:, 0] - deltaT, est_list[:, 0] + deltaT, color="C3", alpha=0.3)
plt.axhline(T_true, color="grey", ls="--", label="T true")
plt.xlabel("k"); plt.ylabel(r"$\tilde\vartheta_k$ (nK)"); plt.legend()
plt.show()

### Plot Nb estimate (with error bars)
plt.plot(k, est_list[:, 1], color="C0", label=r"$N_b$ Bayes estimate")
plt.fill_between(k, est_list[:, 1] - deltaNb, est_list[:, 1] + deltaNb, color="C0", alpha=0.3)
plt.axhline(Nb_true, color="grey", ls="--", label=r"$N_b$ true")
plt.xlabel("k"); plt.ylabel(r"$\tilde N_{b,k}$"); plt.legend()
plt.show()

### CRB plot
F_list = np.array(F_list)
crb = np.full_like(F_list, np.nan)
ok = np.linalg.det(F_list) > 0 # F is singular until two settings have been used
crb[ok] = np.linalg.inv(F_list[ok])

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
for ax, y, c, t in zip(axes,
        (cov_list[:, 0, 0], cov_list[:, 1, 1], cov_list[:, 0, 0] + cov_list[:, 1, 1]),
        (crb[:, 0, 0], crb[:, 1, 1], crb[:, 0, 0] + crb[:, 1, 1]),
        (r"$\epsilon_T$", r"$\epsilon_{N_b}$", r"$\epsilon_T + \epsilon_{N_b}$")):
    ax.loglog(k, y, color="C3", label="Bayes")
    ax.loglog(k, c, color="k", ls="--", label=r"CRB $(\sum F)^{-1}$")
    ax.set_title(t); ax.set_xlabel("k")
axes[0].legend(); plt.tight_layout()
plt.show()

# Variance of the combination the data actually measure: c = u . (ln T, ln Nb)
w, V = np.linalg.eigh(F_list[-1] - F_prior) # data-only Fisher
u = V[:, -1] # its best-measured direction (u[0]/u[1] = beta)
var_c = np.array([u @ C @ u for C in cov_list]) # Bayes posterior variance of c
crb_c = np.array([u @ C @ u for C in crb]) # CRB for c
#var_c = np.einsum('i,kij,j->k', u, cov_list, u)
#crb_c = np.einsum('i,kij,j->k', u, crb, u)

plt.loglog(k, var_c, color="C3", label="Bayes")
plt.loglog(k, crb_c, color="k", ls="--", label="CRB")
plt.loglog(k, crb_c[-1] * k[-1] / k, color="grey", lw=0.8, label=r"$\propto 1/k$")
plt.xlabel("k"); plt.ylabel(r"Var$[\ln N_b + \beta \ln T]$")
plt.legend()
plt.show()

### Plot final posterior as a colour map
plt.pcolormesh(Nbs, thetas, post, shading="auto")
plt.xlabel(r"$N_b$"); plt.ylabel("T (nK)"); plt.colorbar(label="posterior")
plt.show()

### Information gain. Commented out because this takes some time
"""
def info_gain(prior, thetas, Nbs, q_ab, mu, n_max=100):
    # Returns (K_T, K_Nb); K_sum = K_T + K_Nb (W = identity on log-parameters).
    lT, lN = np.log(thetas)[:, None], np.log(Nbs) # note [:, None] turns ln T into a column, so its value runs down the T axis and is copied across every N_b column
    m0 = np.array([trapz(prior*lT, thetas, Nbs), trapz(prior*lN, thetas, Nbs)])
    n = np.arange(n_max + 1)[:, None, None]
    lik = np.exp(log_likelihood(n, q_ab[None], mu)) * prior[None]     # (n, NT, NN)
    p_n = trapz(lik, thetas, Nbs)
    ok = p_n > 1e-300
    dT = trapz(lik*lT, thetas, Nbs)[ok] / p_n[ok] - m0[0]
    dN = trapz(lik*lN, thetas, Nbs)[ok] / p_n[ok] - m0[1]
    return np.sum(p_n[ok] * dT**2), np.sum(p_n[ok] * dN**2)

prior = posterior(log_prior, thetas, Nbs)
# Plot prior info gains
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
for i in range(7):
    K = np.array([info_gain(prior, thetas, Nbs, q[:, :, b, i], mu) for b in range(len(taus))])
    for ax, y in zip(axes, (K[:, 0], K[:, 1], K.sum(1))):
        ax.plot(taus * 1e3, y, label=f"$m_F = {3 - i:+d}$")
for ax, t in zip(axes, (r"$K_T$", r"$K_{N_b}$", r"$K_T + K_{N_b}$")):
    ax.set_title(t); ax.set_xlabel(r"$\tau$ (ms)")
axes[0].legend(ncol=2, fontsize=8); plt.tight_layout(); plt.show()

# Plot final posterior info gains
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
for i in range(7):
    K = np.array([info_gain(post, thetas, Nbs, q[:, :, b, i], mu) for b in range(len(taus))])
    for ax, y in zip(axes, (K[:, 0], K[:, 1], K.sum(1))):
        ax.plot(taus * 1e3, y, label=f"$m_F = {3 - i:+d}$")
for ax, t in zip(axes, (r"$K_T$", r"$K_{N_b}$", r"$K_T + K_{N_b}$")):
    ax.set_title(t); ax.set_xlabel(r"$\tau$ (ms)")
axes[0].legend(ncol=2, fontsize=8); plt.tight_layout(); plt.show()
"""