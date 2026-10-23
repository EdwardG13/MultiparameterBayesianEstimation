import sys
import csv
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.special import gammaln

# ---------------------------------------------------------------------------
#  Functions
# ---------------------------------------------------------------------------

required = ["a0", "a1", "a2", "b1", "b2", "c1", "c2", "c3", "c4"] # every file must have these
optional = ["a3", "c5", "c6"] # cubic terms, default 0

# Read fit coefficients for the rates for a given B field. Return a dict with each coefficient
def load_fit(B_field, folder="data/fits", suffix=""):
    loaded = {}
    with open(f"{folder}/sigmav_fit_coeffs_B_{B_field}mG{suffix}.csv", "r") as file:
        for row in csv.reader(file, delimiter=","):
            try:
                loaded[row[0].strip()] = float(row[2])
            except (IndexError, ValueError):
                continue # 'sep=,' line, header, or blank line
    missing = set(required) - set(loaded)
    unknown = set(loaded) - set(required + optional)
    assert not missing, f"missing coefficients: {missing}"
    assert not unknown, f"unknown coefficients: {unknown}"
    fit = dict.fromkeys(optional, 0.0) # only the cubic terms default to 0
    fit.update(loaded)
    return fit

# <sigma v> function from fit coefficients
def sigv(fit, mF, T):
    return ((fit["a0"] + fit["a1"]*mF + fit["a2"]*mF**2 + fit["a3"]*mF**3)
          + (fit["b1"] + fit["c1"]*mF + fit["c2"]*mF**2 + fit["c5"]*mF**3) * T
          + (fit["b2"] + fit["c3"]*mF + fit["c4"]*mF**2 + fit["c6"]*mF**3) * T**2)

# Spin-exchange rates out of state mF (Hz).
def rate(fit, mF, T, Nb):
    # This is the product of <sigma v> and the mean density of the Rb bath, here taken as Nb T^-3/2
    return 0.159 * sigv(fit,mF,T)*1e-10 * Nb * (T/1e9) ** (-1.5)

# Build rate matrix describing the (classical) population dynamics of the 7 mF levels
def build_Gamma(fit_down, fit_up, thetas, Nbs):
    # Returns rate matrices at every grid point, shape (NT, NN, 7, 7)
    T, N = np.asarray(thetas, float)[:, None], np.asarray(Nbs, float)[None, :]
    G = np.zeros((T.shape[0], N.shape[1], 7, 7))
    for i in range(7):
        mF = 3 - i # index i <-> mF = 3 - i
        if i < 6: # down: mF -> mF-1
            r = rate(fit_down, mF, T, N)
            assert np.all(r >= 0), f"negative rate for mF={mF:+d}: min {r.min():.2e} (check the fit / T range)"
            G[..., i + 1, i] += r
            G[..., i, i] -= r
        if fit_up is not None and i > 0: # up (endothermic): mF -> mF+1
            r = rate(fit_up, mF, T, N)
            assert np.all(r >= 0), f"negative rate for mF={mF:+d}: min {r.min():.2e} (check the fit / T range)"
            G[..., i - 1, i] += r
            G[..., i, i] -= r
    return G

"""
# Solve the classical rate equations using the Bateman equations. This can be found on Wikipedia, but Claude wrote this bit. Assumes the state is initialised in mF=3.
def populations_bateman(fit, thetas, Nbs, taus):
    # q[a, c, b, i]: T = thetas[a], Nb = Nbs[c], tau = taus[b], level i (mF = 3 - i)
    thetas, Nbs, taus = (np.asarray(x, float) for x in (thetas, Nbs, taus))
    mF = 3 - np.arange(6)
    lam = rate(fit,mF[None, None, :], thetas[:, None, None], Nbs[None, :, None])  # (NT, NN, 6)
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
"""

def populations(fit_down, fit_up, thetas, Nbs, taus):
    # q[a, c, b, i]: T = thetas[a], Nb = Nbs[c], tau = taus[b], level i (mF = 3 - i). Start in mF = +3.
    # This precomputes the population probabilities q for every point on the parameter space and for all times tau.
    # It works by stepping through the tau array, using the formula q(t+dt)=exp(Gamma(t+dt))q_0= P q(t), where P=exp(Gamma dt) and q(t)=exp(Gamma t)q_0.
    # Importantly, the tau grid must be equally spaced.
    taus = np.asarray(taus, float)
    dtau = taus[1] - taus[0]
    G = build_Gamma(fit_down, fit_up, thetas, Nbs) # (NT, NN, 7, 7)
    P = expm(G * dtau) # propagator over one tau step, one per grid point
    q = np.zeros(G.shape[:2] + (len(taus), 7))
    q[..., 0, 0] = 1.0 # tau = 0: all population in mF = +3
    for b in range(1, len(taus)):
        # q[..., b, :] = np.einsum('acij,acj->aci', P, q[..., b - 1, :]) # q(tau_b) = P q(tau_{b-1})
        for a in range(len(thetas)):
            for c in range(len(Nbs)):
                q[a, c, b, :] = P[a, c] @ q[a, c, b - 1, :]  # (7x7) @ (7,) -> (7,)
    assert np.all(q > -1e-10) and np.all(q < 1 + 1e-10), "populations outside [0, 1]"
    return q

# Define a 2d integration function over the T,Nb parameter grids. Uses the numpy trapezium rule
def trapz(f, thetas, Nbs):
    return np.trapezoid(np.trapezoid(f, Nbs, axis=-1), thetas, axis=-1)

# Poisson log-likelihood
def log_likelihood(n, q, mu):
    # log p(n | theta) for every theta. q is the vector q_{mF}(theta, tau).
    nu = np.maximum(mu * q, 1e-300) # avoid log(0) when a level is empty
    return n * np.log(nu) - nu - gammaln(n + 1)

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

# Calculates the posterior
def posterior(log_post, thetas, Nbs):
    p = np.exp(log_post - log_post.max()) # Subtract max before exp. This makes computing the evidence easier.
    return p / trapz(p, thetas, Nbs)

# Calculates the estimators and covariance matrix for a given posterior
def estimators(post, thetas, Nbs):
    lT, lN = np.log(thetas)[:, None], np.log(Nbs)
    m = np.array([trapz(post*lT, thetas, Nbs), trapz(post*lN, thetas, Nbs)])
    dT, dN = lT - m[0], lN - m[1]
    cov = np.array([[trapz(post*dT*dT, thetas, Nbs), trapz(post*dT*dN, thetas, Nbs)],
                    [trapz(post*dT*dN, thetas, Nbs), trapz(post*dN*dN, thetas, Nbs)]])
    return np.exp(m), cov # Returns (T_est, Nb_est), Cov[ln T, ln Nb]

# Bayesian information gain
# For each parameter: K = sum_n p(n) [ln(posterior estimate | n) - ln(prior estimate)]^2.
def info_gain(prior, thetas, Nbs, q_TN, mu):
    # Returns (K_T, K_Nb). Total info gain is the sum K_T + K_Nb.
    # q_TN needs to be (len(T),len(Nb)), i.e. for a fixed (tau, mF).

    lT, lN = np.log(thetas)[:, None], np.log(Nbs) # note [:, None] turns ln T into a column, so its value runs down the T axis and is copied across every N_b column
    # Log prior estimates
    log_T_prior = trapz(prior * lT, thetas, Nbs)
    log_N_prior = trapz(prior * lN, thetas, Nbs)

    # In principle, the sum should go over all possible measurement counts (up to infinity).
    # Since p(n)=int dtheta p(n|theta) p(theta) and p(n|theta) is Poisson with mean nu=mu*q, cut the sum off to nu_max + 6sqrt(nu_max)+5 (for safety).
    nu_max = mu * q_TN.max()
    if nu_max < 1e-8: # e.g. tau = 0, mF < 3, where there is no info gain
        return 0.0, 0.0
    n_max = int(np.ceil(nu_max + 6 * np.sqrt(nu_max) + 5))

    # Bayes' theorem for every possible n at once
    n = np.arange(n_max + 1)[:, None, None] # size (n_max+1, 1, 1)
    joint = np.exp(log_likelihood(n, q_TN[None], mu)) * prior[None] # p(n|theta) p(theta). This has shape (n, NT, NN). Note [None] inserts an axis of length 1
    p_n = trapz(joint, thetas, Nbs) # evidence p(n), shape (n,)
    ok = p_n > 1e-100 # drop counts whose evidence underflows (avoids 0/0)
    post_n = joint[ok] / p_n[ok, None, None] # normalised posterior for each n

    # Log posterior estimates, one per n
    log_T_post = trapz(post_n * lT, thetas, Nbs)
    log_N_post = trapz(post_n * lN, thetas, Nbs)

    K_T = np.sum(p_n[ok] * (log_T_post - log_T_prior) ** 2)
    K_N = np.sum(p_n[ok] * (log_N_post - log_N_prior) ** 2)
    return K_T, K_N

# Fisher information of the prior with respect to ln x (for the log-error CRB)
def prior_fisher_1d(x, type, x0, sigma):
    if type == "flat":
        return 0.0
    lx = np.log(x)
    log_p = log_prior_1d(x, type, x0, sigma) + lx   # density in ln x
    p = np.exp(log_p - log_p.max()); p /= np.trapezoid(p, lx)
    score = np.gradient(log_p, lx)
    return np.trapezoid(p * score**2, lx)

# Exact log posterior on any grid: original prior + all data, grouped by setting
def rebuild_log_post(thetas, Nbs, q, N_tot, m_tot):
    lp = make_log_prior(thetas, Nbs)
    for b, i in zip(*np.nonzero(m_tot)):
        nu = np.maximum(mu * q[:, :, b, i], 1e-300)
        lp += N_tot[b, i] * np.log(nu) - m_tot[b, i] * nu
    return lp

 # New grid (same number of points), log-spaced, covering mean ± n_sig sigma, clipped to the prior range
def zoom(post, thetas, Nbs, n_sig=8):
    NG = len(thetas)
    est, cov = estimators(post, thetas, Nbs)
    sT, sN = np.sqrt(np.diag(cov))
    lT0, lN0 = np.log(est)
    lT = np.linspace(max(lT0 - n_sig*sT, np.log(500)), min(lT0 + n_sig*sT, np.log(1500)), NG)
    lN = np.linspace(max(lN0 - n_sig*sN, np.log(2e3)), min(lN0 + n_sig*sN, np.log(10e3)), NG)
    return np.exp(lT), np.exp(lN)

# ---------------------------------------------------------------------------
#  Grid and parameter values
# ---------------------------------------------------------------------------

# Discretise parameter values
thetas = np.linspace(500, 1500, 501) # Temperature hypothesis grid (nK)
Nbs = np.linspace(2e3, 10e3, 501) # Bath atom hypothesis grid (no. atoms)
taus = np.linspace(0, 1.2, 21) # Interaction time (s)

B_field = 50.0 # mG (can only choose from known fits)
fit = load_fit(B_field) # Read fit coefficients for the rates for a given B field
#fit_up = None # Replace None by fit_up for endothermic rates (only for B=50mG)
fit_up = load_fit(B_field,suffix="_endo")

# Precompute the populations of the 7 levels for all parameter values and interation times.
q = populations(fit, fit_up, thetas, Nbs, taus) # Has dimensions (len(thetas), len(Nbs), len(taus), 7).

mu = 30 # Mean Cs atom number
T_true, Nb_true = 975.0, 6.6e3 # True T and Nb of Rb bath

seed=42
rng = np.random.default_rng(seed)
q_true = populations(fit,fit_up, [T_true], [Nb_true], taus)[0, 0] # (Ntau, 7)
q_true = np.clip(q_true, 0, None) # Stop floating-point numerical issues

# Prior choices
type_T,  T0,  sigma_T  = "flat", 1000.0, 150.0 # For a flat prior T0 and sigma_T are not used. They are just needed for the function
#type_Nb,  Nb0,  sigma_Nb  = "flat", 6e6, 200
type_Nb, Nb0, sigma_Nb = "gauss", 6e3 , 1000
#type_Nb, Nb0, sigma_Nb = "gauss", Nb_true + rng.normal(0, 500.0),1000.0 # Random prior mean
#type_Nb, Nb0, sigma_Nb = "lognormal", Nb_true, 500

# Because the prior factorises, build each 1D log prior and add them as a column plus a row.
def make_log_prior(thetas, Nbs):
    return (log_prior_1d(thetas, type_T, T0, sigma_T)[:, None]
            + log_prior_1d(Nbs, type_Nb, Nb0, sigma_Nb)[None, :])

log_prior = make_log_prior(thetas, Nbs)

# Fisher information of the prior. This will be used for the CRB/Van-Trees
F_prior = np.diag([prior_fisher_1d(thetas, type_T, T0, sigma_T),
                   prior_fisher_1d(Nbs, type_Nb, Nb0, sigma_Nb)])

# Per-setting Fisher matrix in (ln T, ln Nb) at the true point: F[b, i] is 2x2
h = 1e-4
q_p = populations(fit,fit_up,[T_true*np.exp(h)], [Nb_true], taus)[0, 0]
q_m = populations(fit,fit_up,[T_true*np.exp(-h)], [Nb_true], taus)[0, 0]
dq_T = (q_p - q_m) / (2*h)  # dq/dlnT
q_p = populations(fit,fit_up,[T_true], [Nb_true*np.exp(h)], taus)[0, 0]
q_m = populations(fit,fit_up,[T_true], [Nb_true*np.exp(-h)], taus)[0, 0]
dq_N = (q_p - q_m) / (2*h) # dq/dlnNb
g = np.stack([dq_T, dq_N], axis=-1) # (Ntau, 7, 2)
F_set = mu * g[..., :, None] * g[..., None, :] / np.maximum(q_true, 1e-12)[..., None, None] # Poisson Fisher
F_set[q_true < 1e-8] = 0.0 # For empty levels (e.g. tau = 0, mF < 3), q has rounding error

# Define the scan protocol (old)
#settings = [(5, 0), (50,6)]   # (tau index, level index)
#settings = [(tau_index, level_index) for level_index in range(7) for tau_index in range(len(taus))] # Scan over all interaction times and states

# Calculates in advance which blocks will be chosen
def block_schedule(n_shots, M_min, M_frac, M_max):
    Ms, k = [], 0
    while k < n_shots:
        M = min(max(M_min, int(M_frac * k)), M_max, n_shots - k)
        Ms.append(M); k += M
    return Ms

# ---------------------------------------------------------------------------
#  Main loop
# ---------------------------------------------------------------------------

n_shots = 100000 # total number of shots
M_min, M_max, M_frac = 100, 10000, 0.05 # shots per adaptive decision = min(M_max,max(M_min, M_frac of shots so far)).
n_sig = 8 # number of standard deviations the new grid will update to (= posterior mean ± n_sig sigma (in log units))

log_post = log_prior.copy()
post = posterior(log_post, thetas, Nbs)
N_tot = np.zeros((len(taus), 7)) # total counts per setting
m_tot = np.zeros((len(taus), 7)) # total shots per setting
k_list, est_list, cov_list, F_list = [], [], [], []
chosen_b, chosen_i, M_list, K_maps = [], [], [], []
F_cum = F_prior.copy()
k = 0

Ms = block_schedule(n_shots, M_min, M_frac, M_max) # Calculates in advance which blocks will be chosen
#print(f"{len(Ms)} decisions in total")

for j, M in enumerate(tqdm.tqdm(Ms, file=sys.stdout, dynamic_ncols=True)):
    # Choose the number of shots per adaptive step (M=1 is fully adaptive). See comment above. The n_shots - k is to make sure the last shot is such that there is n_shots total.
    # M = min(max(M_min, int(M_frac * k)), M_max, n_shots - k)

    # Zoom the grid onto the posterior and rebuild it from all data so far
    if k >= 1000:
        thetas, Nbs = zoom(post, thetas, Nbs, n_sig)
        q = populations(fit,fit_up,thetas, Nbs, taus)
        log_post = rebuild_log_post(thetas, Nbs, q, N_tot, m_tot)
        post = posterior(log_post, thetas, Nbs)

    # Decision: expected single-shot gain for every (tau, mF)
    K = np.zeros((len(taus), 7))
    for b in range(len(taus)):
        for i in range(7):
            #K[b, i] = sum(info_gain(post, thetas, Nbs, q[:, :, b, i], mu))
            s = slice(None, None, 2) # For a ~4x speedup, compute K on every second point on the parameter grid
            post_c = post[s, s] / trapz(post[s, s], thetas[s], Nbs[s])
            K[b, i] = sum(info_gain(post_c, thetas[s], Nbs[s], q[s, s, b, i], mu))
    b, i = np.unravel_index(np.argmax(K), K.shape)

    # Run the block: total counts of M shots, one exact posterior update
    N = rng.poisson(M * mu * q_true[b, i])
    log_post += log_likelihood(N, q[:, :, b, i], M * mu)
    N_tot[b, i] += N
    m_tot[b, i] += M
    k += M

    post = posterior(log_post, thetas, Nbs)
    est, cov = estimators(post, thetas, Nbs)
    F_cum = F_cum + M * F_set[b, i]

    k_list.append(k); est_list.append(est); cov_list.append(cov); F_list.append(F_cum)
    chosen_b.append(b); chosen_i.append(i); M_list.append(M); K_maps.append(K)

    tqdm.tqdm.write(f"[{j + 1}/{len(Ms)}] k={k:8d}  M={M:6d}  tau={taus[b]:.3f}s  mF={3 - i:+d} T={est[0]:.1f} ± {est[0] * np.sqrt(cov[0, 0]):.1f}  Nb={est[1]:.0f} ± {est[1] * np.sqrt(cov[1, 1]):.1f} ", file=sys.stdout)

# For the file name, create a string for the number of shots in simplified standard form (e.g. 1e3 or 1e6)
n_shots_str = f"{n_shots:.0e}"
mantissa, exponent = n_shots_str.split("e")
n_shots_str = f"{mantissa}e{int(exponent)}"

np.savez(f"data/adaptive_run_B_{B_field}mG_{n_shots_str}_seed{seed}.npz",
    thetas=thetas, Nbs=Nbs, taus=taus, mu=mu, T_true=T_true, Nb_true=Nb_true,
    k=np.array(k_list), M=np.array(M_list),
    est=np.array(est_list), cov=np.array(cov_list), F=np.array(F_list), F_prior=F_prior,
    b=np.array(chosen_b), i=np.array(chosen_i), N_tot=N_tot, m_tot=m_tot,
    K_maps=np.array(K_maps), post=post)