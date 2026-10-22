import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.special import gammaln

# ---------------------------------------------------------------------------
#  Functions
# ---------------------------------------------------------------------------

# <sigma v> fit (Julian Table 1) (mF,T) coefficients. units are 1e-10 cm^3/s, T in nK
fit = {'a0': 1.946e-1, 'a1': 1.001e-2, 'a2': -1.542e-2,
       'b1': -7.342e-5, 'b2': 1.495e-8,
       'c1': 2.116e-7, 'c2': 5.173e-6, 'c3': -1.381e-9, 'c4': -6.899e-10}

# <sigma v> function from fit coefficients
def sigv(m, T):
    return (fit["a0"] + fit["a1"] * m + fit["a2"] * m**2
            + (fit["b1"] + fit["c1"] * m + fit["c2"] * m**2) * T
            + (fit["b2"] + fit["c3"] * m + fit["c4"] * m**2) * T**2)

# Spin-exchange rates out of state mF (Hz).
def rate(mF, T, Nb):
    # This is the product of <sigma v> and the mean density of the Rb bath, here taken as Nb T^-3/2
    return 0.159 *sigv(mF,T)*1e-10 * Nb * (T/1e9) ** (-1.5)

# Build rate matrix describing the (classical) population dynamics of the 7 mF levels
def build_rate_matrix(T, Nb):
    Gamma = np.zeros((7, 7)) # index i <-> mF = 3 - i
    for i in range(6):
        r = rate(3 - i, T, Nb)
        Gamma[i, i] = -r
        Gamma[i + 1, i] = r
    return Gamma

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

# ---------------------------------------------------------------------------
#  Grid and parameter values
# ---------------------------------------------------------------------------

pop_init = np.zeros(7); pop_init[0] = 1.0 # Initialize in the ground (mF=3) state

# Discretise parameter values
thetas = np.linspace(500, 1500, 501) # Temperature hypothesis grid (nK)
Nbs = np.linspace(2e3, 10e3, 501) # Bath atom hypothesis grid (no. atoms)
taus = np.linspace(0, 1.2, 21) # Interaction time (s)

# Precompute the populations of the 7 levels for all parameter values and interation times.
q = populations_bateman(thetas, Nbs, taus) # Has dimensions (len(thetas), len(Nbs), len(taus), 7)

mu = 30 # Mean Cs atom number
T_true, Nb_true = 975.0, 6.6e3 # True T and Nb of Rb bath

rng = np.random.default_rng(42)
q_true = populations_bateman([T_true], [Nb_true], taus)[0, 0] # (Ntau, 7)
q_true = np.clip(q_true, 0, None) # Stop floating-point numerical issues

# Prior choices
type_T,  T0,  sigma_T  = "flat", 1000.0, 150.0 # For a flat prior T0 and sigma_T are not used. They are just needed for the function
#type_Nb,  Nb0,  sigma_Nb  = "flat", 6e6, 200
type_Nb, Nb0, sigma_Nb = "gauss", 6e3 , 1000
#type_Nb, Nb0, sigma_Nb = "gauss", Nb_true + rng.normal(0, 500.0),1000.0 # Random prior mean
#type_Nb, Nb0, sigma_Nb = "lognormal", Nb_true, 500

# Because the prior factorises, build each 1D log prior and add them as a column plus a row.
log_prior = (log_prior_1d(thetas, type_T, T0, sigma_T)[:, None]
             + log_prior_1d(Nbs, type_Nb, Nb0, sigma_Nb)[None, :])

# Fisher information of the prior. This will be used for the CRB/Van-Trees
F_prior = np.diag([prior_fisher_1d(thetas, type_T, T0, sigma_T),
                   prior_fisher_1d(Nbs, type_Nb, Nb0, sigma_Nb)])

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
F_set[q_true < 1e-8] = 0.0 # For empty levels (e.g. tau = 0, mF < 3), q has rounding error

# Define the scan protocol (old)
#settings = [(5, 0), (50,6)]   # (tau index, level index)
#settings = [(tau_index, level_index) for level_index in range(7) for tau_index in range(len(taus))] # Scan over all interaction times and states

# ---------------------------------------------------------------------------
#  Main loop
# ---------------------------------------------------------------------------

M = 100 # shots per adaptive decision (M = 1 is fully adaptive)
n_shots = 10000 # total number of shots

log_post=log_prior.copy()
post = posterior(log_post, thetas, Nbs)
est_list, cov_list, F_list = [], [], []
chosen_b, chosen_i = [], [] # per shot (so they line up with est_list)
n_list = [] # Observed outcomes
K_maps = [] # per decision: full (Ntau, 7) map of K_T + K_Nb
F_cum = F_prior.copy()
t0 = time.time()

for k in range(n_shots):
    if k % M == 0:
        # Expected gain for every (tau, mF) setting under the current posterior
        K = np.zeros((len(taus), 7))
        for b in range(len(taus)):
            for i in range(7):
                K[b, i] = sum(info_gain(post, thetas, Nbs, q[:, :, b, i], mu)) # K_T + K_Nb
        b, i = np.unravel_index(np.argmax(K), K.shape) # b = tau index, i = level index (mF = 3 - i)
        K_maps.append(K)

    n = rng.poisson(mu * q_true[b, i]) # Data generation by sampling from the likelihood
    log_post += log_likelihood(n, q[:, :, b, i], mu) # Update the (log) posterior
    post = posterior(log_post, thetas, Nbs)
    est, cov = estimators(post, thetas, Nbs) # Calculate estimators and covariances from posterior
    F_cum = F_cum + F_set[b, i] # Calculate cumulative fisher info

    # Store the estimator, covariance, cumulative fisher info and the setting used
    est_list.append(est)
    cov_list.append(cov)
    F_list.append(F_cum)
    n_list.append(n)
    chosen_b.append(b)
    chosen_i.append(i)
    if (k + 1) % M == 0:
        print(f"k={k + 1:6d}  tau={taus[b]:.3f}s  mF={3 - i:+d}  K={K[b, i]:.2e}  "
              f"T={est[0]:.1f}  Nb={est[1]:.0f} ({time.time() - t0:.0f}s)")

est_list, cov_list, F_list = np.array(est_list), np.array(cov_list), np.array(F_list)

chosen_b, chosen_i = np.array(chosen_b), np.array(chosen_i)
chosen_tau, chosen_mF = taus[chosen_b], 3 - chosen_i
K_maps = np.array(K_maps) # (n_decisions, Ntau, 7)

np.savez_compressed("adaptive_run_1e6.npz",
    thetas=thetas, Nbs=Nbs, taus=taus,
    mu=mu, M=M, T_true=T_true, Nb_true=Nb_true,
    est=est_list, cov=cov_list, F=F_list, F_prior=F_prior,
    b=np.array(chosen_b), i=np.array(chosen_i), n=np.array(n_list),
    K_maps=K_maps,
    post=post, log_prior=log_prior)