import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.special import gammaln

# <sigma v> fit (Julian Table 1) in 1e-10 cm^3/s, T in nK
fit = {'a0': 1.946e-1, 'a1': 1.001e-2, 'a2': -1.542e-2,
       'b1': -7.342e-5, 'b2': 1.495e-8,
       'c1': 2.116e-7, 'c2': 5.173e-6, 'c3': -1.381e-9, 'c4': -6.899e-10}

def sigv(m, T):
    return (fit["a0"] + fit["a1"] * m + fit["a2"] * m**2
            + (fit["b1"] + fit["c1"] * m + fit["c2"] * m**2) * T
            + (fit["b2"] + fit["c3"] * m + fit["c4"] * m**2) * T**2)

def rate(mF, T, Nb):
    # SE rate out of state mF (Hz). T in nK.
    # return 42 * Nb * T**(-1.5)
    return 0.159 *sigv(mF,T)*1e-10 * Nb * (T/1e9) ** (-1.5)

def build_rate_matrix(T, Nb):
    Gamma = np.zeros((7, 7)) # index i <-> mF = 3 - i
    for i in range(6):
        r = rate(3 - i, T, Nb)
        Gamma[i, i] = -r
        Gamma[i + 1, i] = r
    return Gamma

pop_init = np.zeros(7); pop_init[0] = 1.0 # Initialize in the ground (mF=-3) state

thetas = np.linspace(500, 1500, 101)   # temperature hypothesis grid (nK)
Nbs    = np.linspace(5e3, 8e3, 61) # bath atom hypothesis grid (no. atoms)
taus   = np.linspace(0, 1.2, 121) # Interaction time (s)

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

q = populations_bateman(thetas, Nbs, taus)   # (101, 61, 121, 7): ~0.2 s, 42 MB

mu = 30 # Rb bath atom number and mean Cs atom number

# Poisson log-likelihood
def log_likelihood(n, q, mu):
    # log p(n | theta) for every theta. q is the vector q_m(theta, tau).
    nu = np.maximum(mu * q, 1e-300) # avoid log(0) when a level is empty
    return n * np.log(nu) - nu - gammaln(n + 1)

def trapz(f, thetas, Nbs):
    return np.trapezoid(np.trapezoid(f, Nbs, axis=-1), thetas, axis=-1)

# Posterior on the theta grid
def posterior(log_post, thetas, Nbs):
    p = np.exp(log_post - log_post.max()) # subtract max before exp
    return p / trapz(p, thetas, Nbs)

log_post = np.zeros((len(thetas), len(Nbs)))  # flat in T
#log_post += -0.5 * ((Nbs[None, :] - 6.6e3) / 0.3e3)**2  # Gaussian Nb prior from imaging

def estimators(post, thetas, Nbs):
    lT, lN = np.log(thetas)[:, None], np.log(Nbs)[None, :]
    m = np.array([trapz(post*lT, thetas, Nbs), trapz(post*lN, thetas, Nbs)])
    dT, dN = lT - m[0], lN - m[1]
    cov = np.array([[trapz(post*dT*dT, thetas, Nbs), trapz(post*dT*dN, thetas, Nbs)],
                    [trapz(post*dT*dN, thetas, Nbs), trapz(post*dN*dN, thetas, Nbs)]])
    return np.exp(m), cov        # (T_est, Nb_est), Cov[ln T, ln Nb]

rng = np.random.default_rng(3)
T_true, Nb_true = 975.0, 6.6e3
q_true = populations_bateman([T_true], [Nb_true], taus)[0, 0]     # (Ntau, 7)
settings = [(19, 0), (40, 6), (50, 3)]   # (tau index, level index)

est_list, cov_list = [], []
for k in range(300):
    b, i = settings[k % len(settings)]
    n = rng.poisson(mu * q_true[b, i])
    log_post += log_likelihood(n, q[:, :, b, i], mu)
    post = posterior(log_post, thetas, Nbs)
    est, cov = estimators(post, thetas, Nbs)
    est_list.append(est)
    cov_list.append(cov)

est_list, cov_list = np.array(est_list), np.array(cov_list)
k = np.arange(1, len(est_list) + 1)
deltaT  = est_list[:, 0] * np.sqrt(cov_list[:, 0, 0])   # absolute T error (nK)
deltaNb = est_list[:, 1] * np.sqrt(cov_list[:, 1, 1])   # absolute Nb error (atoms)

plt.plot(k, est_list[:, 0], color="C3", label="T Bayes estimate")
plt.fill_between(k, est_list[:, 0] - deltaT, est_list[:, 0] + deltaT, color="C3", alpha=0.3)
plt.axhline(T_true, color="grey", ls="--", label="T true")
plt.xlabel("k"); plt.ylabel(r"$\tilde\vartheta_k$ (nK)"); plt.legend()
plt.show()

plt.plot(k, est_list[:, 1], color="C0", label=r"$N_b$ Bayes estimate")
plt.fill_between(k, est_list[:, 1] - deltaNb, est_list[:, 1] + deltaNb, color="C0", alpha=0.3)
plt.axhline(Nb_true, color="grey", ls="--", label=r"$N_b$ true")
plt.xlabel("k"); plt.ylabel(r"$\tilde N_{b,k}$"); plt.legend()
plt.show()

def info_gain(prior, thetas, Nbs, q_ab, mu, n_max=100):
    # Returns (K_T, K_Nb); K_sum = K_T + K_Nb (W = identity on log-parameters).
    lT, lN = np.log(thetas)[:, None], np.log(Nbs)[None, :]
    m0 = np.array([trapz(prior*lT, thetas, Nbs), trapz(prior*lN, thetas, Nbs)])
    n = np.arange(n_max + 1)[:, None, None]
    lik = np.exp(log_likelihood(n, q_ab[None], mu)) * prior[None]     # (n, NT, NN)
    p_n = trapz(lik, thetas, Nbs)
    ok = p_n > 1e-300
    dT = trapz(lik*lT, thetas, Nbs)[ok] / p_n[ok] - m0[0]
    dN = trapz(lik*lN, thetas, Nbs)[ok] / p_n[ok] - m0[1]
    return np.sum(p_n[ok] * dT**2), np.sum(p_n[ok] * dN**2)

prior = np.exp(-0.5 * ((Nbs[None, :] - 6.6e3) / 0.3e3)**2) * np.ones((len(thetas), 1))
prior /= trapz(prior, thetas, Nbs)
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

plt.pcolormesh(Nbs, thetas, post, shading="auto")
plt.xlabel(r"$N_b$"); plt.ylabel("T (nK)"); plt.colorbar(label="posterior")
plt.show()
