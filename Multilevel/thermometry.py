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

# Populations on a (theta, tau) grid
def populations(thetas, taus, Nb, pop_init):
    # q[a, b, i] = population of level i (mF = 3 - i) at theta_a, tau_b.
    q = np.zeros((len(thetas), len(taus), 7))
    for a, theta in enumerate(thetas):
        G = build_rate_matrix(theta, Nb) # build once per theta
        for b, tau in enumerate(taus):
            q[a, b, :] = linalg.expm(G * tau) @ pop_init
    return q

def populations_bateman(thetas, taus, Nb):
    # Bateman solution for the chain starting in |+3> (index 0).
    # q[a, b, i] = population of level i (mF = 3 - i) at theta_a, tau_b.
    thetas, taus = np.asarray(thetas, float), np.asarray(taus, float)
    mF = 3 - np.arange(6)
    lam = rate(mF[None, :], thetas[:, None], Nb)          # (Ntheta, 6): rate out of level i
    diff = lam[:, :, None] - lam[:, None, :]              # diff[a,l,k] = lam_l - lam_k
    diff[:, np.arange(6), np.arange(6)] = 1.0
    P = np.cumprod(diff, axis=1)                          # prod_{l<=i, l!=k} (lam_l - lam_k)
    N = np.concatenate([np.ones((len(thetas), 1)),
                        np.cumprod(lam[:, :-1], axis=1)], axis=1)   # prod_{j<i} lam_j
    C = np.tril(N[:, :, None] / P)                        # Bateman coefficients C[a,i,k], k<=i
    E = np.exp(-lam[:, None, :] * taus[None, :, None])    # (Ntheta, Ntau, 6)
    q = np.empty((len(thetas), len(taus), 7))
    q[..., :6] = np.einsum('aik,abk->abi', C, E)
    q[..., 6] = 1.0 - q[..., :6].sum(-1)                  # absorbing |-3>
    return q

thetas = np.linspace(500, 1500, 201) # temperature hypothesis grid (nK)
taus   = np.linspace(0, 1.0, 101) # Interaction time (s)
Nb, mu = 6.6e3, 30 # Rb bath atom number and mean Cs atom number
#q = populations(thetas, taus, Nb, pop_init)
q = populations_bateman(thetas, taus, Nb)

# Poisson log-likelihood
def log_likelihood(n, q, mu):
    # log p(n | theta) for every theta. q is the vector q_m(theta, tau).
    nu = np.maximum(mu * q, 1e-300) # avoid log(0) when a level is empty
    return n * np.log(nu) - nu - gammaln(n + 1)

# Posterior on the theta grid
def posterior(thetas, log_prior, log_lik):
    log_post = log_prior + log_lik
    post = np.exp(log_post - log_post.max()) # subtract max before exp
    return post / np.trapezoid(post, thetas)

# Test: simulate data at T_true, measure |+3> at tau = 250 ms
rng = np.random.default_rng(3)
T_true, b, i = 975.0, 25, 0 # tau index, level index (mF = +3)
q_true = populations([T_true], [taus[b]], Nb, pop_init)[0, 0, i]


def estimator(post, thetas):
    # Optimal estimator for log loss: theta_u * exp<log(theta/theta_u)> = exp<log theta>.
    return np.exp(np.trapezoid(post * np.log(thetas), thetas))

def log_error(post, thetas, est):
    # Mean logarithmic error  eps = ∫ p(theta|n) log^2(est/theta) dtheta.
    return np.trapezoid(post * np.log(est / thetas)**2, thetas)

log_post = np.zeros_like(thetas) # flat prior since log p = const


est, est_ml, error = [], [], []
for k in range(100):
    n = rng.poisson(mu * q_true) #+ rng.normal(0,10)
    log_post += log_likelihood(n, q[:, b, i], mu) # sequential update
    post = posterior(thetas, log_post, 0.0)
    est.append(estimator(post, thetas))
    error.append(log_error(post, thetas, est[k]))
    est_ml.append(thetas[np.argmax(log_post)]) # maximum-likelihood estimate

est, error = np.array(est), np.array(error)
k = np.arange(1, len(est) + 1)
delta = est * np.sqrt(error) # absolute error in nK

plt.plot(k, est, color="C3", label="Bayes estimate")
plt.fill_between(k, est - delta, est + delta, color="C3", alpha=0.3)
plt.plot(k, est_ml, color="C0", lw=0.8, label="max likelihood")   # optional
plt.axhline(T_true, color="grey", ls="--", label="T true")
plt.xlabel("k"); plt.ylabel(r"$\tilde\vartheta_k$ (nK)"); plt.legend()
plt.show()


def info_gain(prior, thetas, q_theta, mu, n_max=100):
    m0 = np.trapezoid(prior * np.log(thetas), thetas)  # <log theta> under prior
    K = 0.0
    for n in range(n_max + 1):  # all possible outcomes
        lik = np.exp(log_likelihood(n, q_theta, mu)) # p(n|theta) for all theta
        p_n = np.trapezoid(lik * prior, thetas) # evidence p(n)
        if p_n < 1e-300:
            continue
        post_n = lik * prior / p_n  # posterior if n observed
        m_n = np.trapezoid(post_n * np.log(thetas), thetas)
        K += p_n * (m_n - m0)**2
    return K

prior = np.ones_like(thetas) / (thetas[-1] - thetas[0])  # flat prior

for i in range(7):
    K = [info_gain(prior, thetas, q[:, b, i], mu) for b in range(len(taus))]
    plt.plot(taus * 1e3, K, label=f"$m_F = {3 - i:+d}$")   # index i <-> mF = 3 - i

plt.xlabel(r"$\tau$ (ms)")
plt.ylabel(r"$K(\tau)$")
plt.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.show()

for i in range(7):
    K = [info_gain(post, thetas, q[:, b, i], mu) for b in range(len(taus))]
    plt.plot(taus * 1e3, K, label=f"$m_F = {3 - i:+d}$")   # index i <-> mF = 3 - i

plt.xlabel(r"$\tau$ (ms)")
plt.ylabel(r"$K(\tau)$")
plt.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.show()

