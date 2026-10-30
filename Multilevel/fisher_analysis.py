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

def dsigv_dT(fit, mF, T):
    return ((fit["b1"] + fit["c1"]*mF + fit["c2"]*mF**2 + fit["c5"]*mF**3)
          + 2*(fit["b2"] + fit["c3"]*mF + fit["c4"]*mF**2 + fit["c6"]*mF**3) * T)

# Spin-exchange rates out of state mF (Hz).
def rate(fit, mF, T, Nb):
    # This is the product of <sigma v> and the mean density of the Rb bath, here taken as 0.159 * Nb T^-3/2
    return 0.159 * sigv(fit,mF,T)*1e-10 * Nb * (T/1e9) ** (-1.5)

# Build rate matrix describing the (classical) population dynamics of the 7 mF levels
def build_Gamma(fit_down, fit_up, thetas, Nbs):
    # Returns rate matrices at every grid point, shape (len(T), len(N), 7, 7)
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

def build_Gamma2(fit, T, Nb):
    G = np.zeros((7, 7))
    for i in range(6):               # down only: mF -> mF-1, mF = 3 - i
        r = rate(fit, 3 - i, T, Nb)
        G[i + 1, i] += r
        G[i, i] -= r
    return G

def populations(fit_down, fit_up, thetas, Nbs, taus):
    # This precomputes the population probabilities q for every point on the parameter space and for all times tau.
    # q[a, c, b, i] where T = thetas[a], Nb = Nbs[c], tau = taus[b], level i (mF = 3 - i). Start in mF = +3.
    # It works by stepping through the tau array, using the formula q(t+dt)=exp(Gamma(t+dt))q_0= P q(t), where P=exp(Gamma dt) and q(t)=exp(Gamma t)q_0.
    # Importantly, the tau grid must be equally spaced.
    taus = np.asarray(taus, float)
    dtau = taus[1] - taus[0]
    G = build_Gamma(fit_down, fit_up, thetas, Nbs) # (NT, NN, 7, 7)
    P = expm(G * dtau) # propagator over one tau step, one per grid point
    q = np.zeros(G.shape[:2] + (len(taus), 7))
    q[..., 0, 0] = 1.0 # tau = 0: all population in mF = +3
    for b in range(1, len(taus)):
        q[..., b, :] = (P @ q[..., b - 1, :, None])[..., 0] # batched (7x7)@(7,) over the whole grid at once, instead of a Python loop over grid points
        # q[..., b, :] = np.einsum('acij,acj->aci', P, q[..., b - 1, :]) # q(tau_b) = P q(tau_{b-1})
        # for a in range(len(thetas)):
        #     for c in range(len(Nbs)):
        #         q[a, c, b, :] = P[a, c] @ q[a, c, b - 1, :]  # (7x7) @ (7,) -> (7,)
    assert np.all(q > -1e-10) and np.all(q < 1 + 1e-10), "populations outside [0, 1]"
    return q

def populations2(fit, T, Nb, taus):
    # This computes the population probabilities q for a single (T,Nb) for all times taus.
    G = build_Gamma2(fit, T, Nb)
    q0 = np.zeros(7); q0[0] = 1.0
    return np.array([expm(G*t) @ q0 for t in taus])      # (Ntau, 7)

# Poisson log-likelihood
def log_likelihood(n, q, mu):
    # log p(n | theta) for every theta. q is the vector q_{mF}(theta, tau).
    nu = np.maximum(mu * q, 1e-300) # avoid log(0) when a level is empty
    return n * np.log(nu) - nu - gammaln(n + 1)

# ---------------------------------------------------------------------------
#  Checking the q gradient formula in terms of w
# ---------------------------------------------------------------------------

gamma = 1.5
T, Nb = 975.0, 6.6e3
mu = 30
taus = np.linspace(0, 1.2, 21)

B_field = 400.0 # mG (can only choose from known fits)
fit = load_fit(B_field) # Read fit coefficients for the rates for a given B field
fit_up = None # Replace None by fit_up for endothermic rates (only for B=50mG)
#fit_up = load_fit(B_field,suffix="_endo")

def grad_q_FD(T, Nb, taus):
    # Calculate grad_theta q via finite differences
    hT, hN = 1e-5*T, 1e-5*Nb
    dT = (populations2(fit, T+hT, Nb, taus) - populations2(fit, T-hT, Nb, taus)) / (2*hT)
    dN = (populations2(fit, T, Nb+hN, taus) - populations2(fit, T, Nb-hN, taus)) / (2*hN)
    return np.stack([dT, dN], axis=-1) # (Ntau, 7, 2)

def weights(T, Nb, tau, nt=2001):
    # w[mF, n] = int_0^tau dt' [ e^{G(tau-t')} Gamma_n E_n e^{G t'} q0 ]_mF  (trapezoid rule)
    G = build_Gamma2(fit, T, Nb)
    q0 = np.zeros(7); q0[0] = 1.0
    ts = np.linspace(0, tau, nt)
    U = np.array([expm(G*t) for t in ts]) # U[k] = e^{G t_k}; e^{G(tau-t_k)} = U[-1-k]
    w = np.zeros((7, 6))
    for n in range(6):
        # channel n: mF = 3-n -> 2-n
        E = np.zeros((7, 7)); E[n+1, n] = 1; E[n, n] = -1
        Gn = rate(fit, 3 - n, T, Nb)
        integrand = np.array([U[-1-k] @ (Gn*E) @ U[k] @ q0 for k in range(nt)])
        w[:, n] = np.trapezoid(integrand, ts, axis=0)
    return w

def grad_q_weights(T, Nb, tau):
    w = weights(T, Nb, tau)
    s = np.array([T*dsigv_dT(fit, 3-n, T)/sigv(fit, 3-n, T) for n in range(6)])
    dlogG = np.stack([(-gamma + s)/T, np.full(6, 1/Nb)], axis=-1) # (6, 2)
    return w @ dlogG, w, s # (7, 2)

g_FD = grad_q_FD(T, Nb, taus)
print("Grad q in terms of weights vs true (finite differences) (max |rel err| over mF, both components)")
for b in [2, 5, 10, 20]:
    grad_w, w, s = grad_q_weights(T, Nb, taus[b])
    qb = populations2(fit, T, Nb, [taus[b]])[0]
    ok = qb > 1e-8  # populated levels only
    err = np.abs(grad_w[ok] - g_FD[b][ok]) / np.abs(g_FD[b][ok])
    print(f"  tau={taus[b]:.2f}s  max rel err = {err.max():.1e}  min q = {qb.min():.1e}")

# Consistency of Eq. 19: sum_n w_{mF n} = tau (Gamma q)_mF
b = 10
grad_w, w, s = grad_q_weights(T, Nb, taus[b])
G = build_Gamma2(fit, T, Nb); qb = populations2(fit, T, Nb, [taus[b]])[0]
print("  check sum_n w = tau*(Gamma q):", np.allclose(w.sum(1), taus[b]*G@qb, atol=1e-6))
print("  check mF=+3 row of w has only w_33 nonzero:", np.allclose(w[0, 1:], 0))

# ---------------------------------------------------------------------------
# Fisher matrices for every setting (tau, mF), in log-parameters
# ---------------------------------------------------------------------------
q = populations2(fit, T, Nb, taus)  # (Ntau, 7)
D = np.diag([T, Nb])  # d/dlogT = T d/dT etc.
F = np.zeros((len(taus), 7, 2, 2))
for b in range(len(taus)):
    for i in range(7):
        if q[b, i] > 1e-8:
            a = D @ g_FD[b, i]
            F[b, i] = mu * np.outer(a, a) / q[b, i]

eigs = np.linalg.eigvalsh(F)  # (Ntau, 7, 2), ascending
trF = eigs.sum(-1)
print("\nSingle settings (log-parameters)")
print(f"  max over settings of lambda_min / lambda_max = {np.max(eigs[..., 0] / np.maximum(eigs[..., 1], 1e-300)):.1e}  (rank one)")
b, i = np.unravel_index(np.argmax(trF), trF.shape)
print(f"  max tr F:  tau={taus[b]:.2f}s, mF={3 - i:+d}, tr F={trF[b, i]:.3g}")

# Pairs of settings: F_tot = F_lam + F_lam'
settings = [(b, i) for b in range(len(taus)) for i in range(7) if q[b, i] > 1e-8]
best_min, best_tr = (-1, None), (np.inf, None)
for k1 in range(len(settings)):
    for k2 in range(k1 + 1, len(settings)):
        Ft = F[settings[k1]] + F[settings[k2]]
        lmin = np.linalg.eigvalsh(Ft)[0]
        if lmin > best_min[0]:
            best_min = (lmin, (settings[k1], settings[k2]))
        if lmin > 1e-12 * np.trace(Ft):
            tri = np.trace(np.linalg.inv(Ft))
            if tri < best_tr[0]:
                best_tr = (tri, (settings[k1], settings[k2]))


def fmt(pair): return ", ".join(f"(tau={taus[b]:.2f}s, mF={3 - i:+d})" for b, i in pair)

print("\nPart 1: pairs of settings")
print(f"  max lambda_min:  {fmt(best_min[1])}, lambda_min={best_min[0]:.3g}")
print(f"  min tr F^-1:     {fmt(best_tr[1])}, tr F^-1={best_tr[0]:.3g}")

# Check Eq. 24 on the best pair: det = mu^2/(q q') (D'S - DS')^2 / (T Nb)^2   (raw units)
(b1, i1), (b2, i2) = best_tr[1]


def DS(b, i):
    _, w, s = grad_q_weights(T, Nb, taus[b]);
    return w[i].sum(), (s * w[i]).sum()


D1, S1 = DS(b1, i1);
D2, S2 = DS(b2, i2)
a1 = np.sqrt(mu / q[b1, i1]) * g_FD[b1, i1];
a2 = np.sqrt(mu / q[b2, i2]) * g_FD[b2, i2]
det_num = np.linalg.det(np.outer(a1, a1) + np.outer(a2, a2))
pref = mu ** 2 / (q[b1, i1] * q[b2, i2])
print("\nEq. 24 check (raw units):")
print(f"  numerical det = {det_num:.6e}")
print(f"  Eq. 24 formula: mu^2/(q q') (D'S-DS')^2/(T Nb)^2  = {pref * (D2 * S1 - D1 * S2) ** 2 / (T * Nb) ** 2:.6e}")

# ---------------------------------------------------------------------------
# Fisher matrices for every setting (tau, mF)
# ---------------------------------------------------------------------------

T, Nb = 975.0, 6.6e3
q = populations2(fit, T, Nb, taus)  # (Ntau, 7)
thetas = np.linspace(500, 1500, 201) # Temperature hypothesis grid (nK)
Nbs = np.linspace(2e3, 10e3, 201) # Bath atom hypothesis grid (no. atoms)
taus = np.linspace(0, 1.2, 21) # Interaction time (s)

# Per-setting Fisher matrix for estimation of (T, Nb) evaluated at the true parameters
# F is calculated for all control parameters at once: F is (len(taus),7,2,2) and F[b, i] is 2x2
h = 1e-4
q_p = populations(fit,fit_up,[T+h], [Nb], taus)[0, 0]
q_m = populations(fit,fit_up,[T-h], [Nb], taus)[0, 0]
dq_T = (q_p - q_m) / (2*h)  # dq/dT
q_p = populations(fit,fit_up,[T], [Nb+h], taus)[0, 0]
q_m = populations(fit,fit_up,[T], [Nb-h], taus)[0, 0]
dq_N = (q_p - q_m) / (2*h) # dq/dNb
g = np.stack([dq_T, dq_N], axis=-1) # (Ntau, 7, 2)
F_set = mu * g[..., :, None] * g[..., None, :] / np.maximum(q, 1e-12)[..., None, None] # Poisson Fisher
F_set[q < 1e-8] = 0.0 # For empty levels (e.g. tau = 0, mF < 3), q has rounding error

pops=populations(fit,fit_up,thetas,Nbs,taus)


eigenvalues, eigenvectors = np.linalg.eig(F_set[1,:])

print(eigenvalues)

eigenvalues_list = np.linalg.eigvals(F_set[:,:])

print(np.shape(eigenvalues_list))


