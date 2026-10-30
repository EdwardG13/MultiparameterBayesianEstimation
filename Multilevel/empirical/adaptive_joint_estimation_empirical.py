# Run the adaptive (T, Nb) estimation for many seeds in parallel, one .npz per seed.
import os
# One BLAS/OpenMP thread per worker process, otherwise N workers x N threads oversubscribe the CPU.
# Must be set before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import csv
import tqdm
import numpy as np
from scipy.linalg import expm
from scipy.special import gammaln
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
#  Configuration (cheap constants only: this block runs again in every worker)
# ---------------------------------------------------------------------------

T_RANGE  = (500.0, 1500.0) # prior support for T (nK)
NB_RANGE = (2e3, 10e3) # prior support for Nb
NG = 501 # grid points per parameter
taus = np.linspace(0, 1.2, 21)  # interaction times (s), must be equally spaced

B_field = 50.0    # mG (only known fits)
USE_ENDO = True   # True -> include endothermic rates (only for B = 50 mG)

mu = 30                           # mean Cs atom number
T_true, Nb_true = 975.0, 6.6e3    # true T and Nb of the Rb bath

type_T,  T0,  sigma_T  = "flat", 1000.0, 150.0
type_Nb, Nb0, sigma_Nb = "flat", 6e3, 200
#type_Nb, Nb0, sigma_Nb = "gauss", 6e3, 1000
#type_Nb, Nb0, sigma_Nb = "lognormal", Nb_true, 500

n_shots = 5000
#M_min, M_max, M_frac = 100, 1000, 0.1
M_min, M_max, M_frac = 100, 100, 0
n_sig = 8
k_zoom = 1000        # start zooming the grid once this many shots are taken

seeds = range(1, 21)
n_workers = None     # None -> min(#seeds, #cores). Each worker needs ~0.5-1 GB at NG = 501.
VERBOSE = False      # per-block printout (interleaves between workers)
SKIP_EXISTING = True # don't redo seeds whose output file already exists

# ---------------------------------------------------------------------------
#  Functions
# ---------------------------------------------------------------------------

required = ["a0", "a1", "a2", "b1", "b2", "c1", "c2", "c3", "c4"]
optional = ["a3", "c5", "c6"]

def load_fit(B_field, folder="../data/fits", suffix=""):
    loaded = {}
    with open(f"{folder}/sigmav_fit_coeffs_B_{B_field}mG{suffix}.csv", "r") as file:
        for row in csv.reader(file, delimiter=","):
            try:
                loaded[row[0].strip()] = float(row[2])
            except (IndexError, ValueError):
                continue
    missing = set(required) - set(loaded)
    unknown = set(loaded) - set(required + optional)
    assert not missing, f"missing coefficients: {missing}"
    assert not unknown, f"unknown coefficients: {unknown}"
    fit = dict.fromkeys(optional, 0.0)
    fit.update(loaded)
    return fit

def sigv(fit, mF, T):
    return ((fit["a0"] + fit["a1"]*mF + fit["a2"]*mF**2 + fit["a3"]*mF**3)
          + (fit["b1"] + fit["c1"]*mF + fit["c2"]*mF**2 + fit["c5"]*mF**3) * T
          + (fit["b2"] + fit["c3"]*mF + fit["c4"]*mF**2 + fit["c6"]*mF**3) * T**2)

def rate(fit, mF, T, Nb):
    return 0.159 * sigv(fit, mF, T)*1e-10 * Nb * (T/1e9) ** (-1.5)

def build_Gamma(fit_down, fit_up, thetas, Nbs):
    T, N = np.asarray(thetas, float)[:, None], np.asarray(Nbs, float)[None, :]
    G = np.zeros((T.shape[0], N.shape[1], 7, 7))
    for i in range(7):
        mF = 3 - i
        if i < 6:
            r = rate(fit_down, mF, T, N)
            assert np.all(r >= 0), f"negative rate for mF={mF:+d}: min {r.min():.2e}"
            G[..., i + 1, i] += r
            G[..., i, i] -= r
        if fit_up is not None and i > 0:
            r = rate(fit_up, mF, T, N)
            assert np.all(r >= 0), f"negative rate for mF={mF:+d}: min {r.min():.2e}"
            G[..., i - 1, i] += r
            G[..., i, i] -= r
    return G

def populations(fit_down, fit_up, thetas, Nbs, taus):
    # q[a, c, b, i]: T = thetas[a], Nb = Nbs[c], tau = taus[b], level i (mF = 3 - i). Start in mF = +3.
    taus = np.asarray(taus, float)
    dtau = taus[1] - taus[0]
    G = build_Gamma(fit_down, fit_up, thetas, Nbs)
    P = expm(G * dtau)
    q = np.zeros(G.shape[:2] + (len(taus), 7))
    q[..., 0, 0] = 1.0
    for b in range(1, len(taus)):
        # batched (7x7)@(7,) over the whole grid at once, instead of a Python loop over grid points
        q[..., b, :] = (P @ q[..., b - 1, :, None])[..., 0]
    assert np.all(q > -1e-10) and np.all(q < 1 + 1e-10), "populations outside [0, 1]"
    return q

def trapz(f, thetas, Nbs):
    return np.trapezoid(np.trapezoid(f, Nbs, axis=-1), thetas, axis=-1)

def log_likelihood(n, q, mu):
    nu = np.maximum(mu * q, 1e-300)
    return n * np.log(nu) - nu - gammaln(n + 1)

def log_prior_1d(x, type, x0, sigma):
    if type == "flat":
        return np.zeros_like(x)
    elif type == "gauss":
        return -0.5 * ((x - x0) / sigma)**2
    elif type == "lognormal":
        s = sigma / x0
        return -0.5 * (np.log(x / x0) / s)**2 - np.log(x)
    raise ValueError(f"unknown prior type {type!r}")

def make_log_prior(thetas, Nbs):
    return (log_prior_1d(thetas, type_T, T0, sigma_T)[:, None]
            + log_prior_1d(Nbs, type_Nb, Nb0, sigma_Nb)[None, :])

def posterior(log_post, thetas, Nbs):
    p = np.exp(log_post - log_post.max())
    return p / trapz(p, thetas, Nbs)

def estimators(post, thetas, Nbs):
    lT, lN = np.log(thetas)[:, None], np.log(Nbs)
    m = np.array([trapz(post*lT, thetas, Nbs), trapz(post*lN, thetas, Nbs)])
    dT, dN = lT - m[0], lN - m[1]
    cov = np.array([[trapz(post*dT*dT, thetas, Nbs), trapz(post*dT*dN, thetas, Nbs)],
                    [trapz(post*dT*dN, thetas, Nbs), trapz(post*dN*dN, thetas, Nbs)]])
    return np.exp(m), cov

def info_gain(prior, thetas, Nbs, q_TN, mu):
    lT, lN = np.log(thetas)[:, None], np.log(Nbs)
    log_T_prior = trapz(prior * lT, thetas, Nbs)
    log_N_prior = trapz(prior * lN, thetas, Nbs)
    nu_max = mu * q_TN.max()
    if nu_max < 1e-8:
        return 0.0, 0.0
    n_max = int(np.ceil(nu_max + 6 * np.sqrt(nu_max) + 5))
    n = np.arange(n_max + 1)[:, None, None]
    joint = np.exp(log_likelihood(n, q_TN[None], mu)) * prior[None]
    p_n = trapz(joint, thetas, Nbs)
    ok = p_n > 1e-100
    post_n = joint[ok] / p_n[ok, None, None]
    log_T_post = trapz(post_n * lT, thetas, Nbs)
    log_N_post = trapz(post_n * lN, thetas, Nbs)
    K_T = np.sum(p_n[ok] * (log_T_post - log_T_prior) ** 2)
    K_N = np.sum(p_n[ok] * (log_N_post - log_N_prior) ** 2)
    return K_T, K_N

def prior_fisher_1d(x, type, x0, sigma):
    if type == "flat":
        return 0.0
    lx = np.log(x)
    log_p = log_prior_1d(x, type, x0, sigma) + lx
    p = np.exp(log_p - log_p.max()); p /= np.trapezoid(p, lx)
    score = np.gradient(log_p, lx)
    return np.trapezoid(p * score**2, lx)

def rebuild_log_post(thetas, Nbs, q, N_tot, m_tot):
    lp = make_log_prior(thetas, Nbs)
    for b, i in zip(*np.nonzero(m_tot)):
        nu = np.maximum(mu * q[:, :, b, i], 1e-300)
        lp += N_tot[b, i] * np.log(nu) - m_tot[b, i] * nu
    return lp

def zoom(post, thetas, Nbs, n_sig=8):
    est, cov = estimators(post, thetas, Nbs)
    sT, sN = np.sqrt(np.diag(cov))
    lT0, lN0 = np.log(est)
    lT = np.linspace(max(lT0 - n_sig*sT, np.log(T_RANGE[0])),  min(lT0 + n_sig*sT, np.log(T_RANGE[1])),  len(thetas))
    lN = np.linspace(max(lN0 - n_sig*sN, np.log(NB_RANGE[0])), min(lN0 + n_sig*sN, np.log(NB_RANGE[1])), len(Nbs))
    return np.exp(lT), np.exp(lN)

def block_schedule(n_shots, M_min, M_frac, M_max):
    Ms, k = [], 0
    while k < n_shots:
        M = min(max(M_min, int(M_frac * k)), M_max, n_shots - k)
        Ms.append(M); k += M
    return Ms

def out_path(seed):
    mantissa, exponent = f"{n_shots:.0e}".split("e")
    return (f"data/adaptive_run_B_{B_field}mG_{mantissa}e{int(exponent)}"
            f"_T-{type_T}_Nb-{type_Nb}_seed{seed}.npz")

# ---------------------------------------------------------------------------
#  Seed-independent setup: done once per worker process
# ---------------------------------------------------------------------------

_S = None

def setup():
    fit_down = load_fit(B_field)
    fit_up = load_fit(B_field, suffix="_endo") if USE_ENDO else None
    thetas0 = np.linspace(*T_RANGE, NG)
    Nbs0 = np.linspace(*NB_RANGE, NG)
    q0 = populations(fit_down, fit_up, thetas0, Nbs0, taus)

    q_true = np.clip(populations(fit_down, fit_up, [T_true], [Nb_true], taus)[0, 0], 0, None)

    F_prior = np.diag([prior_fisher_1d(thetas0, type_T, T0, sigma_T),
                       prior_fisher_1d(Nbs0, type_Nb, Nb0, sigma_Nb)])

    # Per-setting Poisson Fisher matrix in (ln T, ln Nb) at the true point
    h = 1e-4
    pop1 = lambda T, N: populations(fit_down, fit_up, [T], [N], taus)[0, 0]
    dq_T = (pop1(T_true*np.exp(h), Nb_true) - pop1(T_true*np.exp(-h), Nb_true)) / (2*h)
    dq_N = (pop1(T_true, Nb_true*np.exp(h)) - pop1(T_true, Nb_true*np.exp(-h))) / (2*h)
    g = np.stack([dq_T, dq_N], axis=-1)
    F_set = mu * g[..., :, None] * g[..., None, :] / np.maximum(q_true, 1e-12)[..., None, None]
    F_set[q_true < 1e-8] = 0.0

    return dict(fit_down=fit_down, fit_up=fit_up, thetas0=thetas0, Nbs0=Nbs0, q0=q0,
                q_true=q_true, F_prior=F_prior, F_set=F_set)

def init_worker():
    global _S
    _S = setup()

# ---------------------------------------------------------------------------
#  One adaptive run
# ---------------------------------------------------------------------------

def run_seed(seed):
    S = _S
    path = out_path(seed)
    rng = np.random.default_rng(seed)

    # Fresh grid for every seed (zoom() replaces these, it never mutates them)
    thetas, Nbs, q = S["thetas0"], S["Nbs0"], S["q0"]
    q_true, F_set = S["q_true"], S["F_set"]

    log_post = make_log_prior(thetas, Nbs)
    post = posterior(log_post, thetas, Nbs)
    N_tot = np.zeros((len(taus), 7))
    m_tot = np.zeros((len(taus), 7))
    k_list, est_list, cov_list, F_list = [], [], [], []
    chosen_b, chosen_i, M_list, K_maps = [], [], [], []
    F_cum = S["F_prior"].copy()
    k = 0
    s = slice(None, None, 2)  # decision uses every second grid point

    Ms = block_schedule(n_shots, M_min, M_frac, M_max)
    for j, M in enumerate(Ms):
        if k >= k_zoom:
            thetas, Nbs = zoom(post, thetas, Nbs, n_sig)
            q = populations(S["fit_down"], S["fit_up"], thetas, Nbs, taus)
            log_post = rebuild_log_post(thetas, Nbs, q, N_tot, m_tot)
            post = posterior(log_post, thetas, Nbs)

        post_c = post[s, s] / trapz(post[s, s], thetas[s], Nbs[s])
        K = np.zeros((len(taus), 7))
        for b in range(len(taus)):
            for i in range(7):
                K[b, i] = sum(info_gain(post_c, thetas[s], Nbs[s], q[s, s, b, i], mu))
        b, i = np.unravel_index(np.argmax(K), K.shape)

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

        if VERBOSE:
            print(f"[seed {seed} | {j + 1}/{len(Ms)}] k={k:8d} M={M:6d} tau={taus[b]:.3f}s mF={3 - i:+d} "
                  f"T={est[0]:.1f} ± {est[0]*np.sqrt(cov[0, 0]):.1f}  "
                  f"Nb={est[1]:.0f} ± {est[1]*np.sqrt(cov[1, 1]):.1f}", flush=True)

    tmp = path[:-4] + ".tmp.npz"   # write then rename, so a killed run never leaves a half-written file
    np.savez(tmp,
        thetas=thetas, Nbs=Nbs, taus=taus, mu=mu, T_true=T_true, Nb_true=Nb_true, seed=seed,
        k=np.array(k_list), M=np.array(M_list),
        est=np.array(est_list), cov=np.array(cov_list), F=np.array(F_list), F_prior=S["F_prior"],
        b=np.array(chosen_b), i=np.array(chosen_i), N_tot=N_tot, m_tot=m_tot,
        K_maps=np.array(K_maps), post=post)
    os.replace(tmp, path)
    return path

# ---------------------------------------------------------------------------
#  Main: farm the seeds out to worker processes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    todo = [sd for sd in seeds if not (SKIP_EXISTING and os.path.exists(out_path(sd)))]
    if not todo:
        sys.exit("all seeds already done")
    nw = n_workers or min(len(todo), os.cpu_count())
    print(f"{len(todo)} seeds on {nw} workers")

    with ProcessPoolExecutor(max_workers=nw, initializer=init_worker) as ex:
        futures = {ex.submit(run_seed, sd): sd for sd in todo}
        for f in tqdm.tqdm(as_completed(futures), total=len(futures), file=sys.stdout):
            sd = futures[f]
            try:
                tqdm.tqdm.write(f"seed {sd} done -> {f.result()}")
            except Exception as e:
                tqdm.tqdm.write(f"seed {sd} FAILED: {e!r}")