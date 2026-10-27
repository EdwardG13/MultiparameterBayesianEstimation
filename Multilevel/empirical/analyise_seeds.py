import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
#  Choose which runs to load (must match adaptive_seeds.py's out_path)
# ---------------------------------------------------------------------------
B_field = 400.0
n_shots_str = "1e4"
type_T, type_Nb = "flat", "flat"

tag = f"B_{B_field}mG_{n_shots_str}_T-{type_T}_Nb-{type_Nb}"
files = [f for f in glob.glob(f"data/adaptive_run_{tag}_seed*.npz") if ".tmp" not in f]  # skip half-written files
files.sort(key=lambda f: int(f.split("seed")[-1].split(".")[0]))
assert files, f"no files found for {tag}"
runs = [np.load(f) for f in files]
seeds = np.array([int(f.split("seed")[-1].split(".")[0]) for f in files])
S = len(runs)
print(f"{S} seeds loaded: {seeds.tolist()}")

plot_dir = f"plots/plots_{tag}_{S}seeds"
os.makedirs(plot_dir, exist_ok=True)

def save_plot(name):
    plt.savefig(os.path.join(plot_dir, f"{name}.pdf"), bbox_inches="tight", dpi=200)
    plt.savefig(os.path.join(plot_dir, f"{name}.png"), bbox_inches="tight", dpi=200)

# ---------------------------------------------------------------------------
#  Stack all seeds. Every seed uses the same block schedule, so k is shared.
# ---------------------------------------------------------------------------
d0 = runs[0]
k = d0["k"]
assert all(np.array_equal(d["k"], k) for d in runs), "seeds have different block schedules"
taus = d0["taus"]
T_true, Nb_true = float(d0["T_true"]), float(d0["Nb_true"])
truth = np.array([T_true, Nb_true])
F_prior = d0["F_prior"]

est = np.stack([d["est"] for d in runs])   # (S, Nk, 2)       estimates (T, Nb)
cov = np.stack([d["cov"] for d in runs])   # (S, Nk, 2, 2)    posterior Cov[ln T, ln Nb]
F   = np.stack([d["F"]   for d in runs])   # (S, Nk, 2, 2)    cumulative Fisher at the true point

# Everything below is in log units, where the estimator and posterior covariance live.
# sqrt of a log-variance is a relative error: sqrt(Var[ln T]) ~ dT/T.
err = np.log(est) - np.log(truth)                                     # (S, Nk, 2)  ln(est/true)

mse        = np.mean(err[..., :, None] * err[..., None, :], axis=0)   # (Nk,2,2) empirical MSE matrix: E[(est-true)(est-true)^T]
mean_err   = err.mean(axis=0)                                         # (Nk,2)   bias of ln estimate
var_est    = err.var(axis=0, ddof=1)                                  # (Nk,2)   sample variance of estimates about their own mean
post_var   = cov.mean(axis=0)                                         # (Nk,2,2) average posterior covariance (average the VARIANCES, not the sigmas)
sem        = np.sqrt(var_est / S)                                     # (Nk,2)   standard error of the seed-averaged estimate

# CRB for the adaptive protocol: the Fisher information of the whole adaptive record is the
# data-averaged cumulative Fisher E[sum_j F(setting_j)], so the bound is inv(mean_s F_s).
F_mean = F.mean(axis=0)
crb = np.full_like(F_mean, np.nan)
ok = np.linalg.det(F_mean) > 0
crb[ok] = np.linalg.inv(F_mean[ok])

# Monte-Carlo uncertainty on the MSE itself (it's an average of S numbers): sd(err^2)/sqrt(S)
mse_sd = np.std(err**2, axis=0, ddof=1) / np.sqrt(S)                  # (Nk,2)

# ---------------------------------------------------------------------------
#  Summary at the final k
# ---------------------------------------------------------------------------
names = ["T", "Nb"]
print(f"\nFinal k = {k[-1]}, S = {S} seeds  (all errors relative, i.e. in ln units)")
print(f"{'':4s}{'mean est':>12s}{'bias':>10s}{'rms err':>10s}{'spread':>10s}{'<post sd>':>11s}{'CRB sd':>9s}{'SEM':>9s}")
for p in range(2):
    print(f"{names[p]:4s}{np.exp(np.log(est[:, -1, p]).mean()):12.1f}"
          f"{mean_err[-1, p]:10.3%}{np.sqrt(mse[-1, p, p]):10.3%}{np.sqrt(var_est[-1, p]):10.3%}"
          f"{np.sqrt(post_var[-1, p, p]):11.3%}{np.sqrt(crb[-1, p, p]):9.3%}{sem[-1, p]:9.3%}")
print("rms err^2 = bias^2 + spread^2 (up to S/(S-1));  <post sd> ~ rms err if posteriors are calibrated")

# ---------------------------------------------------------------------------
#  1) Final estimate of every seed, with its own posterior error bar, normalised to the truth
# ---------------------------------------------------------------------------
x = np.arange(S)
gap = S + 3
fig, ax = plt.subplots(figsize=(max(5, 0.35 * 2 * S), 4))
for p, (m, c, lab, off) in enumerate([("o", "C3", r"$T$", 0), ("s", "C0", r"$N_b$", gap)]):
    y = est[:, -1, p] / truth[p]
    yerr = y * np.sqrt(cov[:, -1, p, p])          # each seed's own posterior sd (relative)
    ax.errorbar(x + off, y, yerr=yerr, fmt=m, color=c, ms=5, capsize=2, lw=1, label=lab)
    # empirical rms error band around the truth, for comparison with the individual bars
    r = np.sqrt(mse[-1, p, p])
    ax.fill_between([off - 0.7, off + S - 0.3], np.exp(-r), np.exp(r), color=c, alpha=0.12, lw=0)
ax.axhline(1, color="grey", ls="--")
ax.set_xticks([(S - 1) / 2, gap + (S - 1) / 2]); ax.set_xticklabels([r"$T$", r"$N_b$"])
ax.set_ylabel(r"estimate / true value"); ax.set_ylim(bottom=0)
ax.set_title(f"Final estimates, {S} seeds, k = {k[-1]}  (band: empirical rms error)", fontsize=9)
save_plot("final_estimates_scatter")
plt.show()

# ---------------------------------------------------------------------------
#  2) Estimates vs k: every seed (thin) + seed-average and empirical spread
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for p, (ax, c, lab) in enumerate(zip(axes, ("C3", "C0"), (r"$T$ (nK)", r"$N_b$"))):
    ax.semilogx(k, est[:, :, p].T, color=c, lw=0.5, alpha=0.35)
    gm = np.exp(np.log(est[:, :, p]).mean(axis=0))            # geometric mean (mean in ln units)
    sd = np.sqrt(var_est[:, p])
    ax.semilogx(k, gm, color="k", lw=1.5, label="seed average")
    ax.fill_between(k, gm * np.exp(-sd), gm * np.exp(sd), color=c, alpha=0.3, label="empirical spread")
    ax.axhline(truth[p], color="grey", ls="--", label="true")
    ax.set_xlabel("k"); ax.set_ylabel(lab)
axes[0].legend(fontsize=8)
plt.tight_layout()
save_plot("estimates_vs_k")
plt.show()

# ---------------------------------------------------------------------------
#  3) Error vs k: empirical MSE vs average posterior variance vs CRB
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
tr = lambda A: A[..., 0, 0] + A[..., 1, 1]
panels = [(mse[:, 0, 0], post_var[:, 0, 0], var_est[:, 0], crb[:, 0, 0], mse_sd[:, 0], r"$\epsilon_T$"),
          (mse[:, 1, 1], post_var[:, 1, 1], var_est[:, 1], crb[:, 1, 1], mse_sd[:, 1], r"$\epsilon_{N_b}$"),
          (tr(mse), tr(post_var), var_est.sum(1), tr(crb), np.sqrt((mse_sd**2).sum(1)), r"$\epsilon_T+\epsilon_{N_b}$")]
for ax, (m, pv, ve, c, msd, t) in zip(axes, panels):
    ax.loglog(k, m, color="C3", label="empirical MSE")
    ax.fill_between(k, np.maximum(m - msd, 1e-12), m + msd, color="C3", alpha=0.2)
    ax.loglog(k, ve, color="C3", ls=":", label="empirical variance")
    ax.loglog(k, pv, color="C0", label="mean posterior var")
    ax.loglog(k, c, color="k", ls="--", label=r"CRB $(\mathbb{E}\sum F)^{-1}$")
    ax.set_title(t); ax.set_xlabel("k")
axes[0].legend(fontsize=8); plt.tight_layout()
save_plot("errors_vs_k")
plt.show()

# ---------------------------------------------------------------------------
#  4) Best-measured direction c = u . (ln T, ln Nb)
# ---------------------------------------------------------------------------
w, V = np.linalg.eigh(F_mean[-1] - F_prior)
u = V[:, -1]
proj = lambda A: np.einsum("i,...ij,j->...", u, A, u)
e_c = err @ u                                          # (S, Nk)
plt.loglog(k, np.mean(e_c**2, axis=0), color="C3", label="empirical MSE")
plt.loglog(k, proj(post_var), color="C0", label="mean posterior var")
crb_c = proj(crb)
plt.loglog(k, crb_c, color="k", ls="--", label="CRB")
plt.loglog(k, crb_c[-1] * k[-1] / k, color="grey", lw=0.8, label=r"$\propto 1/k$")
plt.xlabel("k"); plt.ylabel(rf"Var$[{u[1]:.2f}\,\ln N_b {u[0]:+.2f}\,\ln T]$"); plt.legend(fontsize=8)
save_plot("var_best_direction")
plt.show()

# ---------------------------------------------------------------------------
#  5) Settings chosen, pooled over seeds
# ---------------------------------------------------------------------------
m_tot = np.stack([d["m_tot"] for d in runs]).sum(0)
freq = m_tot / m_tot.sum()
mF_axis = 3 - np.arange(7)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].bar(mF_axis, freq.sum(0)); axes[0].set_xlabel(r"$m_F$"); axes[0].set_ylabel("fraction of shots (all seeds)")
axes[1].bar(taus, freq.sum(1), width=0.8 * (taus[1] - taus[0])); axes[1].set_xlabel(r"$\tau$ (s)")
im = axes[2].pcolormesh(taus, mF_axis, freq.T, shading="nearest", cmap="Greys")
axes[2].set_xlabel(r"$\tau$ (s)"); axes[2].set_ylabel(r"$m_F$"); plt.colorbar(im, ax=axes[2])
plt.tight_layout()
save_plot("setting_histograms_pooled")
plt.show()