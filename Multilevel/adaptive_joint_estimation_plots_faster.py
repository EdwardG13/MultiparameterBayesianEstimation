import os
import numpy as np
import matplotlib.pyplot as plt

# Choose file to load
B_field=500.0
n_shots_str="1e5"
seed=42

d = np.load(f"data/adaptive_run_B_{B_field}mG_{n_shots_str}_seed{seed}.npz")
#d = np.load("adaptive_run.npz")

plot_dir = f"plots/plots_B_{B_field}mG_{n_shots_str}_seed{seed}"
os.makedirs(plot_dir, exist_ok=True) # creates the folder; no error if it already exists


thetas, Nbs, taus = d["thetas"], d["Nbs"], d["taus"]
est_list, cov_list, F_list, F_prior = d["est"], d["cov"], d["F"], d["F_prior"]
chosen_b, chosen_i, K_maps, post = d["b"], d["i"], d["K_maps"], d["post"]
T_true, Nb_true = float(d["T_true"]), float(d["Nb_true"]) # scalars come back as 0-d arrays
M = d["M"]
k = d["k"]
chosen_tau, chosen_mF = taus[chosen_b], 3 - chosen_i
mF_axis = 3 - np.arange(7)
deltaT  = est_list[:, 0] * np.sqrt(cov_list[:, 0, 0]) # Absolute T error (nK)
deltaNb = est_list[:, 1] * np.sqrt(cov_list[:, 1, 1]) # Absolute Nb error (atoms)

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_plot(name):
    # Save the current figure to plots/<name> as a pdf and png
    plt.savefig(os.path.join(plot_dir, f"{name}.pdf"), bbox_inches="tight", dpi=200)
    plt.savefig(os.path.join(plot_dir, f"{name}.png"), bbox_inches="tight", dpi=200)

### Plot T estimate (with error bars)
plt.plot(k, est_list[:, 0], color="C3", label="T Bayes estimate")
plt.fill_between(k, est_list[:, 0] - deltaT, est_list[:, 0] + deltaT, color="C3", alpha=0.3)
plt.axhline(T_true, color="grey", ls="--", label="T true")
plt.xlabel("k"); plt.ylabel(r"$\tilde\vartheta_k$ (nK)"); plt.legend()
save_plot("T_estimate")
plt.show()

### Plot Nb estimate (with error bars)
plt.plot(k, est_list[:, 1], color="C0", label=r"$N_b$ Bayes estimate")
plt.fill_between(k, est_list[:, 1] - deltaNb, est_list[:, 1] + deltaNb, color="C0", alpha=0.3)
plt.axhline(Nb_true, color="grey", ls="--", label=r"$N_b$ true")
plt.xlabel("k"); plt.ylabel(r"$\tilde N_{b,k}$"); plt.legend()
save_plot("Nb_estimate")
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
save_plot("CRB")
plt.show()

# Variance of the combination the data actually measure: c = u . (ln T, ln Nb)
w, V = np.linalg.eigh(F_list[-1] - F_prior) # data-only Fisher
u = V[:, -1] # its best-measured direction (u[0]/u[1] = beta)
var_c = np.array([u @ C @ u for C in cov_list]) # Bayes posterior variance of c
crb_c = np.array([u @ C @ u for C in crb]) # CRB for c

plt.loglog(k, var_c, color="C3", label="Bayes")
plt.loglog(k, crb_c, color="k", ls="--", label="CRB")
plt.loglog(k, crb_c[-1] * k[-1] / k, color="grey", lw=0.8, label=r"$\propto 1/k$")
plt.xlabel("k"); plt.ylabel(r"Var$[\ln N_b + \beta \ln T]$")
plt.legend()
save_plot("var_best_direction")
plt.show()

### Plot final posterior as a colour map
plt.pcolormesh(Nbs, thetas, post, shading="auto")
plt.xlabel(r"$N_b$"); plt.ylabel("T (nK)"); plt.colorbar(label="posterior")
save_plot("final_posterior")
plt.show()

### Plot chosen settings vs shot k
fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
axes[0].semilogx(k, chosen_tau, '.', ms=3, color="C3", label="adaptive (Bayes gain)")
axes[0].set_ylabel(r"$\tau_k$ (s)"); axes[0].legend(fontsize=8, loc="upper left")
axes[1].semilogx(k, chosen_mF, '.', ms=3, color="C0", label="adaptive (Bayes gain)")
axes[1].set_ylabel(r"$m_{F,k}$"); axes[1].set_xlabel("k"); axes[1].set_yticks(range(-3, 4))
axes[0].set_title("Chosen control parameters")
plt.tight_layout()
save_plot("chosen_settings")
plt.show()

### How often each setting was chosen (counted per shot) as a histogram
freq = d["m_tot"] / d["m_tot"].sum()
mF_axis = 3 - np.arange(7)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].bar(mF_axis, freq.sum(0)); axes[0].set_xlabel(r"$m_F$"); axes[0].set_ylabel("fraction of shots")
axes[1].bar(taus, freq.sum(1), width=0.8 * (taus[1] - taus[0])); axes[1].set_xlabel(r"$\tau$ (s)")
im = axes[2].pcolormesh(taus, mF_axis, freq.T, shading="nearest", cmap="Greys")
axes[2].set_xlabel(r"$\tau$ (s)"); axes[2].set_ylabel(r"$m_F$"); plt.colorbar(im, ax=axes[2])
plt.tight_layout()
save_plot("setting_histograms")
plt.show()

### Plot snapshots of the info gain at a few decisions
js = [0, 1, 3, 10, len(K_maps) - 1]
fig, axes = plt.subplots(1, len(js), figsize=(4*len(js), 3.5), sharey=True, layout="constrained")
for ax, j in zip(axes, js):
    Kn = K_maps[j] / K_maps[j].max()
    im = ax.pcolormesh(taus, mF_axis, Kn.T, shading="nearest", vmin=0, vmax=1)
    bb, ii = np.unravel_index(np.argmax(K_maps[j]), K_maps[j].shape)
    ax.plot(taus[bb], 3 - ii, "rx", ms=10, mew=2)
    ax.set_title(f"shots {k[j] - M[j] + 1}–{k[j]}"); ax.set_xlabel(r"$\tau$ (s)")
axes[0].set_ylabel(r"$m_F$")
fig.colorbar(im, ax=axes, label=r"$K/K_{\max}$")
save_plot("info_gain_snapshots")
plt.show()

### Plot in 3D the (normalised) info gain for each tau and mF for a given shot.
j = len(K_maps)-1 # Choose final shot
TAU, MF = np.meshgrid(taus, mF_axis, indexing="ij") # both (Ntau, 7), same shape as K_maps[j]
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(TAU, MF, K_maps[j] / K_maps[j].max(), cmap="viridis", edgecolor="k", lw=0.2)
ax.set_xlabel(r"$\tau$ (s)"); ax.set_ylabel(r"$m_F$"); ax.set_zlabel(r"$K/K_{\max}$")
ax.view_init(elev=30, azim=-60)
save_plot("info_gain_3D")
plt.show()