"""
Neural Stem Cell Gene Regulatory Network (GRN) Model
=====================================================
A system of ODEs (Hill-function based) modeling 20+ key genes
in neural stem cell fate regulation.

Core gene categories:
  - Stemness: SOX2, NES, PROM1, TERT
  - Notch:    NOTCH1, HES1, HES5, ASCL1
  - Wnt:      CTNNB1, MYC, CCND1, NEUROG2
  - SHH:      GLI1, MYCN
  - BMP:      SMAD1, ID1, GFAP
  - Differentiation: DCX, TUBB3, RBFOX3, NEUROD1, MBP

Usage:
  from core.grn_model import run_grn_simulation
  run_grn_simulation()                        # baseline
  run_grn_simulation('NOTCH1', 'knock_out')   # KO
  run_grn_simulation('MYC', 'overexpress')     # OE
  run_grn_simulation('CTNNB1', 'drug_inhibit') # drug
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── Gene List ──────────────────────────────────────────────────────────────
GENES = [
    "SOX2", "NES", "PROM1", "TERT",
    "NOTCH1", "HES1", "HES5", "ASCL1",
    "CTNNB1", "MYC", "CCND1", "NEUROG2",
    "GLI1", "MYCN",
    "SMAD1", "ID1", "GFAP",
    "DCX", "TUBB3", "RBFOX3", "NEUROD1", "MBP",
]

N_GENES = len(GENES)
GENE2IDX = {g: i for i, g in enumerate(GENES)}

# ── Default parameters per gene ──────────────────────────────────────────
# basal, degradation_rate
_BASAL_DEG = {
    # Each gene: (basal_production_rate, degradation_rate)
    # Higher deg = faster turnover = more sensitive to regulation changes
    "SOX2":    (0.5,  0.5),
    "NES":     (0.1,  0.3),
    "PROM1":   (0.05, 0.3),
    "TERT":    (0.05, 0.3),
    "NOTCH1":  (0.4,  0.4),
    "HES1":    (0.05, 0.5),   # HES1: highly dependent on NOTCH1 signal
    "HES5":    (0.03, 0.5),   # HES5: highly dependent on NOTCH1 signal
    "ASCL1":   (0.1,  0.3),   # ASCL1: repressed by HES1, self-activating
    "CTNNB1":  (0.3,  0.3),
    "MYC":     (0.05, 0.4),
    "CCND1":   (0.03, 0.3),
    "NEUROG2": (0.02, 0.4),
    "GLI1":    (0.1,  0.3),
    "MYCN":    (0.05, 0.3),
    "SMAD1":   (0.2,  0.3),
    "ID1":     (0.03, 0.4),
    "GFAP":    (0.02, 0.2),
    "DCX":     (0.01, 0.3),
    "TUBB3":   (0.01, 0.3),
    "RBFOX3":  (0.005, 0.2),
    "NEUROD1": (0.01, 0.3),
    "MBP":     (0.005, 0.15),
}

REGULATIONS = [
    # SOX2 -> stemness
    ("NES",     "SOX2",  "act", 1.2, 0.5, 2),
    ("PROM1",   "SOX2",  "act", 1.0, 0.5, 2),
    ("TERT",    "SOX2",  "act", 0.8, 0.5, 2),

    # Notch: NOTCH1 -> HES1/HES5 (strong activation), HES repress ASCL1/NEUROG2 (strong)
    ("HES1",    "NOTCH1","act", 3.0, 0.4, 2),
    ("HES5",    "NOTCH1","act", 2.5, 0.4, 2),
    ("ASCL1",   "HES1",  "rep", 2.0, 0.3, 2),
    ("NEUROG2", "HES1",  "rep", 1.5, 0.3, 2),
    ("ASCL1",   "HES5",  "rep", 1.5, 0.3, 2),
    ("NEUROG2", "HES5",  "rep", 1.2, 0.3, 2),

    # ASCL1 -> neurogenesis
    ("NEUROG2", "ASCL1", "act", 1.5, 0.4, 2),
    ("DCX",     "ASCL1", "act", 1.2, 0.4, 2),
    ("TUBB3",   "ASCL1", "act", 1.0, 0.4, 2),

    # Cross-repression: (ASCL1 mildly represses HES)
    ("HES1",    "ASCL1","rep", 0.6, 0.6, 2),
    ("HES5",    "ASCL1","rep", 0.5, 0.6, 2),

    # Wnt: CTNNB1 -> MYC/CCND1/NEUROG2
    ("MYC",     "CTNNB1","act", 2.0, 0.5, 2),
    ("CCND1",   "CTNNB1","act", 1.5, 0.5, 2),
    ("CCND1",   "MYC",   "act", 1.0, 0.4, 2),
    ("NEUROG2", "CTNNB1","act", 1.0, 0.5, 2),
    ("TERT",    "MYC",   "act", 0.8, 0.4, 2),

    # SHH: GLI1 -> MYCN/CCND1
    ("MYCN",    "GLI1",  "act", 1.5, 0.4, 2),
    ("CCND1",   "GLI1",  "act", 1.0, 0.4, 2),

    # BMP: SMAD1 -> ID1/GFAP, ID1 represses neurogenesis
    ("ID1",     "SMAD1", "act", 1.5, 0.4, 2),
    ("GFAP",    "SMAD1", "act", 1.2, 0.4, 2),
    ("NEUROG2", "ID1",   "rep", 1.2, 0.4, 2),
    ("ASCL1",   "ID1",   "rep", 1.0, 0.4, 2),

    # Neuronal diff cascade
    ("DCX",     "NEUROG2","act", 1.5, 0.4, 2),
    ("TUBB3",   "NEUROG2","act", 1.2, 0.4, 2),
    ("RBFOX3",  "NEUROG2","act", 1.0, 0.4, 2),
    ("RBFOX3",  "NEUROD1","act", 1.2, 0.4, 2),
    ("NEUROD1", "ASCL1", "act",  1.5, 0.4, 2),
    ("DCX",     "NEUROD1","act", 1.0, 0.4, 2),
    ("MBP",     "NEUROD1","act", 1.2, 0.4, 2),

    # Cross-talk
    ("GFAP",    "NEUROG2","rep", 0.8, 0.5, 2),
    ("NOTCH1",  "ASCL1", "act",  0.5, 0.6, 2),
    ("CCND1",   "HES1",  "rep", 0.8, 0.4, 2),
    ("MYC",     "HES1",  "rep", 0.6, 0.4, 2),
    ("DCX",     "HES1",  "rep", 0.5, 0.6, 2),
    ("GFAP",    "DCX",   "rep", 0.6, 0.5, 2),
    ("SOX2",    "HES1",  "rep", 0.5, 0.6, 2),

    # SOX2 self-maintenance
    ("SOX2",    "SOX2",  "act", 0.5, 0.6, 2),
]


# ── Hill function helpers# ── Hill function helpers ────────────────────────────────────────────────

def hill_activate(x, V, K, n):
    """Hill activation: V * x^n / (K^n + x^n)"""
    # clip to avoid overflow
    if x <= 0:
        return 0.0
    ratio = (x / K) ** n
    return V * ratio / (1.0 + ratio)


def hill_repress(x, V, K, n):
    """Hill repression: V * K^n / (K^n + x^n) = V / (1 + (x/K)^n)"""
    if x <= 0:
        return V
    ratio = (x / K) ** n
    return V / (1.0 + ratio)


# ── ODE system ───────────────────────────────────────────────────────────

def build_regulation_lookup(regulations):
    """Build per-target lists of (regulator_idx, type, V, K, n)."""
    act_map = {g: [] for g in range(N_GENES)}
    rep_map = {g: [] for g in range(N_GENES)}
    for target, regulator, rtype, V, K, n in regulations:
        t_idx = GENE2IDX.get(target)
        r_idx = GENE2IDX.get(regulator)
        if t_idx is None:
            # try fuzzy match: some entries use 'ASC1' for 'ASCL1'
            for g, idx in GENE2IDX.items():
                if target in g or g in target:
                    t_idx = idx
                    break
        if t_idx is None or r_idx is None:
            continue
        if rtype == "act":
            act_map[t_idx].append((r_idx, V, K, n))
        else:
            rep_map[t_idx].append((r_idx, V, K, n))
    return act_map, rep_map


_ACT_MAP, _REP_MAP = build_regulation_lookup(REGULATIONS)


def grn_ode(t, y, basal_vec, deg_vec, act_map, rep_map):
    """Compute dydt for the GRN with amplified sensitivity."""
    dydt = np.zeros_like(y)

    for i in range(N_GENES):
        # Total basal + activation + repression
        act_total = 0.0
        rep_total = 0.0

        for r_idx, V, K, n in act_map[i]:
            act_total += hill_activate(y[r_idx], V, K, n)
        for r_idx, V, K, n in rep_map[i]:
            rep_total += hill_repress(y[r_idx], V, K, n)

        # Effective production = basal + activation - repression * multiplier
        # Repression reduces the net production rate
        net_prod = basal_vec[i] + act_total
        if rep_total > 0:
            net_prod *= (1.0 / (1.0 + rep_total))  # repression scales down production

        dydt[i] = net_prod - deg_vec[i] * y[i]

    return dydt


def get_default_basal_and_deg():
    basal = np.array([_BASAL_DEG[g][0] for g in GENES], dtype=float)
    deg   = np.array([_BASAL_DEG[g][1] for g in GENES], dtype=float)
    return basal, deg


def steady_state(basal, deg, t_span=(0, 1000), n_points=4000, method="LSODA"):
    """Integrate until steady state."""
    y0 = np.full(N_GENES, 0.01)
    sol = solve_ivp(
        lambda t, y: grn_ode(t, y, basal, deg, _ACT_MAP, _REP_MAP),
        t_span, y0, method=method,
        dense_output=True,
        max_step=1.0,
        rtol=1e-8, atol=1e-10,
    )
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    y_traj = sol.sol(t_eval)
    y_final = y_traj[:, -1]
    return y_traj, t_eval, y_final


# ── Perturbation ─────────────────────────────────────────────────────────

def apply_perturbation(gene_name, perturbation_type, basal, deg):
    """Modify basal/deg vectors for a given perturbation.

    Args:
        gene_name:         Gene symbol (str)
        perturbation_type: 'knock_out', 'overexpress', or 'drug_inhibit'
        basal, deg:        numpy arrays (will be copied)
    Returns:
        modified_basal, modified_deg, description
    """
    basal = basal.copy()
    deg   = deg.copy()
    idx   = GENE2IDX.get(gene_name)
    if idx is None:
        raise ValueError(f"Unknown gene: {gene_name}")

    if perturbation_type == "knock_out":
        basal[idx] = 0.0
        desc = f"{gene_name} knockout (basal → 0)"
    elif perturbation_type == "overexpress":
        basal[idx] *= 10.0
        desc = f"{gene_name} overexpression (basal × 10)"
    elif perturbation_type == "drug_inhibit":
        basal[idx] *= 0.2
        desc = f"{gene_name} drug inhibition (activity → 20%)"
    else:
        raise ValueError(f"Unknown perturbation: {perturbation_type}")
    return basal, deg, desc


# ── Plotting ─────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_heatmap(y_final_ctrl, y_final_pert, ctrl_label, pert_label):
    """Steady-state expression heatmap: control vs perturbed."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    data = np.vstack([y_final_ctrl, y_final_pert])
    vmin = min(data.min(), 0)
    vmax = max(data.max(), 1.5)

    for ax, vec, title in zip(axes, [y_final_ctrl, y_final_pert],
                               [ctrl_label, pert_label]):
        im = ax.imshow(vec.reshape(-1, 1), aspect="auto", cmap="viridis",
                       vmin=vmin, vmax=vmax)
        ax.set_yticks(range(N_GENES))
        ax.set_yticklabels(GENES, fontsize=8)
        ax.set_xticks([])
        ax.set_title(title, fontsize=10)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Expression level")
    fig.suptitle("Steady-State Expression Profiles", fontsize=12)
    return _save_fig(fig, "heatmap.png")


def plot_time_series(t_eval, y_traj):
    """Time series of all genes (control simulation)."""
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < N_GENES:
            ax.plot(t_eval, y_traj[i], lw=1.2)
            ax.set_title(GENES[i], fontsize=9)
            ax.set_xlabel("Time", fontsize=7)
            ax.set_ylabel("Expr.", fontsize=7)
            ax.tick_params(labelsize=6)
        else:
            ax.set_visible(False)
    fig.suptitle("Gene Expression Time Series (Control)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save_fig(fig, "time_series.png")


def plot_bar_comparison(y_final_ctrl, y_final_pert, ctrl_label, pert_label):
    """Bar chart comparing control vs perturbed steady-state."""
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(N_GENES)
    w = 0.35
    ax.bar(x - w/2, y_final_ctrl, w, label=ctrl_label, alpha=0.85)
    ax.bar(x + w/2, y_final_pert, w, label=pert_label, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(GENES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Steady-state expression", fontsize=10)
    ax.set_title(f"Perturbation Comparison: {ctrl_label} vs {pert_label}",
                 fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save_fig(fig, "bar_comparison.png")


def plot_differential_heatmap(y_final_ctrl, y_final_pert, pert_label):
    """Log2 fold-change heatmap (perturbation vs control)."""
    eps = 1e-10
    l2fc = np.log2((y_final_pert + eps) / (y_final_ctrl + eps))
    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(l2fc.reshape(-1, 1), aspect="auto", cmap="RdBu_r")
    ax.set_yticks(range(N_GENES))
    ax.set_yticklabels(GENES, fontsize=8)
    ax.set_xticks([])
    ax.set_title(f"Log2 Fold Change ({pert_label} vs Control)", fontsize=10)
    cbar = fig.colorbar(im, label="log2 FC")
    fig.tight_layout()
    return _save_fig(fig, "diff_heatmap.png")


# ── Main simulation entry-point ─────────────────────────────────────────

def run_grn_simulation(gene_name=None, perturbation_type=None):
    """Run GRN simulation and generate all plots.

    Args:
        gene_name:         Gene to perturb (None = baseline only)
        perturbation_type: 'knock_out', 'overexpress', or 'drug_inhibit'

    Returns:
        dict with paths to generated figures and final expression values.
    """
    print("=" * 60)
    print("Neural Stem Cell GRN Simulation")
    print("=" * 60)

    # ── Control simulation ──
    print("\n[1/4] Running control (baseline) simulation ...")
    basal_ctrl, deg_ctrl = get_default_basal_and_deg()
    y_traj_ctrl, t_eval, y_final_ctrl = steady_state(basal_ctrl, deg_ctrl)
    print(f"     Control steady state reached at t = {t_eval[-1]:.0f}")
    for g, val in zip(GENES, y_final_ctrl):
        print(f"     {g:>8s}: {val:.4f}")

    # ── Perturbation simulation ──
    if gene_name and perturbation_type:
        print(f"\n[2/4] Applying perturbation: {gene_name} ({perturbation_type}) ...")
        basal_pert, deg_pert, desc = apply_perturbation(
            gene_name, perturbation_type, basal_ctrl, deg_ctrl
        )
        y_traj_pert, _, y_final_pert = steady_state(basal_pert, deg_pert)
        print(f"     Perturbed steady state:")
        for g, val in zip(GENES, y_final_pert):
            print(f"     {g:>8s}: {val:.4f}")
    else:
        print("\n[2/4] No perturbation requested — running control only.")
        y_final_pert = y_final_ctrl
        desc = "Control"

    # ── Generate plots ──
    print(f"\n[3/4] Generating plots ...")
    ctrl_label = "Control"
    pert_label = desc if gene_name else ctrl_label

    heat_path     = plot_heatmap(y_final_ctrl, y_final_pert, ctrl_label, pert_label)
    time_path     = plot_time_series(t_eval, y_traj_ctrl)
    bar_path      = plot_bar_comparison(y_final_ctrl, y_final_pert, ctrl_label, pert_label)

    paths = {
        "heatmap":        heat_path,
        "time_series":    time_path,
        "bar_comparison": bar_path,
    }

    if gene_name:
        diff_path = plot_differential_heatmap(y_final_ctrl, y_final_pert, pert_label)
        paths["differential_heatmap"] = diff_path
        print(f"     Heatmap:             {heat_path}")
        print(f"     Diff heatmap:         {diff_path}")
    else:
        print(f"     Heatmap:             {heat_path}")

    print(f"     Time series:         {time_path}")
    print(f"     Bar comparison:      {bar_path}")

    # ── Summary ──
    print(f"\n[4/4] Simulation complete.")
    print(f"     Output directory: {OUTPUT_DIR}")
    if gene_name:
        print(f"     Perturbation:      {pert_label}")
        # Print top changed genes
        diff = y_final_pert - y_final_ctrl
        top_up = np.argsort(-diff)[:5]
        top_dn = np.argsort(diff)[:5]
        print(f"     Top upregulated:   {', '.join(GENES[i] for i in top_up)}")
        print(f"     Top downregulated: {', '.join(GENES[i] for i in top_dn)}")
    print("=" * 60)

    return {
        "control_expression": dict(zip(GENES, y_final_ctrl)),
        "perturbed_expression": dict(zip(GENES, y_final_pert)) if gene_name else None,
        "perturbation_description": desc if gene_name else None,
        "figure_paths": paths,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        run_grn_simulation(sys.argv[1], sys.argv[2])
    else:
        run_grn_simulation()
