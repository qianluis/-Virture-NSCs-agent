"""
Neural Stem Cell Gene Regulatory Network (GRN) Model
=====================================================
A system of ODEs modeling 22 key genes in NSC fate regulation.
Uses additive Hill-function regulation: each gene's expression
rate = basal_production + Σ activation_terms - Σ repression_terms.

All regulations use hill_activate(regulator, V, K, n) — the
increasing form — meaning higher regulator = stronger effect.

Pathways: Notch, Wnt, SHH, BMP, MAPK, Hippo
Core toggle: HES1/5 ↔ ASCL1 (proneural vs anti-neural)
Fate branching: SOX2(stemness) ↔ NEUROG2(neuronal) ↔ GFAP(glial)

Usage:
  from core.grn_model import run_grn_simulation
  run_grn_simulation()                              # baseline
  run_grn_simulation('NOTCH1', 'knock_out')         # KO
  run_grn_simulation('MYC',     'overexpress')      # OE
  run_grn_simulation('CTNNB1',  'drug_inhibit')     # drug
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── Gene List ─────────────────────────────────────────────────────────────
GENES = [
    "SOX2",   "NES",    "PROM1",  "TERT",          # Stemness
    "NOTCH1", "HES1",   "HES5",   "ASCL1",          # Notch / Proneural
    "CTNNB1", "MYC",    "CCND1",  "NEUROG2",        # Wnt / Proliferation
    "GLI1",   "MYCN",                                # SHH
    "SMAD1",  "ID1",    "GFAP",                      # BMP / Glial
    "DCX",    "TUBB3",  "RBFOX3", "NEUROD1", "MBP",  # Neuronal diff
]

N_GENES = len(GENES)
GENE2IDX = {g: i for i, g in enumerate(GENES)}

# ── Default production & degradation ──────────────────────────────────────
# (basal_production, degradation_rate)
# Design principle:
#   - Basal rate sets the "leak" expression without any regulator
#   - Degradation sets turnover speed (higher = faster response)
#   - Ratio basal/deg = steady-state without regulation
_BASAL_DEG = {
    # Stemness: moderate basal, slow degradation (stable)
    "SOX2":    (0.08, 0.12),
    "NES":     (0.02, 0.10),
    "PROM1":   (0.01, 0.10),
    "TERT":    (0.01, 0.10),

    # Notch pathway
    "NOTCH1":  (0.12, 0.25),  # Moderate basal, fast turnover
    "HES1":    (0.01, 0.35),  # Very low basal → depends on NOTCH1 signal
    "HES5":    (0.01, 0.35),  # Same as HES1
    "ASCL1":   (0.05, 0.25),  # Moderate basal, repressed by HES

    # Wnt pathway
    "CTNNB1":  (0.20, 0.18),  # High basal (constitutive activity)
    "MYC":     (0.03, 0.20),
    "CCND1":   (0.02, 0.15),
    "NEUROG2": (0.01, 0.15),  # Low basal, strongly regulated

    # SHH
    "GLI1":    (0.08, 0.15),
    "MYCN":    (0.02, 0.15),

    # BMP
    "SMAD1":   (0.15, 0.15),
    "ID1":     (0.01, 0.20),
    "GFAP":    (0.01, 0.08),  # Slow degradation → stable glial marker

    # Neuronal differentiation
    "DCX":     (0.005, 0.12),
    "TUBB3":   (0.005, 0.10),
    "RBFOX3":  (0.002, 0.08),
    "NEUROD1": (0.005, 0.15),
    "MBP":     (0.001, 0.06),
}

# ── Regulatory interactions ────────────────────────────────────────────────
# (target, regulator, type, V_max, K, n)
# All use hill_activate() — increasing function of regulator.
# For REPRESSION: higher regulator → larger hill_activate → MORE subtracted.
# For ACTIVATION: higher regulator → larger hill_activate → MORE added.
REGULATIONS = [
    # ── SOX2 → stemness self-maintenance ──
    ("SOX2",    "SOX2",  "act", 0.30, 0.5, 2),
    ("NES",     "SOX2",  "act", 0.25, 0.4, 2),
    ("PROM1",   "SOX2",  "act", 0.20, 0.4, 2),
    ("TERT",    "SOX2",  "act", 0.18, 0.4, 2),

    # ── Notch → HES (strong activation) ──
    ("HES1",    "NOTCH1","act", 0.80, 0.5, 2),
    ("HES5",    "NOTCH1","act", 0.60, 0.5, 2),

    # ── HES1/HES5 → ASCL1/NEUROG2 (strong repression) ──
    ("ASCL1",   "HES1",  "rep", 10.0, 1.0, 4),   # bistable switch   # n=3: sharp switch-like repression
    ("ASCL1",   "HES5",  "rep", 8.0,  0.8, 4),    # bistable switch
    ("NEUROG2", "HES1",  "rep", 10.0, 1.0, 4),   # bistable switch
    ("NEUROG2", "HES5",  "rep", 8.0,  0.8, 4),    # bistable switch

    # ── ASCL1 → neurogenesis cascade ──
    ("NEUROG2", "ASCL1", "act", 0.50, 0.4, 2),
    ("DCX",     "ASCL1", "act", 0.25, 0.4, 2),
    ("TUBB3",   "ASCL1", "act", 0.20, 0.4, 2),

    # ── Cross-repression: ASCL1 → HES1/5 (weak) ──
    ("HES1",    "ASCL1", "rep", 5.0,  0.4, 4),   # bistable switch
    ("HES5",    "ASCL1", "rep", 3.0,  0.4, 4),    # bistable switch

    # ── ASCL1 auto-activation ──
    ("ASCL1",   "ASCL1", "act", 0.25, 0.5, 2),

    # ── Wnt: CTNNB1 → MYC/CCND1 ──
    ("MYC",     "CTNNB1","act", 0.50, 0.4, 2),
    ("CCND1",   "CTNNB1","act", 0.40, 0.4, 2),
    ("CCND1",   "MYC",   "act", 0.20, 0.4, 2),
    ("TERT",    "MYC",   "act", 0.15, 0.4, 2),
    ("NEUROG2", "CTNNB1","act", 0.20, 0.5, 2),

    # ── SHH: GLI1 → MYCN/CCND1 ──
    ("MYCN",    "GLI1",  "act", 0.40, 0.4, 2),
    ("CCND1",   "GLI1",  "act", 0.25, 0.4, 2),

    # ── BMP: SMAD1 → ID1/GFAP ──
    ("ID1",     "SMAD1", "act", 0.35, 0.4, 2),
    ("GFAP",    "SMAD1", "act", 0.30, 0.5, 2),
    ("NEUROG2", "ID1",   "rep", 5.0,  0.6, 3),
    ("ASCL1",   "ID1",   "rep", 3.0,  0.6, 3),

    # ── NEUROG2 → differentiation cascade ──
    ("DCX",     "NEUROG2","act",0.40, 0.4, 2),
    ("TUBB3",   "NEUROG2","act",0.25, 0.4, 2),
    ("RBFOX3",  "NEUROG2","act",0.20, 0.5, 2),
    ("NEUROD1", "NEUROG2","act",0.25, 0.5, 2),
    ("DCX",     "NEUROD1","act",0.15, 0.4, 2),
    ("MBP",     "NEUROD1","act",0.25, 0.4, 2),

    # ── NEUROD1 amplifies ASCL1 (positive feedback) ──
    ("ASCL1",   "NEUROD1","act",0.15, 0.5, 2),
    ("RBFOX3",  "NEUROD1","act",0.15, 0.4, 2),

    # ── Cross-talk: glial vs neuronal ──
    ("NEUROG2", "GFAP",  "rep", 5.0,  0.6, 3),
    ("DCX",     "GFAP",  "rep", 3.0,  0.6, 3),

    # ── Proliferation control: HES1 represses CCND1 ──
    ("CCND1",   "HES1",  "rep", 5.0,  0.5, 3),

    # ── GFAP ↔ HES1 cross-rep ──
    ("GFAP",    "HES1",  "rep", 3.0,  0.6, 3),
    ("HES1",    "GFAP",  "act", 0.20, 0.6, 2),  # HES1 weakly promotes GFAP (stemness)
]


# ── Hill functions ────────────────────────────────────────────────────────

def hill_activate(x, V, K, n):
    """Hill activation: V * x^n / (K^n + x^n) — increasing function of x.
    V = max rate, K = half-max, n = cooperativity."""
    if x <= 0:
        return 0.0
    ratio = (x / K) ** n
    return V * ratio / (1.0 + ratio)


# ── ODE system ───────────────────────────────────────────────────────────

def build_regulation_lookup(regulations):
    """Build per-target lists of (reg_idx, V, K, n) for act and rep."""
    act_map = {g: [] for g in range(N_GENES)}
    rep_map = {g: [] for g in range(N_GENES)}
    for target, regulator, rtype, V, K, n in regulations:
        t_idx = GENE2IDX.get(target)
        r_idx = GENE2IDX.get(regulator)
        if t_idx is None or r_idx is None:
            continue
        if rtype == "act":
            act_map[t_idx].append((r_idx, V, K, n))
        else:
            rep_map[t_idx].append((r_idx, V, K, n))
    return act_map, rep_map


_ACT_MAP, _REP_MAP = build_regulation_lookup(REGULATIONS)


def grn_ode(t, y, basal_vec, deg_vec, act_map, rep_map):
    """
    Fold-change GRN ODE with multiplicative repression:

      production_i = (basal_i + Σ activators) × fold_repression_i
      fold_repression_i = 1 / (1 + Σ hill_activate(repressors))

      d[gene_i]/dt = production_i - deg_i × gene_i

    This multiplicative repression (fold-change) model produces
    realistic bistable switches (e.g. HES ↔ ASCL1 toggle),
    where a repressor at high concentration can suppress expression
    by orders of magnitude, while at low concentration it has
    negligible effect — accurately reflecting Notch-Delta lateral
    inhibition in biology.
    """
    dydt = np.zeros_like(y)
    for i in range(N_GENES):
        # Total activation (additive)
        act = 0.0
        for r_idx, V, K, n in act_map[i]:
            act += hill_activate(y[r_idx], V, K, n)

        # Total repression (as multiplicative fold-change)
        rep = 0.0
        for r_idx, V, K, n in rep_map[i]:
            rep += hill_activate(y[r_idx], V, K, n)

        # Fold-repression: rep=0 → factor=1 (no repression)
        # rep=1 → factor=0.5 (50% expression)
        # rep=10 → factor≈0.09 (90% repression)
        fold_rep = 1.0 / (1.0 + rep)

        prod = (basal_vec[i] + act) * fold_rep
        dydt[i] = prod - deg_vec[i] * y[i]

    return dydt


def get_default_basal_and_deg():
    basal = np.array([_BASAL_DEG[g][0] for g in GENES], dtype=float)
    deg   = np.array([_BASAL_DEG[g][1] for g in GENES], dtype=float)
    return basal, deg


# ── Integration ──────────────────────────────────────────────────────────

def steady_state(basal, deg, t_span=(0, 500), n_points=2000):
    """Integrate ODE system until steady state."""
    y0 = np.ones(N_GENES) * 0.001  # small initial values
    sol = solve_ivp(
        lambda t, y: grn_ode(t, y, basal, deg, _ACT_MAP, _REP_MAP),
        t_span, y0, method="LSODA",
        dense_output=True,
        max_step=0.5,
        rtol=1e-8, atol=1e-10,
    )
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    y_traj = sol.sol(t_eval)
    y_final = y_traj[:, -1]
    return y_traj, t_eval, y_final


# ── Perturbations ────────────────────────────────────────────────────────

def apply_perturbation(gene_name, perturbation_type, basal, deg):
    """Modify basal/deg vectors for a given perturbation."""
    basal = basal.copy()
    deg   = deg.copy()
    idx   = GENE2IDX.get(gene_name)
    if idx is None:
        raise ValueError(f"Unknown gene: {gene_name}")

    if perturbation_type == "knock_out":
        basal[idx] = 0.0
        deg[idx]   = deg[idx] * 3.0  # moderately increased degradation
        desc = f"{gene_name} knockout (basal→0, deg↑)"
    elif perturbation_type == "overexpress":
        basal[idx] *= 20.0
        desc = f"{gene_name} overexpression (basal×20)"
    elif perturbation_type == "drug_inhibit":
        basal[idx] *= 0.1
        desc = f"{gene_name} drug inhibition (basal→10%)"
    else:
        raise ValueError(f"Unknown perturbation: {perturbation_type}")
    return basal, deg, desc


# ── Plotting ──────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_heatmap(y_final_ctrl, y_final_pert, ctrl_label, pert_label):
    """Steady-state expression heatmap."""
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
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(N_GENES)
    w = 0.35
    ax.bar(x - w/2, y_final_ctrl, w, label=ctrl_label, alpha=0.85)
    ax.bar(x + w/2, y_final_pert, w, label=pert_label, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(GENES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Steady-state expression", fontsize=10)
    ax.set_title(f"Perturbation Comparison", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save_fig(fig, "bar_comparison.png")


def plot_differential_heatmap(y_final_ctrl, y_final_pert, pert_label):
    eps = 1e-10
    l2fc = np.log2((y_final_pert + eps) / (y_final_ctrl + eps))
    fig, ax = plt.subplots(figsize=(6, 8))
    colors = ["#053061", "#2166ac", "#4393c3", "#92c5de",
              "#f7f7f7", "#f7f7f7", "#f4a582", "#d6604d", "#b2182b", "#67001f"]
    im = ax.imshow(l2fc.reshape(-1, 1), aspect="auto", cmap="RdBu_r",
                   vmin=-2, vmax=2)
    ax.set_yticks(range(N_GENES))
    ax.set_yticklabels(GENES, fontsize=8)
    ax.set_xticks([])
    ax.set_title(f"Log2 FC ({pert_label} vs Control)", fontsize=10)
    cbar = fig.colorbar(im, label="log2 FC")
    fig.tight_layout()
    return _save_fig(fig, "diff_heatmap.png")


# ── Main entry point ─────────────────────────────────────────────────────

def run_grn_simulation(gene_name=None, perturbation_type=None):
    """
    Run GRN simulation and generate plots.

    Args:
        gene_name: Gene to perturb (None = baseline only)
        perturbation_type: 'knock_out' | 'overexpress' | 'drug_inhibit'

    Returns:
        dict with control_expression, perturbed_expression, figure_paths
    """
    print("=" * 60)
    print("Neural Stem Cell GRN Simulation (Additive ODE)")
    print("=" * 60)

    # Control
    print("\n[1/4] Running control (baseline) simulation ...")
    basal_ctrl, deg_ctrl = get_default_basal_and_deg()
    y_traj_ctrl, t_eval, y_final_ctrl = steady_state(basal_ctrl, deg_ctrl)
    print(f"     Control steady state at t={t_eval[-1]:.0f}")
    for g, val in zip(GENES, y_final_ctrl):
        print(f"     {g:>8s}: {val:.4f}")

    # Perturbation
    if gene_name and perturbation_type:
        print(f"\n[2/4] Perturbation: {gene_name} ({perturbation_type})")
        basal_pert, deg_pert, desc = apply_perturbation(
            gene_name, perturbation_type, basal_ctrl, deg_ctrl
        )
        y_traj_pert, _, y_final_pert = steady_state(basal_pert, deg_pert)
        print("     Perturbed steady state:")
        for g, val in zip(GENES, y_final_pert):
            print(f"     {g:>8s}: {val:.4f}")
    else:
        y_final_pert = y_final_ctrl
        desc = "Control"

    # Plots
    print(f"\n[3/4] Generating plots ...")
    ctrl_label = "Control"
    pert_label = desc if gene_name else ctrl_label

    heat_path = plot_heatmap(y_final_ctrl, y_final_pert, ctrl_label, pert_label)
    time_path = plot_time_series(t_eval, y_traj_ctrl)
    bar_path  = plot_bar_comparison(y_final_ctrl, y_final_pert, ctrl_label, pert_label)

    paths = {
        "heatmap":        heat_path,
        "time_series":    time_path,
        "bar_comparison": bar_path,
    }

    if gene_name:
        diff_path = plot_differential_heatmap(y_final_ctrl, y_final_pert, pert_label)
        paths["differential_heatmap"] = diff_path
        print(f"     Heatmap:       {heat_path}")
        print(f"     Diff heatmap:  {diff_path}")
    else:
        print(f"     Heatmap:       {heat_path}")
    print(f"     Time series:   {time_path}")
    print(f"     Bar comp:      {bar_path}")

    # Summary
    print(f"\n[4/4] Summary")
    print(f"     Output: {OUTPUT_DIR}")
    if gene_name:
        print(f"     Perturbation: {pert_label}")
        diff = y_final_pert - y_final_ctrl
        top_up = np.argsort(-diff)[:6]
        top_dn = np.argsort(diff)[:6]
        print(f"     ↑ upregulated:   {', '.join(GENES[i] for i in top_up)}")
        print(f"     ↓ downregulated: {', '.join(GENES[i] for i in top_dn)}")
    print("=" * 60)

    return {
        "control_expression":  dict(zip(GENES, y_final_ctrl)),
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
