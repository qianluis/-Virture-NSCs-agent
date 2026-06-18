"""
GRN Parameter Refinement & Benchmark
======================================
Practical approach (paper-quality without expensive MCMC):

1. Nelder-Mead refinement of 12 key toggle parameters
   (minimising log2FC MSE across 4 perturbations × ~15 genes)
2. Synthetic data validation: generate pseudo-data from known params,
   recover them → demonstrates identifiability
3. Full perturbation benchmark with publication-ready table
4. High-resolution volcano/toggle/heatmap figures
"""

import os, sys, json, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
FIT_DIR = PROJ / "parameter_fitting"

from core.grn_model import (
    GENES, N_GENES, GENE2IDX, REGULATIONS,
    get_default_basal_and_deg, steady_state,
    build_regulation_lookup,
)


# ════════════════════════════════════════════════════════════════════════
# GROUND TRUTH
# ════════════════════════════════════════════════════════════════════════

PERT_TARGETS = {
    "NOTCH1_knock_out": {
        "NOTCH1": -8.0,  "HES1": -1.6,   "HES5": -3.0,
        "ASCL1":  +2.0,  "NEUROG2": +1.5, "DCX": +1.2,
        "TUBB3":  +0.8,  "RBFOX3": +1.0,  "NEUROD1": +1.0,
        "MBP":    +1.5,  "CCND1":  +0.5,
    },
    "HES1_knock_out": {
        "HES1": -8.0,   "ASCL1":  +1.0, "NEUROG2": +0.8,
        "DCX":   +0.3,  "CCND1":  +1.2,
    },
    "ASCL1_knock_out": {
        "ASCL1": -8.0,  "NEUROG2": -1.5, "DCX": -1.0,
        "TUBB3": -0.8,  "HES1":  +0.5,   "HES5": +0.5,
    },
    "CTNNB1_knock_out": {
        "CTNNB1": -8.0, "MYC": -2.0,     "CCND1": -1.5,
        "NEUROG2": -1.0,
    },
}

# Define 12 key parameters for refinement
KEY_PARAMS = [
    ("r_H1_A_V",  "ASCL1", "HES1", "rep", "V", 10.0),
    ("r_H1_A_K",  "ASCL1", "HES1", "rep", "K", 1.0),
    ("r_H5_A_V",  "ASCL1", "HES5", "rep", "V", 8.0),
    ("r_H5_A_K",  "ASCL1", "HES5", "rep", "K", 0.8),
    ("a_A_A_V",   "ASCL1", "ASCL1", "act", "V", 0.25),
    ("a_A_A_K",   "ASCL1", "ASCL1", "act", "K", 0.5),
    ("a_N1_H1_V", "HES1", "NOTCH1", "act", "V", 0.80),
    ("a_N1_H1_K", "HES1", "NOTCH1", "act", "K", 0.5),
    ("a_A_NG_V",  "NEUROG2", "ASCL1", "act", "V", 0.50),
    ("a_CB_MC_V", "MYC", "CTNNB1", "act", "V", 0.50),
    ("a_CB_MC_K", "MYC", "CTNNB1", "act", "K", 0.4),
    ("a_NG_DX_V", "DCX", "NEUROG2", "act", "V", 0.40),
]


# ════════════════════════════════════════════════════════════════════════
# PARAM ↔ VECTOR IO
# ════════════════════════════════════════════════════════════════════════

def get_pvec() -> np.ndarray:
    """Read current param values from REGULATIONS."""
    vals = []
    for name, t, r, rt, field, default in KEY_PARAMS:
        found = False
        for reg in REGULATIONS:
            if reg[0] == t and reg[1] == r and reg[2] == rt:
                if field == "V":
                    vals.append(float(reg[3]))
                else:
                    vals.append(float(reg[4]))
                found = True
                break
        if not found:
            vals.append(float(default))
    return np.array(vals)


def set_pvec(x: np.ndarray):
    """Write param vector into REGULATIONS."""
    for i, (name, t, r, rt, field, default) in enumerate(KEY_PARAMS):
        for j, reg in enumerate(REGULATIONS):
            if reg[0] == t and reg[1] == r and reg[2] == rt:
                list_reg = list(reg)
                if field == "V":
                    list_reg[3] = float(x[i])
                else:
                    list_reg[4] = float(x[i])
                REGULATIONS[j] = tuple(list_reg)
                break

    # Rebuild regulation maps used by steady_state
    a_map, r_map = build_regulation_lookup(REGULATIONS)
    import core.grn_model as gm
    for k in range(N_GENES):
        gm._ACT_MAP[k] = a_map[k]
        gm._REP_MAP[k] = r_map[k]


# ════════════════════════════════════════════════════════════════════════
# ODE WRAPPER
# ════════════════════════════════════════════════════════════════════════

def solve_ctrl():
    b, d = get_default_basal_and_deg()
    _, _, y = steady_state(b, d)
    return dict(zip(GENES, y))


def solve_pert(gene: str, ptype: str):
    from core.grn_model import apply_perturbation
    b, d = get_default_basal_and_deg()
    bp, dp, _ = apply_perturbation(gene, ptype, b, d)
    _, _, y = steady_state(bp, dp)
    return dict(zip(GENES, y))


def get_log2fc(gene: str, ptype: str, ctrl=None) -> Dict[str, float]:
    """Compute log2(perturbed/control) for all genes."""
    if ctrl is None:
        ctrl = solve_ctrl()
    pert = solve_pert(gene, ptype)
    eps = 1e-10
    l2fc = {}
    for g in GENES:
        c = max(ctrl.get(g, eps), eps)
        p = max(pert.get(g, eps), eps)
        l2fc[g] = np.log2(p / c)
    return l2fc


# ════════════════════════════════════════════════════════════════════════
# OPTIMIZATION
# ════════════════════════════════════════════════════════════════════════

def obj_func(x: np.ndarray, sigma: float = 0.5) -> float:
    """MSE loss between model log2FC and targets."""
    set_pvec(x)
    ctrl = solve_ctrl()
    total = 0.0
    for pert_name, gts in PERT_TARGETS.items():
        gene, ptype = pert_name.split("_", 1)
        l2fc = get_log2fc(gene, ptype, ctrl)
        for tgene, tval in gts.items():
            if tgene in l2fc:
                total += ((l2fc[tgene] - tval) / sigma) ** 2
    return total


def refine_params(x0: Optional[np.ndarray] = None,
                  maxiter: int = 500) -> np.ndarray:
    """Nelder-Mead refinement of key parameters."""
    if x0 is None:
        x0 = get_pvec()

    print(f"{'='*70}")
    print("Parameter Refinement (Nelder-Mead)")
    print(f"{'='*70}")
    print(f"Parameters: {len(KEY_PARAMS)}")
    print(f"Targets: {sum(len(v) for v in PERT_TARGETS.values())} gene×pert data points")
    print(f"Initial param vector: {np.round(x0, 4)}\n")

    # Initial loss
    l0 = obj_func(x0)
    print(f"Initial χ² loss: {l0:.4f}")

    # Bounds
    bounds = []
    for i, (name, t, r, rt, field, default) in enumerate(KEY_PARAMS):
        iv = x0[i]
        if "_V" in name or "V" in name:
            bounds.append((max(0.001, iv/5), min(50, iv*5)))
        else:
            bounds.append((max(0.01, iv/3), min(5.0, iv*3)))

    t0 = time.time()
    res = minimize(obj_func, x0, method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-3})
    t1 = time.time()

    x_best = res.x
    set_pvec(x_best)
    l_best = obj_func(x_best)

    print(f"\nNelder-Mead: {t1-t0:.1f}s, nfev={res.nfev}")
    print(f"Final χ² loss: {l_best:.4f} (Δ={l_best-l0:+.4f})")

    print(f"\n{'Param':18s} {'Initial':>8s} {'Final':>8s} {'Δ%':>8s}")
    print("-"*50)
    for i, (name, *_) in enumerate(KEY_PARAMS):
        iv = x0[i]; fv = x_best[i]
        dp = (fv - iv) / max(abs(iv), 1e-10) * 100
        print(f"  {name:16s} {iv:>8.4f} {fv:>8.4f} {dp:>+7.0f}%")

    # Save
    result = {
        "initial": x0.tolist(),
        "final": x_best.tolist(),
        "chi2_initial": l0,
        "chi2_final": l_best,
        "params": [p[0] for p in KEY_PARAMS],
    }
    with open(FIT_DIR / "refinement_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {FIT_DIR / 'refinement_result.json'}")

    return x_best


# ════════════════════════════════════════════════════════════════════════
# SYNTHETIC VALIDATION
# ════════════════════════════════════════════════════════════════════════

def validate_identifiability(noise_sigma: float = 0.1,
                             n_trials: int = 10) -> Dict:
    """
    Generate synthetic log2FC data from perturbed true parameters,
    then attempt to recover them via optimization.

    If recovery error < 20% → parameters are identifiable.
    This is a key check for paper credibility.
    """
    print(f"\n{'='*70}")
    print("SYNTHETIC DATA VALIDATION")
    print(f"{'='*70}")

    x_true = get_pvec().copy()
    recovery_errors = []

    for trial in range(n_trials):
        # Create "true" parameters by jittering current ones
        x_perturbed = x_true * (1.0 + 0.3 * np.random.randn(len(x_true)))
        x_perturbed = np.clip(x_perturbed, 0.01, 50.0)
        set_pvec(x_perturbed)

        # Generate "observed" log2FC (with noise)
        ctrl = solve_ctrl()
        observations = {}
        for pert_name, gts in PERT_TARGETS.items():
            gene, ptype = pert_name.split("_", 1)
            l2fc = get_log2fc(gene, ptype, ctrl)
            for tgene in gts:
                obs = l2fc[tgene] + noise_sigma * np.random.randn()
                if pert_name not in observations:
                    observations[pert_name] = {}
                observations[pert_name][tgene] = obs

        # Replace targets with these synthetic observations
        global PERT_TARGETS_SYNTH
        PERT_TARGETS_SYNTH = observations

        # Try to recover using different random starting point
        x_start = x_perturbed * (1.0 + 0.5 * np.random.randn(len(x_perturbed)))
        x_start = np.clip(x_start, 0.01, 50.0)

        def obj_synth(x):
            set_pvec(x)
            ctrl = solve_ctrl()
            total = 0.0
            for pn, gts in PERT_TARGETS_SYNTH.items():
                gene, ptype = pn.split("_", 1)
                l2fc = get_log2fc(gene, ptype, ctrl)
                for tg, tv in gts.items():
                    if tg in l2fc:
                        total += ((l2fc[tg] - tv) / noise_sigma) ** 2
            return total

        res = minimize(obj_synth, x_start, method="Nelder-Mead",
                       options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-2})

        x_recovered = res.x
        # Relative recovery error
        errs = np.abs(x_recovered - x_perturbed) / np.maximum(np.abs(x_perturbed), 1e-6)
        mean_err = np.mean(errs)
        recovery_errors.append(mean_err)
        print(f"  Trial {trial+1:2d}: mean recovery error = {mean_err:.2%}")

    # Restore original targets
    globals().pop("PERT_TARGETS_SYNTH", None)

    avg_err = np.mean(recovery_errors)
    std_err = np.std(recovery_errors)
    print(f"\n  → Average recovery error: {avg_err:.2%} ± {std_err:.2%}")
    if avg_err < 0.20:
        print(f"  ✅ Parameters ARE identifiable (< 20% error)")
    else:
        print(f"  ⚠️ Parameter identifiability marginal (≥ 20% error)")

    # Restore original params
    set_pvec(x_true)
    return {"mean_error": float(avg_err), "std_error": float(std_err)}


# ════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ════════════════════════════════════════════════════════════════════════

def run_benchmark(save: bool = True) -> Dict:
    """
    Run systematic benchmark: all 4 perturbations across all 22 genes.
    Returns dict and saves benchmark.json
    """
    print(f"\n{'='*70}")
    print("SYSTEMATIC PERTURBATION BENCHMARK")
    print(f"{'='*70}")

    ctrl = solve_ctrl()
    results = {"control": {g: round(float(ctrl[g]), 4) for g in GENES}}

    all_perts = [
        ("NOTCH1", "knock_out"),
        ("HES1", "knock_out"),
        ("ASCL1", "knock_out"),
        ("CTNNB1", "knock_out"),
        ("SOX2", "knock_out"),
        ("SMAD1", "knock_out"),
        ("NOTCH1", "overexpress"),
        ("MYC", "overexpress"),
        ("ASCL1", "overexpress"),
        ("CTNNB1", "drug_inhibit"),
    ]

    for gene, ptype in all_perts:
        l2fc = get_log2fc(gene, ptype, ctrl)
        key = f"{gene}_{ptype}"
        results[key] = {g: round(float(l2fc[g]), 4) for g in GENES}
        print(f"  {key:25s} done")

    if save:
        with open(FIT_DIR / "benchmark.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {FIT_DIR / 'benchmark.json'}")

    return results


def print_benchmark_table(results: Dict):
    """Pretty-print benchmark as markdown table."""
    perts = [k for k in results if k != "control"]
    print("\n## Benchmark Results (log2 fold-change)\n")
    header = "| Gene | " + " | ".join(p.replace("_"," ") for p in perts) + " |"
    sep = "|-----|" + "|".join("------" for _ in perts) + "|"
    print(header)
    print(sep)
    for g in GENES:
        row = f"| {g} "
        for p in perts:
            v = results[p].get(g, 0)
            icon = "↑" if v > 0.5 else ("↓" if v < -0.5 else "·")
            row += f"| {v:>+5.2f}{icon} "
        print(row + "|")


def plot_benchmark_heatmap(results: Dict):
    """Publication-quality heatmap of all perturbations × genes."""
    perts = [k for k in results if k != "control"]
    data = np.zeros((len(GENES), len(perts)))
    for j, p in enumerate(perts):
        for i, g in enumerate(GENES):
            data[i, j] = results[p].get(g, 0)

    fig, ax = plt.subplots(figsize=(max(10, len(perts)*1.5), 10))
    vmax = max(abs(data.min()), abs(data.max()), 1.0)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    # Labels
    ax.set_yticks(range(len(GENES)))
    ax.set_yticklabels(GENES, fontsize=9)
    pert_labels = [p.replace("_", "\n") for p in perts]
    ax.set_xticks(range(len(perts)))
    ax.set_xticklabels(pert_labels, fontsize=7, rotation=45, ha="right")

    # Cell values
    for i in range(len(GENES)):
        for j in range(len(perts)):
            v = data[i, j]
            c = "white" if abs(v) > vmax * 0.6 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=5.5, color=c, fontweight="bold")

    ax.set_title("log2(Fold Change) — GRN Perturbation Benchmark", fontsize=13)
    plt.colorbar(im, label="log2 fold change", shrink=0.6)
    fig.tight_layout()
    path = str(FIT_DIR / "benchmark_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Benchmark heatmap: {path}")


def plot_toggle_diagram():
    """HES↔ASCL1 bistable toggle phase diagram."""
    from core.grn_model import hill_activate
    import core.grn_model as gm

    # Get current regulations for the toggle
    v_h1a = [r[3] for r in REGULATIONS if r[0]=="ASCL1" and r[1]=="HES1"][0]
    k_h1a = [r[4] for r in REGULATIONS if r[0]=="ASCL1" and r[1]=="HES1"][0]
    n_h1a = [r[5] for r in REGULATIONS if r[0]=="ASCL1" and r[1]=="HES1"][0]
    v_aa  = [r[3] for r in REGULATIONS if r[0]=="ASCL1" and r[1]=="ASCL1"][0]
    k_aa  = [r[4] for r in REGULATIONS if r[0]=="ASCL1" and r[1]=="ASCL1"][0]

    # Compute effective ASCL1 production as function of HES1
    hes_range = np.logspace(-2, 1.5, 200)
    ascl1_ss = []

    for hes in hes_range:
        # Fold-repression: fold_rep = 1/(1 + hill(hes, V, K, n))
        rep = hill_activate(hes, v_h1a, k_h1a, n_h1a)
        fold_rep = 1.0 / (1.0 + rep)

        # ASCL1 steady state (self-activation + fold rep) / deg
        # At SS: (basal + act) * fold_rep = deg * ascl1
        # ascl1 ≈ basal_ascl1 * fold_rep / deg_ascl1 (ignoring self-act)
        basal_a = 0.05
        deg_a = 0.25
        ascl1_hes_dominated = (basal_a * fold_rep) / deg_a

        # With self-activation, need iteration
        # Simple approximation: ascl1 ≈ (basal + v_aa*ascl1^n/(k^n+ascl1^n)) * fold_rep / deg
        ascl1 = 0.001
        for _ in range(50):
            act = hill_activate(ascl1, v_aa, k_aa, 2)
            ascl1 = (basal_a + act) * fold_rep / deg_a
        ascl1_ss.append(ascl1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: ASCL1 expression vs HES1
    ax1.semilogx(hes_range, ascl1_ss, "b-", lw=2)
    ax1.axvline(1.5, color="gray", ls="--", alpha=0.5, label="Ctrl HES1≈1.5")
    ax1.axvline(0.5, color="red", ls="--", alpha=0.5, label="KO HES1≈0.5")
    ax1.set_xlabel("HES1 expression level", fontsize=11)
    ax1.set_ylabel("ASCL1 steady-state expression", fontsize=11)
    ax1.set_title("HES1 → ASCL1 Toggle Response", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Right: fold_rep function (biological switch characterisation)
    rep_vals = np.logspace(-2, 2, 200)
    fold_rep_vals = 1 / (1 + rep_vals)
    ax2.semilogx(rep_vals, fold_rep_vals, "m-", lw=2)
    ax2.axhline(0.5, color="gray", ls="--", alpha=0.4)
    ax2.set_xlabel("Total repression (Σ hill_rep)", fontsize=11)
    ax2.set_ylabel("Fold-repression factor", fontsize=11)
    ax2.set_title("Fold-Change Repression Function\n$f(r) = 1/(1+r)$", fontsize=12)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = str(FIT_DIR / "toggle_diagram.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Toggle diagram: {path}")


# ════════════════════════════════════════════════════════════════════════
# PIPELINE
# ════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--refine", action="store_true", help="Run Nelder-Mead refinement")
    parser.add_argument("--validate", action="store_true", help="Synthetic data validation")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--all", action="store_true", help="Run everything sequentially")
    args = parser.parse_args()

    if not any(vars(args).values()):
        args.all = True

    if args.all or args.refine:
        x_best = refine_params(maxiter=300)

    if args.all or args.validate:
        # Quick synthetic validation (fewer trials)
        validate_identifiability(noise_sigma=0.2, n_trials=5)

    if args.all or args.benchmark:
        results = run_benchmark(save=True)

    if args.all or args.plot:
        results = run_benchmark(save=False)
        plot_benchmark_heatmap(results)
        plot_toggle_diagram()
        print_benchmark_table(results)

    if args.all:
        print(f"\n{'='*70}")
        print("All outputs in:", FIT_DIR)
        for f in sorted(FIT_DIR.iterdir()):
            if f.is_file():
                print(f"  {f.name}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
