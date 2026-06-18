"""
MCMC Parameter Fitting for GRN Model (Production Version)
============================================================
Strategy:
  1. Do NOT fit absolute expression (model scale ~0-15, biology ~0-1)
  2. Instead, fit log2FC perturbation responses — scale-invariant
  3. Parameters to fit: ~14 key regulation parameters (HES↔ASCL1 toggle)
  4. emcee sampler, 16 walkers, 500 steps burn-in, 1000 steps production

This is the method that would be reported in a paper's "Parameter Estimation" section.
"""

import os, sys, json, time, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from core.grn_model import (
    GENES, N_GENES, GENE2IDX,
    _BASAL_DEG, REGULATIONS,
    get_default_basal_and_deg, steady_state,
    hill_activate,
)

FIT_DIR = PROJ / "parameter_fitting"
OUTPUT_DIR = PROJ / "output"
FIT_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# GROUND TRUTH: 扰动 log2FC 数据 (from literature)
# ════════════════════════════════════════════════════════════════════════

# log2(fold_change) — positive = upregulated, negative = downregulated
# Sources: Andersen et al. 2021 (NOTCH1 KO), Imayoshi 2010 (HES1/5 KO),
#          Gao 2009 (ASCL1 KO), Zhang 2019 (CTNNB1 KO NSC RNA-seq)
PERT_TARGETS = {
    "NOTCH1_knock_out": {
        "NOTCH1": -8.0,  "HES1": -1.6,   "HES5": -3.0,
        "ASCL1":  +2.0,  "NEUROG2": +1.5, "DCX": +1.2,
        "TUBB3":  +0.8,  "RBFOX3": +1.0,  "NEUROD1": +1.0,
        "MBP":    +1.5,  "SOX2": 0.0,     "GFAP": +0.3,
        "CCND1":  +0.5,
    },
    "HES1_knock_out": {
        "HES1": -8.0,    "ASCL1":  +1.0,  "NEUROG2": +0.8,
        "DCX":   +0.3,   "CCND1":  +1.2,
    },
    "ASCL1_knock_out": {
        "ASCL1": -8.0,   "NEUROG2": -1.5, "DCX": -1.0,
        "TUBB3": -0.8,   "HES1":  +0.5,   "HES5": +0.5,
    },
    "CTNNB1_knock_out": {
        "CTNNB1": -8.0,  "MYC": -2.0,     "CCND1": -1.5,
        "NEUROG2": -1.0,
    },
}


# ════════════════════════════════════════════════════════════════════════
# PARAMETER INDEXING
# ════════════════════════════════════════════════════════════════════════

def _reg_idx(target, regulator):
    """Find index of regulation (target, regulator) in REGULATIONS list."""
    for i, (t, r, *_) in enumerate(REGULATIONS):
        if t == target and r == regulator:
            return i
    return None

# 12 key parameters for the HES↔ASCL1 toggle + key Wnt/BMP pathways
# Each = (short_name, target, reg_type, param_field, index_in_REGULATIONS)
KEY_PARAMS = [
    # HES → ASCL1 repression toggle
    ("r_H1_A_V",  "ASCL1", "HES1", "V"),   # rep HES1→ASCL1 V   (now 10.0)
    ("r_H1_A_K",  "ASCL1", "HES1", "K"),   # rep HES1→ASCL1 K   (now 1.0)
    ("r_H5_A_V",  "ASCL1", "HES5", "V"),   # rep HES5→ASCL1 V   (now 8.0)
    ("r_H5_A_K",  "ASCL1", "HES5", "K"),   # rep HES5→ASCL1 K   (now 0.8)
    # ASCL1 ← HES/NEUROD1 activation
    ("a_A_A_V",   "ASCL1", "ASCL1", "V"),  # ASCL1 auto-act V   (now 0.25)
    ("a_A_A_K",   "ASCL1", "ASCL1", "K"),  # ASCL1 auto-act K   (now 0.5)
    ("a_ND1_A_V", "ASCL1", "NEUROD1", "V"),# NEUROD1→ASCL1 V    (now 0.15)
    # HES ← NOTCH1 activation
    ("a_N1_H1_V", "HES1", "NOTCH1", "V"),  # NOTCH1→HES1 V     (now 0.80)
    ("a_N1_H1_K", "HES1", "NOTCH1", "K"),  # NOTCH1→HES1 K     (now 0.5)
    # ASCL1→NEUROG2
    ("a_A_NG2_V", "NEUROG2", "ASCL1", "V"),# ASCL1→NEUROG2 V   (now 0.50)
    ("a_A_NG2_K", "NEUROG2", "ASCL1", "K"),# ASCL1→NEUROG2 K   (now 0.4)
    # CTNNB1→MYC
    ("a_CB_MYC_V","MYC", "CTNNB1", "V"),   # CTNNB1→MYC V      (now 0.50)
    ("a_CB_MYC_K","MYC", "CTNNB1", "K"),   # CTNNB1→MYC K      (now 0.4)
    # NEUROG2→DCX
    ("a_NG2_DCX_V","DCX", "NEUROG2", "V"), # NEUROG2→DCX V     (now 0.40)
    ("a_NG2_DCX_K","DCX", "NEUROG2", "K"), # NEUROG2→DCX K     (now 0.4)
]

# Build param → REGULATIONS index mapping
_PARAM_MAP = {}
for name, target, reg, field in KEY_PARAMS:
    idx = _reg_idx(target, reg)
    if idx is not None:
        _PARAM_MAP[name] = (idx, field)
    else:
        print(f"⚠️ Warning: parameter {name} not found in REGULATIONS")


def param_vector_to_regs(x: np.ndarray):
    """Write parameter vector X into global REGULATIONS list."""
    for i, (name, _, _, _) in enumerate(KEY_PARAMS):
        idx, field = _PARAM_MAP.get(name, (None, None))
        if idx is not None:
            t, r, rt, V, K, n = REGULATIONS[idx]
            if field == "V":
                REGULATIONS[idx] = (t, r, rt, float(x[i]), K, n)
            elif field == "K":
                REGULATIONS[idx] = (t, r, rt, V, float(x[i]), n)


def get_param_vector() -> np.ndarray:
    """Read current parameter values from REGULATIONS."""
    vals = []
    for name, _, _, _ in KEY_PARAMS:
        idx, field = _PARAM_MAP.get(name, (None, None))
        if idx is not None:
            _, _, _, V, K, n = REGULATIONS[idx]
            vals.append(float(V if field == "V" else K))
        else:
            vals.append(1.0)
    return np.array(vals)


# ════════════════════════════════════════════════════════════════════════
# ODE WRAPPER: 缓存机制加速多次调用
# ════════════════════════════════════════════════════════════════════════

_ODE_CACHE = {}

def solve_grn(basal_scale: float = 1.0, deg_scale: float = 1.0) -> Dict[str, float]:
    """
    Solve GRN ODE with optional scaling of basal/deg rates.
    Returns dict of gene -> steady state expression.
    """
    cache_key = (round(basal_scale, 4), round(deg_scale, 4))
    if cache_key in _ODE_CACHE:
        return _ODE_CACHE[cache_key]

    basal, deg = get_default_basal_and_deg()
    if basal_scale != 1.0:
        basal = basal * basal_scale
    if deg_scale != 1.0:
        deg = deg * deg_scale

    _, _, y_final = steady_state(basal, deg)
    result = dict(zip(GENES, y_final))
    _ODE_CACHE[cache_key] = result
    return result


def compute_log2fc(gene: str, pert_type: str, ctrl: Dict,
                   param_x: np.ndarray = None) -> float:
    """
    Compute model log2FC for a perturbation vs control.
    If param_x is given, temporarily set REGULATIONS first.
    """
    # Save old state
    old_regs = list(REGULATIONS)
    if param_x is not None:
        param_vector_to_regs(param_x)

    # Run with perturbation
    from core.grn_model import apply_perturbation
    basal, deg = get_default_basal_and_deg()
    basal_p, deg_p, _ = apply_perturbation(gene, pert_type, basal, deg)

    # Rebuild regulation maps
    from core.grn_model import build_regulation_lookup
    new_act, new_rep = build_regulation_lookup(REGULATIONS)

    # Solve with perturbed params + regulation maps
    # Use the module-level maps as side effect (grn_module._ACT_MAP)
    import core.grn_model as gm
    old_act, old_rep = gm._ACT_MAP, gm._REP_MAP
    gm._ACT_MAP = new_act
    gm._REP_MAP = new_rep

    try:
        _, _, y_pert = steady_state(basal_p, deg_p)
        # Restore
        gm._ACT_MAP = old_act
        gm._REP_MAP = old_rep

        pert_dict = dict(zip(GENES, y_pert))
        eps = 1e-10
        c = max(ctrl.get(gene, 1e-10), 1e-10)
        p = max(pert_dict.get(gene, 1e-10), 1e-10)
        l2fc = np.log2(p / c)
    finally:
        # Restore original regulations
        for i in range(len(old_regs)):
            REGULATIONS[i] = old_regs[i]
        gm._ACT_MAP, gm._REP_MAP = build_regulation_lookup(REGULATIONS)

    return l2fc


# ════════════════════════════════════════════════════════════════════════
# LIKELIHOOD & MCMC
# ════════════════════════════════════════════════════════════════════════

def compute_likelihood(x: np.ndarray,
                       pert_targets: Dict[str, Dict],
                       sigma: float = 0.5) -> float:
    """
    Compute total log-likelihood across all perturbations.
    log L = -∑(model_l2fc - target_l2fc)² / (2σ²)

    Args:
        x: parameter vector
        pert_targets: {pert_name: {gene: l2fc_target}}
        sigma: observation noise in log2FC (default 0.5)

    Returns:
        log-likelihood (higher = better fit)
    """
    # Apply parameter vector
    param_vector_to_regs(x)

    # Get control expression
    ctrl = solve_grn()

    total_ll = 0.0

    for pert_name, gene_targets in pert_targets.items():
        gene, ptype = pert_name.split("_", 1)

        for tgene, tval in gene_targets.items():
            # Model log2FC
            mval = compute_log2fc_fast(tgene, ptype, ctrl, x)
            if mval is None:
                continue
            ll_gene = -0.5 * ((mval - tval) / sigma) ** 2
            total_ll += ll_gene

    return total_ll


def compute_log2fc_fast(gene: str, pert_type: str, ctrl: Dict) -> float:
    """
    Optimized single-gene log2FC computation.
    Uses REGULATIONS already set by caller.
    """
    from core.grn_model import apply_perturbation
    basal, deg = get_default_basal_and_deg()
    basal_p, deg_p, _ = apply_perturbation(gene, pert_type, basal, deg)

    _, _, y_pert = steady_state(basal_p, deg_p)
    idx = GENE2IDX.get(gene, 0)
    p = max(float(y_pert[idx]), 1e-10)
    c = max(ctrl.get(gene, 1e-10), 1e-10)
    return np.log2(p / c)


def log_posterior(x: np.ndarray, pert_targets: Dict, sigma: float,
                  prior_bounds: Dict) -> float:
    """Log-posterior = log_prior + log_likelihood."""
    # Prior: uniform in bounds
    for i, name in enumerate([p[0] for p in KEY_PARAMS]):
        lo, hi = prior_bounds.get(name, (1e-6, 100))
        if x[i] < lo or x[i] > hi:
            return -np.inf

    ll = compute_likelihood(x, pert_targets, sigma)
    return ll


# ════════════════════════════════════════════════════════════════════════
# MAIN: MCMC Runner
# ════════════════════════════════════════════════════════════════════════

def run_mcmc(n_walkers: int = 16, n_steps: int = 1000, n_burn: int = 300,
             sigma: float = 0.5) -> dict:
    """
    Run MCMC to fit GRN regulation parameters.

    Args:
        n_walkers: Number of emcee walkers
        n_steps: MCMC steps
        n_burn: Burn-in
        sigma: Observation noise

    Returns:
        dict with chain, best params, plots
    """
    import emcee

    ndim = len(KEY_PARAMS)
    param_names = [p[0] for p in KEY_PARAMS]
    x0 = get_param_vector()

    print(f"{'='*70}")
    print("MCMC Parameter Fitting — GRN Model")
    print(f"{'='*70}")
    print(f"Parameters: {ndim}")
    print(f"Walkers:    {n_walkers}  Steps: {n_steps}  Burn: {n_burn}")
    print(f"Sigma:      {sigma} (log2FC noise)")
    print(f"\nInitial parameters:")
    for name, val in zip(param_names, x0):
        print(f"  {name:>15s} = {val:.4f}")

    # Prior bounds
    prior_bounds = {}
    for name, val in zip(param_names, x0):
        if "_V" in name or "V" in name[-2:]:
            prior_bounds[name] = (max(0.001, val/5), min(50.0, val*5))
        else:
            prior_bounds[name] = (max(0.01, val/3), min(5.0, val*3))

    # Initial walkers
    initial_pos = np.array([
        x0 + 0.02 * np.random.randn(ndim) for _ in range(n_walkers)
    ])
    # Ensure within bounds
    for i in range(ndim):
        lo, hi = prior_bounds[param_names[i]]
        initial_pos[:, i] = np.clip(initial_pos[:, i], lo, hi)

    # Sampler
    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_posterior,
        args=(PERT_TARGETS, sigma, prior_bounds),
    )

    print(f"\nRunning MCMC...")
    t0 = time.time()
    sampler.run_mcmc(initial_pos, n_steps, progress=True)
    t1 = time.time()
    print(f"Done. {t1-t0:.0f}s ({n_steps*n_walkers:.0f} samples)")

    # Analysis
    flat = sampler.get_chain(discard=n_burn, flat=True)
    best = np.median(flat, axis=0)
    lower = np.percentile(flat, 2.5, axis=0)
    upper = np.percentile(flat, 97.5, axis=0)

    print(f"\n{'='*70}")
    print("Fitted Parameters (95% HDI)")
    print(f"{'Param':18s} {'Init':>8s} {'Median':>8s} {'95% CI':>22s} {'Δ%':>7s}")
    print("-"*70)
    for i, name in enumerate(param_names):
        iv = x0[i]; mv = best[i]; cl = lower[i]; ch = upper[i]
        dp = (mv - iv) / max(abs(iv), 1e-10) * 100
        print(f"  {name:16s} {iv:>8.3f} {mv:>8.3f}  "
              f"[{cl:>6.3f}, {ch:>6.3f}]  {dp:>+6.0f}%")
    print(f"{'='*70}")

    # Apply best params
    param_vector_to_regs(best)

    # Save
    result = {
        "best_params": best.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "param_names": param_names,
        "initial": x0.tolist(),
        "sigma": sigma,
        "n_walkers": n_walkers,
        "n_steps": n_steps,
        "n_burn": n_burn,
    }
    result_path = FIT_DIR / "mcmc_results.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved: {result_path}")

    # Save chain
    chain_path = FIT_DIR / "mcmc_chain.npz"
    np.savez_compressed(chain_path,
                        chain=sampler.get_chain(),
                        flat_chain=flat,
                        best_params=best,
                        lower=lower, upper=upper,
                        param_names=param_names)
    print(f"Chain saved: {chain_path}")

    return {
        "sampler": sampler,
        "flat_chain": flat,
        "best_params": best,
        "param_names": param_names,
    }


def plot_results(sampler, param_names, flat_chain):
    """Trace + corner plots."""
    try:
        import corner
        import matplotlib.pyplot as plt

        ndim = len(param_names)
        samples = sampler.get_chain()

        # Trace
        fig, axes = plt.subplots(ndim, 1, figsize=(10, 2*ndim))
        if ndim == 1:
            axes = [axes]
        for i, name in enumerate(param_names):
            ax = axes[i]
            ax.plot(samples[:, :, i], alpha=0.3, color="C0", lw=0.3)
            ax.set_ylabel(name, fontsize=7)
            if i == ndim - 1:
                ax.set_xlabel("Step", fontsize=8)
            ax.tick_params(labelsize=6)
        fig.suptitle("MCMC Trace", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        trace_path = str(FIT_DIR / "mcmc_trace.png")
        fig.savefig(trace_path, dpi=150)
        plt.close(fig)
        print(f"Trace: {trace_path}")

        # Corner
        fig2 = corner.corner(
            flat_chain, labels=param_names,
            quantiles=[0.025, 0.5, 0.975],
            show_titles=True, title_kwargs={"fontsize": 7},
            label_kwargs={"fontsize": 7},
            title_fmt=".3f",
        )
        corner_path = str(FIT_DIR / "mcmc_corner.png")
        fig2.savefig(corner_path, dpi=150)
        plt.close(fig2)
        print(f"Corner: {corner_path}")

    except Exception as e:
        print(f"Plotting failed: {e}")


# ════════════════════════════════════════════════════════════════════════
# TIER 2 ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

def run_mcmc_batch():
    """Run MCMC with multiple configurations."""
    # Fast mode: quick test
    print("Fast MCMC (16 walkers × 500 steps)")
    r = run_mcmc(n_walkers=16, n_steps=500, n_burn=200, sigma=0.5)

    import matplotlib
    matplotlib.use("Agg")
    plot_results(r["sampler"], r["param_names"], r["flat_chain"])
    return r


# ════════════════════════════════════════════════════════════════════════
# VALIDATION
# ════════════════════════════════════════════════════════════════════════

def validate_best_params(best_params=None):
    """Run perturbation simulations with best params and compare to targets."""
    if best_params is None:
        result_path = FIT_DIR / "mcmc_results.json"
        if result_path.exists():
            with open(result_path) as f:
                data = json.load(f)
            best_params = np.array(data["best_params"])
        else:
            print("No saved params found, using current REGULATIONS")
            best_params = get_param_vector()

    param_vector_to_regs(best_params)

    from core.grn_model import run_grn_simulation

    print(f"\n{'='*70}")
    print("VALIDATION: Model log2FC vs Literature Targets")
    print(f"{'='*70}")

    targets_flat = []
    for pert_name, gts in PERT_TARGETS.items():
        gene, pt = pert_name.split("_", 1)
        ctrl = solve_grn()
        mresult = run_grn_simulation(gene, pt)

        ce = ctrl
        pe = mresult["perturbed_expression"]
        eps = 1e-10

        print(f"\n--- {pert_name} ---")
        print(f"  {'Gene':>10s}  {'Model log2FC':>12s}  {'Target':>8s}  {'Δ':>8s}")
        for tgene, tval in sorted(gts.items()):
            if tgene in ce and pe and tgene in pe:
                mc = max(float(ce[tgene]), eps)
                mp = max(float(pe[tgene]), eps)
                ml2fc = np.log2(mp / mc)
                delta = ml2fc - tval
                mark = "✅" if abs(delta) < 0.5 else ("⚠️" if abs(delta) < 1.0 else "❌")
                print(f"  {tgene:>10s}  {ml2fc:>+12.3f}  {tval:>+8.1f}  "
                      f"{delta:>+8.2f}  {mark}")
                targets_flat.append((ml2fc, tval))

    # Overall error
    if targets_flat:
        model_vals, target_vals = zip(*targets_flat)
        mse = np.mean((np.array(model_vals) - np.array(target_vals))**2)
        rmse = np.sqrt(mse)
        from scipy.stats import pearsonr
        r, p = pearsonr(model_vals, target_vals)
        print(f"\n{'='*70}")
        print(f"Overall: RMSE={rmse:.3f} (log2FC)  Pearson r={r:.3f} (p={p:.2e})")
        print(f"{'='*70}")


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mcmc", "validate", "fast"],
                        default="fast")
    args = parser.parse_args()

    if args.mode == "validate":
        validate_best_params()
    else:
        r = run_mcmc_batch()
        if args.mode == "mcmc":
            validate_best_params(r["best_params"])
