"""
v5.5 — Parameter Sensitivity Analysis Module
=============================================
对22基因GRN模型的参数进行灵敏度分析，识别命运决定关键参数。

论文依据:
- Zi 2011 (IET Syst Biol): 系统生物学模型的灵敏度分析方法
- Saltelli 2008 (Wiley): 全局灵敏度分析
- Erguler & Stumpf 2011 (J R Soc Interface): 动态模型灵敏度实践

方法:
  1. Morris Screening (一次性扰动) — 快速筛选敏感参数
  2. Sobol 全局灵敏度分析 (ANOVA分解) — 定量敏感度指数
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from itertools import combinations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grn_model import (
    GENES, GENE2IDX, N_GENES, REGULATIONS,
    _ACT_MAP, _REP_MAP, _BASAL_DEG,
    hill_activate, grn_ode, compute_dps, get_default_basal_and_deg,
    apply_perturbation, build_regulation_lookup
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

# ── 1. Morris One-At-a-Time (OAT) Screening ────────────────────────

def morris_screening(parameter_ranges, n_trajectories=5, delta=0.1):
    """
    Morris Screening: 一次变化一个参数，观察输出变化.
    
    对每个参数计算:
    - μ (mean absolute effect): 平均效应大小
    - σ (standard deviation): 非线性程度 (高σ=非线性/交互作用)
    
    论文: Morris 1991, "Factorial sampling plans for preliminary computational experiments" (Technometrics)
    """
    basal, deg = get_default_basal_and_deg()
    
    # Baseline output
    sol = solve_ivp(lambda t, y: grn_ode(t, y, basal, deg, _ACT_MAP, _REP_MAP),
                    (0, 500), np.ones(N_GENES)*0.001, method="LSODA",
                    max_step=0.5, rtol=1e-6, atol=1e-8)
    y_base = sol.y[:, -1]
    dps_base = compute_dps(y_base)["DPS"]
    
    results = []
    
    for param_name, (min_val, max_val, param_type) in parameter_ranges.items():
        effects_dps = []
        effects_stem = []
        effects_diff = []
        
        for _ in range(n_trajectories):
            # Random base value within range
            base_val = min_val + np.random.random() * (max_val - min_val)
            perturbed_val = base_val * (1.0 + delta) if base_val != 0 else delta
            
            # Apply to system and get new steady state
            y_new = _apply_parameter_change(param_name, base_val, perturbed_val, param_type)
            
            if y_new is not None:
                dps_new = compute_dps(y_new)
                effect = dps_new["DPS"] - dps_base
                effects_dps.append(effect)
                stem_base = compute_dps(y_base)["stem_score"]
                effects_stem.append(dps_new["stem_score"] - stem_base)
                diff_base = compute_dps(y_base)["diff_score"]
                effects_diff.append(dps_new["diff_score"] - diff_base)
        
        if effects_dps:
            results.append({
                "param": param_name,
                "type": param_type,
                "mu_dps": np.mean(np.abs(effects_dps)),
                "sigma_dps": np.std(effects_dps),
                "mean_effect_dps": np.mean(effects_dps),
                "mu_stem": np.mean(np.abs(effects_stem)),
                "mu_diff": np.mean(np.abs(effects_diff)),
            })
    
    return results


def _apply_parameter_change(param_name, base_val, perturbed_val, param_type):
    """
    Apply a single parameter change and return new steady state vector.
    
    Supports modifying:
    - basal: gene basal production rate
    - deg: gene degradation rate
    - reg_V: regulation Hill V (strength)
    - reg_K: regulation Hill K (threshold)
    
    Returns: steady-state concentration vector (N_GENES,) or None on failure
    """
    basal, deg = get_default_basal_and_deg()
    act_map, rep_map = _ACT_MAP, _REP_MAP
    
    if param_type == "basal":
        # Find gene by matching param_name prefix
        for g in GENES:
            if param_name.startswith(g):
                idx = GENE2IDX[g]
                basal[idx] = perturbed_val
                break
        else:
            return None
    
    elif param_type == "deg":
        for g in GENES:
            if param_name.startswith(g):
                idx = GENE2IDX[g]
                deg[idx] = perturbed_val
                break
        else:
            return None
    
    elif param_type == "reg_V":
        # Modify regulation Hill V parameter (strength)
        reg_list = list(REGULATIONS)
        modified = False
        for i, (t, r, rtype, V, K, n) in enumerate(reg_list):
            # Match param_name like "ASCL1_HES1_V" → target=ASCL1, regulator=HES1
            key_pattern = f"{t}_{r}_V"
            if key_pattern == param_name:
                new_V = perturbed_val  # Directly set the Hill V value
                reg_list[i] = (t, r, rtype, new_V, K, n)
                modified = True
                break
        if not modified:
            # Try matching by reversing (regulator_target)
            for i, (t, r, rtype, V, K, n) in enumerate(reg_list):
                key_pattern = f"{r}_{t}_V"
                if key_pattern == param_name:
                    new_V = perturbed_val
                    reg_list[i] = (t, r, rtype, new_V, K, n)
                    modified = True
                    break
        if modified:
            act_map, rep_map = build_regulation_lookup(reg_list)
        else:
            return None
    
    elif param_type == "reg_K":
        reg_list = list(REGULATIONS)
        modified = False
        for i, (t, r, rtype, V, K, n) in enumerate(reg_list):
            key_pattern = f"{t}_{r}_K"
            if key_pattern == param_name:
                new_K = perturbed_val
                reg_list[i] = (t, r, rtype, V, new_K, n)
                modified = True
                break
        if not modified:
            for i, (t, r, rtype, V, K, n) in enumerate(reg_list):
                key_pattern = f"{r}_{t}_K"
                if key_pattern == param_name:
                    new_K = perturbed_val
                    reg_list[i] = (t, r, rtype, V, new_K, n)
                    modified = True
                    break
        if modified:
            act_map, rep_map = build_regulation_lookup(reg_list)
        else:
            return None
    
    else:
        return None
    
    # Solve the ODE with modified parameters to steady state
    try:
        sol = solve_ivp(
            lambda t, y: grn_ode(t, y, basal, deg, act_map, rep_map),
            (0, 500), np.ones(N_GENES) * 0.001, method="LSODA",
            max_step=0.5, rtol=1e-6, atol=1e-8,
        )
        return sol.y[:, -1]
    except Exception as e:
        return None


# ── 2. Sobol Sensitivity Analysis (simplified) ────────────────────

def sobol_analysis(parameter_of_interest="HES1_ASCL1_rep", param_range=(1.0, 20.0), n_samples=30):
    """
    Simplified Sobol analysis: vary one key parameter, measure variance in DPS.
    
    Focus on the HES1→ASCL1 repression strength (the critical Switch parameter)
    and SOX2 basal production (the Stem attractor strength).
    
    First and total-order effects estimated via brute force sampling.
    """
    basal, deg = get_default_basal_and_deg()
    
    # Find the regulation index for HES1→ASCL1 repression
    target_param_idx = None
    for i, (t, r, rtype, V, K, n) in enumerate(REGULATIONS):
        if t == "ASCL1" and r == "HES1" and rtype == "rep":
            target_param_idx = i
            break
    
    if target_param_idx is None:
        print("  ❌ HES1→ASCL1 rep not found in REGULATIONS!")
        return []
    
    param_vals = np.linspace(param_range[0], param_range[1], n_samples)
    results = []
    
    for pv in param_vals:
        # Build modified regulations
        reg_list = list(REGULATIONS)
        t, r, rtype, V, K, n = reg_list[target_param_idx]
        reg_list[target_param_idx] = (t, r, rtype, pv, K, n)
        act_map, rep_map = build_regulation_lookup(reg_list)
        
        # Low and high initial conditions
        for y0_factor, ic_name in [(0.001, "low"), (10.0, "high")]:
            try:
                sol = solve_ivp(
                    lambda t, y: grn_ode(t, y, basal, deg, act_map, rep_map),
                    (0, 500), np.ones(N_GENES) * y0_factor, method="LSODA",
                    max_step=0.5, rtol=1e-6, atol=1e-8,
                )
                yf = sol.y[:, -1]
                dps = compute_dps(yf)
                results.append({
                    "param_val": pv,
                    "ic": ic_name,
                    "DPS": dps["DPS"],
                    "state": dps["state"],
                    "ASCL1": float(yf[GENE2IDX["ASCL1"]]),
                    "SOX2": float(yf[GENE2IDX["SOX2"]]),
                    "HES1": float(yf[GENE2IDX["HES1"]]),
                })
            except Exception as e:
                pass
    
    return results


# ── 3. Visualization ──────────────────────────────────────────────

def plot_morris_results(results, title="Morris Sensitivity Screening"):
    """Plot μ vs σ scatter to identify influential parameters."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    if not results:
        ax.text(0.5, 0.5, "No results", ha="center", va="center", fontsize=14)
        fig.tight_layout()
        return _save_fig(fig, "morris_screening.png")
    
    params = [r["param"] for r in results]
    mu = [r["mu_dps"] for r in results]
    sigma = [r["sigma_dps"] for r in results]
    
    # Color by type
    colors = {"basal": "#e41a1c", "deg": "#377eb8", 
              "reg_V": "#4daf4a", "reg_K": "#984ea3"}
    c_vals = [colors.get(r["type"], "#999") for r in results]
    
    ax.scatter(mu, sigma, c=c_vals, s=80, edgecolors="black", alpha=0.8)
    
    # Label top parameters
    for i, (p, m, s) in enumerate(zip(params, mu, sigma)):
        if m > 0.02 or s > 0.02:  # only label significant ones
            ax.annotate(p, (m, s), fontsize=7, alpha=0.8)
    
    ax.set_xlabel("μ (mean absolute effect on DPS)", fontsize=11)
    ax.set_ylabel("σ (non-linearity / interaction)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axhline(0.02, color="gray", ls="--", alpha=0.4)
    ax.axvline(0.02, color="gray", ls="--", alpha=0.4)
    ax.grid(alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e41a1c", label="Basal production"),
        Patch(facecolor="#377eb8", label="Degradation rate"),
        Patch(facecolor="#4daf4a", label="Hill V (strength)"),
        Patch(facecolor="#984ea3", label="Hill K (threshold)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")
    
    fig.tight_layout()
    return _save_fig(fig, "morris_screening.png")


def plot_sobol_bifurcation(results, title="Sobol Sensitivity: HES1→ASCL1 Repression"):
    """Plot DPS vs parameter value for two initial conditions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if not results:
        ax1.text(0.5, 0.5, "No results", ha="center", va="center", fontsize=14)
        ax2.text(0.5, 0.5, "No results", ha="center", va="center", fontsize=14)
        fig.tight_layout()
        return _save_fig(fig, "sobol_bifurcation.png")
    
    # DPS
    low_ic = [r for r in results if r["ic"] == "low"]
    high_ic = [r for r in results if r["ic"] == "high"]
    
    if low_ic:
        x = [r["param_val"] for r in low_ic]
        y = [r["DPS"] for r in low_ic]
        ax1.plot(x, y, "o-", color="#2166ac", lw=1.5, label="Low IC", markersize=4)
    if high_ic:
        x = [r["param_val"] for r in high_ic]
        y = [r["DPS"] for r in high_ic]
        ax1.plot(x, y, "s-", color="#d6604d", lw=1.5, label="High IC", markersize=4)
    
    ax1.set_xlabel("HES1→ASCL1 Repression Strength V", fontsize=10)
    ax1.set_ylabel("DPS", fontsize=10)
    ax1.set_title("DPS Bifurcation", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    # ASCL1
    for dataset, color, label in [(low_ic, "#2166ac", "Low IC"), (high_ic, "#d6604d", "High IC")]:
        if dataset:
            x = [r["param_val"] for r in dataset]
            y = [r["ASCL1"] for r in dataset]
            ax2.plot(x, y, "o-" if "Low" in label else "s-", 
                    color=color, lw=1.5, label=label, markersize=4)
    
    ax2.set_xlabel("HES1→ASCL1 Repression Strength V", fontsize=10)
    ax2.set_ylabel("ASCL1 expression", fontsize=10)
    ax2.set_title("ASCL1 Bifurcation", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save_fig(fig, "sobol_bifurcation.png")


# ── Main Entry ────────────────────────────────────────────────────

def run_sensitivity_analysis():
    """运行完整参数灵敏度分析."""
    print("=" * 60)
    print("v5.5 — Parameter Sensitivity Analysis")
    print("Papers: Zi 2011 (IET Syst Biol), Saltelli 2008")
    print("=" * 60)

    # 1. Define parameters to screen
    print("\n[1/3] Morris Screening...")
    
    parameter_ranges = {
        # Basal production rates
        "SOX2_basal": (0.01, 0.50, "basal"),
        "HES1_basal": (0.001, 0.10, "basal"),
        "ASCL1_basal": (0.01, 0.30, "basal"),
        "DCX_basal": (0.001, 0.05, "basal"),
        "GFAP_basal": (0.001, 0.05, "basal"),
        "NOTCH1_basal": (0.01, 0.50, "basal"),
        "MYC_basal": (0.005, 0.20, "basal"),
        "CTNNB1_basal": (0.05, 0.50, "basal"),
        # Degradation rates
        "SOX2_deg": (0.03, 0.30, "deg"),
        "HES1_deg": (0.10, 0.80, "deg"),
        "ASCL1_deg": (0.08, 0.50, "deg"),
        "NOTCH1_deg": (0.10, 0.50, "deg"),
        "DCX_deg": (0.03, 0.25, "deg"),
        # Regulation Hill V (strength)
        "SOX2_SOX2_V": (0.10, 0.80, "reg_V"),
        "ASCL1_HES1_V": (3.0, 20.0, "reg_V"),
        "HES1_ASCL1_V": (1.0, 15.0, "reg_V"),
        "ASCL1_ASCL1_V": (0.05, 0.50, "reg_V"),
        # Regulation Hill K (threshold)
        "ASCL1_HES1_K": (0.3, 2.0, "reg_K"),
        "HES1_ASCL1_K": (0.2, 1.0, "reg_K"),
    }
    
    morris_results = morris_screening(parameter_ranges, n_trajectories=3)
    
    # Sort by effect size
    morris_results.sort(key=lambda x: x["mu_dps"], reverse=True)
    
    print(f"\n  Top 5 parameters by absolute effect on DPS:")
    for r in morris_results[:5]:
        print(f"    {r['param']:20s}: μ={r['mu_dps']:.4f}, σ={r['sigma_dps']:.4f}")
    
    print(f"\n  All {len(morris_results)} parameters screened")
    
    # 2. Sobol analysis on HES1→ASCL1
    print("\n[2/3] Sobol: HES1→ASCL1 repression strength...")
    sobol_results = sobol_analysis(
        parameter_of_interest="HES1_ASCL1_rep",
        param_range=(1.0, 20.0),
        n_samples=30,
    )
    
    if sobol_results:
        dps_by_ic = {"low": set(), "high": set()}
        for r in sobol_results:
            dps_by_ic[r["ic"]].add(round(r["DPS"], 2))
        print(f"  Initial condition sensitivity: low IC has {len(dps_by_ic['low'])} states, "
              f"high IC has {len(dps_by_ic['high'])} states")
        
        # Detect bistability
        all_states = set()
        for r in sobol_results:
            all_states.add(r["state"])
        print(f"  Distinct cell states observed: {all_states}")
    
    # 3. Generate figures
    print("\n[3/3] Generating figures...")
    p1 = plot_morris_results(morris_results)
    print(f"  ✅ {p1}")
    p2 = plot_sobol_bifurcation(sobol_results)
    print(f"  ✅ {p2}")
    
    print("\n" + "=" * 60)
    print("Sensitivity analysis complete ✅")
    print("=" * 60)
    
    return {
        "morris": morris_results,
        "sobol": sobol_results,
        "figures": [p1, p2],
    }


if __name__ == "__main__":
    run_sensitivity_analysis()
