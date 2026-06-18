"""
v5.0 — Waddington Epigenetic Landscape Module
=============================================
基于高层次论文的表观遗传景观定量建模.

论文依据:
- Bhatt et al. 2020, "Single-cell analysis of DNA methylation and transcription"
  (Nat Rev Genet) — 单细胞状态的势能景观表征
- Wang et al. 2022, "From Waddington's epigenetic landscape to single-cell data"
  (WIRES Systems Biology) — 景观的定量势能函数
- Ferrell 2012, "Bistability, bifurcations, and Waddington's landscape"
  (Current Biology) — 分岔产生新吸引子 = 发育开关

核心: 景观在形态发生素梯度下动态重塑
- 无信号: 单山谷 (干细胞稳态)
- 低梯度: 新山谷出现 (命运可塑)
- 高梯度: 深分成山谷 (细胞退出干细胞池)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grn_model import (
    GENES, GENE2IDX, N_GENES, REGULATIONS,
    _ACT_MAP, _REP_MAP, _BASAL_DEG,
    hill_activate, compute_production, grn_ode,
    compute_dps, get_default_basal_and_deg, steady_state,
    build_regulation_lookup
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Morphogen-Modulated Vector Field ──────────────────────────────────

def morphogen_modulated_ode(t, y, basal, deg, act_map, rep_map, 
                             shh_level=0.0, bmp_level=0.0, wnt_level=0.0):
    """
    GRN + 形态发生素 modulation.
    
    Ribes & Briscoe 2009 (Nat Rev Neurosci):
    SHH: 激活 GLI1, 促进 NSC 自我更新
    BMP: 激活 SMAD1/ID1, 促进星形胶质分化
    Wnt: 激活 CTNNB1, 促进增殖
    
    公式: 每个信号的效应以 Hill 函数叠加到对应基因的 basal expression
    """
    basal_mod = basal.copy()
    deg_mod = deg.copy()
    
    # SHH → GLI1 activation (stronger coupling — Ribes & Briscoe 2009)
    if shh_level > 0:
        gli_idx = GENE2IDX.get("GLI1")
        if gli_idx is not None:
            shh_effect = 5.0 * shh_level / (shh_level + 0.8)  # strong Hill
            basal_mod[gli_idx] *= (1.0 + shh_effect)
    
    # BMP → repress SOX2, activate SMAD1/ID1/GFAP (astroglial fate)
    if bmp_level > 0:
        smad_idx = GENE2IDX.get("SMAD1")
        id1_idx = GENE2IDX.get("ID1")
        gfap_idx = GENE2IDX.get("GFAP")
        sox_idx = GENE2IDX.get("SOX2")
        bmp_effect = 3.0 * bmp_level / (bmp_level + 0.5)
        if smad_idx is not None:
            basal_mod[smad_idx] *= (1.0 + bmp_effect)
        if id1_idx is not None:
            basal_mod[id1_idx] *= (1.0 + bmp_effect)
        if gfap_idx is not None:
            basal_mod[gfap_idx] *= (1.0 + bmp_effect)
        if sox_idx is not None:
            deg_mod[sox_idx] *= (1.0 + bmp_effect)  # BMP degrades SOX2
    
    # Wnt → CTNNB1/MYC/CCND1 activation
    if wnt_level > 0:
        ctn_idx = GENE2IDX.get("CTNNB1")
        myc_idx = GENE2IDX.get("MYC")
        ccnd_idx = GENE2IDX.get("CCND1")
        wnt_effect = 3.0 * wnt_level / (wnt_level + 0.6)
        if ctn_idx is not None:
            basal_mod[ctn_idx] *= (1.0 + wnt_effect)
        if myc_idx is not None:
            basal_mod[myc_idx] *= (1.0 + wnt_effect)
        if ccnd_idx is not None:
            basal_mod[ccnd_idx] *= (1.0 + wnt_effect)
    
    return grn_ode(t, y, basal_mod, deg_mod, act_map, rep_map)


def compute_landscape(morphogen_profile={"shh": 0.0, "bmp": 0.0, "wnt": 0.0},
                      resolution=40, sox2_range=(0, 5), dcx_range=(0, 3)):
    """
    在指定形态发生素条件下计算SOX2×DCX景观.
    
    Wang 2022:
    形态发生素梯度 = 景观的外部"力场" — 重塑势能面
    高SHH → 干细胞山谷加深
    高BMP → 分化山谷出现
    """
    basal, deg = get_default_basal_and_deg()
    _, _, y_ss = steady_state(basal, deg)
    
    s_idx = GENE2IDX["SOX2"]
    d_idx = GENE2IDX["DCX"]
    
    s_vals = np.linspace(sox2_range[0], sox2_range[1], resolution)
    d_vals = np.linspace(dcx_range[0], dcx_range[1], resolution)
    S, D = np.meshgrid(s_vals, d_vals, indexing="ij")
    
    dS = np.zeros_like(S)
    dD = np.zeros_like(D)
    
    for i in range(resolution):
        for j in range(resolution):
            y = y_ss.copy()
            y[s_idx] = S[i, j]
            y[d_idx] = D[i, j]
            
            dydt = morphogen_modulated_ode(
                0, y, basal, deg, _ACT_MAP, _REP_MAP,
                shh_level=morphogen_profile.get("shh", 0.0),
                bmp_level=morphogen_profile.get("bmp", 0.0),
                wnt_level=morphogen_profile.get("wnt", 0.0),
            )
            dS[i, j] = dydt[s_idx]
            dD[i, j] = dydt[d_idx]
    
    # Quasi-potential Φ = -∫F·dl (numerical line integral)
    phi = np.zeros_like(S)
    for i in range(1, resolution):
        dx = S[i, 0] - S[i-1, 0]
        phi[i, 0] = phi[i-1, 0] - dS[i-1, 0] * dx
    for i in range(resolution):
        for j in range(1, resolution):
            dy = D[i, j] - D[i, j-1]
            phi[i, j] = phi[i, j-1] - dD[i, j-1] * dy
    
    phi = gaussian_filter(phi, sigma=1.0)
    phi -= phi.min()
    phi_max = phi.max()
    if phi_max > 0:
        phi /= phi_max
    
    return S, D, dS, dD, phi


def find_attractors(S, D, phi, morphogen_profile=None, n_trials=5):
    """
    检测势能面中的吸引子.
    
    用多个初始条件积分ODE到收敛, 聚类分析.
    """
    np.random.seed(42)
    basal, deg = get_default_basal_and_deg()
    s_idx = GENE2IDX["SOX2"]
    d_idx = GENE2IDX["DCX"]
    
    # Diverse initial conditions spanning the landscape
    grid_ics = []
    for si in np.linspace(0.2, 4, n_trials):
        for di in np.linspace(0.01, 2.5, n_trials):
            grid_ics.append((si, di))
    
    attractors_raw = []
    for sox_init, dcx_init in grid_ics:
        y0 = np.ones(N_GENES) * 0.001
        y0[s_idx] = sox_init
        y0[d_idx] = dcx_init
        
        try:
            if morphogen_profile:
                sol = solve_ivp(
                    lambda t, y: morphogen_modulated_ode(
                        t, y, basal, deg, _ACT_MAP, _REP_MAP,
                        shh_level=morphogen_profile.get("shh", 0.0),
                        bmp_level=morphogen_profile.get("bmp", 0.0),
                        wnt_level=morphogen_profile.get("wnt", 0.0),
                    ),
                    (0, 300), y0, method="LSODA",
                    max_step=0.5, rtol=1e-6, atol=1e-8,
                )
            else:
                sol = solve_ivp(
                    lambda t, y: grn_ode(t, y, basal, deg, _ACT_MAP, _REP_MAP),
                    (0, 300), y0, method="LSODA",
                    max_step=0.5, rtol=1e-6, atol=1e-8,
                )
            
            yf = sol.y[:, -1]
            attractors_raw.append({
                "SOX2": float(yf[s_idx]),
                "DCX": float(yf[d_idx]),
            })
        except:
            pass
    
    # Cluster by proximity (round to 2 decimals and deduplicate)
    seen = {}
    for a in attractors_raw:
        key = (round(a["SOX2"], 2), round(a["DCX"], 2))
        seen[key] = a  # last one wins
    
    # Classify attractors by their position
    classified = []
    for key, a in seen.items():
        sox, dcx = key
        
        # DPS from approximate stem/diff scoring
        stem_score = max(0, sox - 0.02) * 3  # 3 active stem genes approx
        diff_score = max(0, dcx - 0.01) * 5  # 5 active diff genes approx
        denom = stem_score + diff_score + 0.001
        dps = round(stem_score / denom, 3)
        
        if sox > 1.5 and dcx < 0.1:
            label = "Stem"
        elif sox > 0.8 and dcx > 0.3:
            label = "Proneural"
        elif sox < 0.5 and dcx < 0.2:
            label = "Astroglial"
        elif sox < 0.3 and dcx > 0.5:
            label = "Neuronal"
        else:
            label = "Intermediate"
        
        classified.append({
            "label": label,
            "SOX2": a["SOX2"],
            "DCX": a["DCX"],
            "DPS": dps,
        })
    
    # Keep distinct attractors sorted by SOX2 descending
    classified.sort(key=lambda x: -x["SOX2"])
    return classified[:5]  # at most 5 distinct attractors


# ── Multi-landscape Visualization ─────────────────────────────────────

def plot_multi_landscape(landscapes, morphogen_labels, 
                         title="Morphogen-Dependent Waddington Landscape"):
    """
    绘制多条件对比景观图.
    
    Ribes & Briscoe 2009: 形态发生素梯度决定神经管细胞命运
    零信号 → 单吸引子 (干细胞)
    加信号 → 新吸引子 (分岔)
    """
    n_plots = len(landscapes)
    fig = plt.figure(figsize=(7*n_plots, 11))
    
    for idx, ((S, D, dS, dD, phi), label) in enumerate(zip(landscapes, morphogen_labels)):
        # ── 3D surface ──
        ax = fig.add_subplot(2, n_plots, idx + 1, projection='3d')
        stride = 2
        X = S[::stride, ::stride]
        Y = D[::stride, ::stride]
        Z = phi[::stride, ::stride]
        
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis_r, alpha=0.9,
                               linewidth=0, antialiased=True)
        
        # Attractors
        attractors = find_attractors(S, D, phi)
        for attr in attractors:
            si = np.argmin(np.abs(S[:, 0] - attr["SOX2"]))
            dj = np.argmin(np.abs(D[0, :] - attr["DCX"]))
            pv = phi[si, dj]
            
            colors_map = {"Stem": "#1b7837", "Intermediate": "#f4a582",
                         "Differentiated": "#2166ac"}
            c = colors_map.get(attr["label"], "#999")
            ax.scatter(attr["SOX2"], attr["DCX"], pv + 0.02, c=c, s=150,
                      edgecolors="black", linewidth=2, zorder=10)
        
        ax.set_xlabel("SOX2", fontsize=9, labelpad=6)
        ax.set_ylabel("DCX", fontsize=9, labelpad=6)
        ax.set_zlabel("Φ", fontsize=9, labelpad=4)
        ax.set_title(f"{label}\n({len(attractors)} attractors)", 
                     fontsize=10, fontweight="bold")
        ax.view_init(elev=22, azim=-65)
        ax.dist = 9
        
        # ── 2D contour + attractors ──
        ax2 = fig.add_subplot(2, n_plots, n_plots + idx + 1)
        contour = ax2.contourf(S, D, phi, levels=15, cmap=cm.viridis_r, alpha=0.8)
        
        for attr in attractors:
            colors_map = {"Stem": "#1b7837", "Intermediate": "#f4a582",
                         "Differentiated": "#2166ac"}
            c = colors_map.get(attr["label"], "#999")
            marker = {"Stem": "o", "Intermediate": "s", "Differentiated": "^"}
            m = marker.get(attr["label"], "o")
            ax2.scatter(attr["SOX2"], attr["DCX"], c=c, s=120,
                       edgecolors="black", linewidth=2, zorder=5, marker=m,
                       label=f"{attr['label']} (DPS={attr['DPS']:.2f})")
        
        # Streamline overlay for flow
        skip = 3
        # Quiver for vector field (more robust than streamplot on this mpl version)
        ax2.quiver(S[::skip, ::skip], D[::skip, ::skip],
                  dS[::skip, ::skip], dD[::skip, ::skip],
                  color="white", alpha=0.5, width=0.003,
                  scale=40, headwidth=3)
        
        ax2.set_xlabel("SOX2 (Stemness)", fontsize=10)
        ax2.set_ylabel("DCX (Differentiation)", fontsize=10)
        ax2.legend(fontsize=7, loc="upper right")
        ax2.grid(alpha=0.1)
    
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save_fig(fig, "morphogen_landscape.png")


def plot_attractor_summary(all_attractors, morphogen_labels):
    """
    吸引子数量随形态发生素浓度的变化 — 分岔图.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = range(len(morphogen_labels))
    n_attractors = [len(a) for a in all_attractors]
    
    ax.bar(x, n_attractors, color=["#1b7837", "#f4a582", "#2166ac", "#d6604d"],
           edgecolor="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(morphogen_labels, fontsize=9)
    ax.set_ylabel("Number of attractors (cell fates)", fontsize=10)
    ax.set_title("Landscape Complexity vs Morphogen Signal\n(Bifurcation to new fates)", 
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(n_attractors) + 2)
    
    for i, n in enumerate(n_attractors):
        ax.text(i, n + 0.15, str(n), ha="center", fontsize=14, fontweight="bold")
    
    fig.tight_layout()
    return _save_fig(fig, "attractor_bifurcation.png")

# ── Main entry point ─────────────────────────────────────────────────

def run_landscape_analysis():
    """运行Waddington景观分析（多形态发生素条件）."""
    print("=" * 60)
    print("v5.0 — Waddington Epigenetic Landscape Analysis")
    print("Paper: Wang 2022 (WIRES), Bhatt 2020 (Nat Rev Genet)")
    print("=" * 60)
    
    # Define morphogen conditions mimicking SVZ neurogenic niche
    conditions = [
        {"shh": 0.0, "bmp": 0.0, "wnt": 0.0},  # No signal — stem only
        {"shh": 2.0, "bmp": 0.0, "wnt": 0.0},  # SHH — promote stemness
        {"shh": 0.0, "bmp": 1.5, "wnt": 0.0},  # BMP — promote differentiation
        {"shh": 0.5, "bmp": 1.0, "wnt": 0.3},  # Mixed — SVZ-like
    ]
    labels = [
        "No morphogen",
        "SHH gradient (stemness)",
        "BMP gradient (differentiation)",
        "SVZ niche mix\n(SHH+BMP+Wnt)",
    ]
    
    landscapes = []
    all_attractors = []
    
    for i, (prof, label) in enumerate(zip(conditions, labels)):
        print(f"\n  [{i+1}/4] Computing: {label}")
        S, D, dS, dD, phi = compute_landscape(morphogen_profile=prof)
        attractors = find_attractors(S, D, phi, morphogen_profile=prof)
        landscapes.append((S, D, dS, dD, phi))
        all_attractors.append(attractors)
        
        print(f"    Found {len(attractors)} attractor(s):")
        for a in attractors:
            print(f"      {a['label']:16s}: SOX2={a['SOX2']:.3f}  DCX={a['DCX']:.3f}  "
                  f"DPS={a['DPS']:.3f}")
    
    print("\n  Plotting multi-landscape comparison...")
    path = plot_multi_landscape(landscapes, labels)
    print(f"  ✅ {path}")
    
    print("  Plotting attractor bifurcation summary...")
    path2 = plot_attractor_summary(all_attractors, labels)
    print(f"  ✅ {path2}")
    
    results = {
        "attractors_by_condition": all_attractors,
        "landscape_figure": path,
        "bifurcation_figure": path2,
    }
    
    print("\n" + "=" * 60)
    print("Landscape analysis complete ✅")
    print("=" * 60)
    return results


if __name__ == "__main__":
    run_landscape_analysis()
