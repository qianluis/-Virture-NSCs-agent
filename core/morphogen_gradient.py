"""
v5.0 — Morphogen Gradient & Niche Feedback Module
==================================================
形态发生素梯度模型 + GRN↔Niche双向耦合.

论文依据:
- Ribes & Briscoe 2009, "Morphogen gradients in neural development"
  (Nat Rev Neurosci): SHH/BMP/Wnt 梯度浓度决定神经管背腹轴命运
- Bollenbach et al. 2007, "Precision of morphogen gradients"
  (Nature): 梯度读取的精确性来自受体饱和
- Karr et al. 2012, "Whole-cell computational model" (Cell):
  单细胞状态驱动群体行为

核心:
  1. Morphogen gradients from discrete sources (exponential decay)
  2. Cells read local concentration, morphogen directly modulates GRN ODE
  3. Cells migrate along gradients (chemotaxis)
  4. Cell state determines fate → division vs quiescence
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grn_model import (
    GENES, GENE2IDX, N_GENES, REGULATIONS,
    _ACT_MAP, _REP_MAP, _BASAL_DEG,
    hill_activate, compute_dps, get_default_basal_and_deg, grn_ode,
    build_regulation_lookup
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

SIGMOID = lambda x, k=0.5: 1.0 / (1.0 + np.exp(-(x-0.5)/k))  # soft step

# ── 1. Morphogen Source & Diffusion ──────────────────────────────────

class MorphogenGradient:
    """1D morphogen gradient field with exponential decay sources."""
    
    def __init__(self, grid_size=64):
        self.grid_size = grid_size
        self.x = np.linspace(0, 1, grid_size)
        self._morphogens = {}
    
    def add_source(self, name, position, amplitude, decay_length=0.15, noise=0.02):
        dist = np.abs(self.x - position)
        conc = amplitude * np.exp(-dist / decay_length)
        conc *= (1.0 + noise * np.random.randn(self.grid_size))
        self._morphogens[name] = np.maximum(conc, 0)
        return self._morphogens[name]

    def get_concentration(self, name, cell_position):
        if name not in self._morphogens:
            return 0.0
        # Clip to valid range
        idx = np.clip(int(np.round(cell_position * (self.grid_size - 1))), 0, self.grid_size - 1)
        return float(self._morphogens[name][idx])

    def get_all_conc(self, cell_position):
        return {name: self.get_concentration(name, cell_position) 
                for name in self._morphogens}

    def plot_gradients(self, title="Morphogen Gradients"):
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = {"SHH": "#e41a1c", "BMP": "#377eb8", "Wnt": "#4daf4a"}
        for name, conc in self._morphogens.items():
            ax.plot(self.x, conc, label=name, color=colors.get(name, "#666"), lw=2.5)
        ax.set_xlabel("Dorso-ventral axis", fontsize=10)
        ax.set_ylabel("Concentration", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return _save_fig(fig, "morphogen_gradients.png")


# ── 2. Morphogen-Modulated GRN ODE ──────────────────────────────────

def morphogen_ode(t, y, basal, deg, act_map, rep_map, morph_dict):
    """
    GRN ODE with morphogen direct modulation.
    
    Ribes & Briscoe 2009:
    - Morphogens modify transcription factor activity via signaling cascades
    - We model this as ODE modifications (direct transcription rate changes)
    - All effects are bounded to prevent numerical blowup
    """
    dydt = grn_ode(t, y, basal, deg, act_map, rep_map)
    
    shh = morph_dict.get("SHH", 0)
    bmp = morph_dict.get("BMP", 0)
    wnt = morph_dict.get("Wnt", 0)
    
    # SHH effect: bound between 0 and 2.0
    if shh > 0.01:
        s_act = 2.0 * shh / (shh + 0.5)
        gli_idx = GENE2IDX.get("GLI1")
        if gli_idx is not None:
            dydt[gli_idx] += s_act
    
    # BMP effect: repress SOX2, promote GFAP — bound
    if bmp > 0.01:
        b_act = 3.0 * bmp / (bmp + 0.5)
        sox_idx = GENE2IDX.get("SOX2")
        if sox_idx is not None:
            # SOX2 repression (bounded)
            sox_rep = b_act * y[sox_idx] / (y[sox_idx] + 0.5)
            dydt[sox_idx] -= min(sox_rep, 2.0)
        gfap_idx = GENE2IDX.get("GFAP")
        if gfap_idx is not None:
            dydt[gfap_idx] += min(b_act * 0.3, 1.0)
    
    # Wnt effect: CTNNB1 activation — bound
    if wnt > 0.01:
        w_act = 2.0 * wnt / (wnt + 0.5)
        ctn_idx = GENE2IDX.get("CTNNB1")
        if ctn_idx is not None:
            dydt[ctn_idx] += min(w_act, 2.0)
    
    return dydt


# ── 3. Gradient-Aware Cell ──────────────────────────────────────────

class GradientCell:
    """Single cell in morphogen gradient field."""
    
    def __init__(self, position, gradient_mgr):
        self.position = np.clip(position, 0.01, 0.99)
        self.gradient = gradient_mgr
        self.state_vector = None
        self.dps_info = None
        self.cell_type = "NSC"
        self.history = []  # track state over time
    
    def simulate(self, t_span=(0, 200)):
        """Run GRN simulation at current position."""
        basal, deg = get_default_basal_and_deg()
        y0 = np.ones(N_GENES) * 0.001
        morph = self.gradient.get_all_conc(self.position)
        
        try:
            sol = solve_ivp(
                lambda t, y: morphogen_ode(t, y, basal, deg, _ACT_MAP, _REP_MAP, morph),
                t_span, y0, method="BDF",  # BDF is more stable for stiff ODEs
                max_step=1.0, rtol=1e-6, atol=1e-8,
            )
            self.state_vector = sol.y[:, -1]
        except:
            # Fallback to LSODA
            sol = solve_ivp(
                lambda t, y: morphogen_ode(t, y, basal, deg, _ACT_MAP, _REP_MAP, morph),
                t_span, y0, method="LSODA",
                max_step=1.0, rtol=1e-6, atol=1e-8,
            )
            self.state_vector = sol.y[:, -1]
        
        self.dps_info = compute_dps(self.state_vector)
        self._classify_cell()
        return self.dps_info
    
    def _classify_cell(self):
        if self.state_vector is None:
            return
        sox2 = self.state_vector[GENE2IDX["SOX2"]]
        gfap = self.state_vector[GENE2IDX["GFAP"]]
        dcx = self.state_vector[GENE2IDX["DCX"]]
        asc = self.state_vector[GENE2IDX["ASCL1"]]
        
        if sox2 > 2.0 and gfap < 1.5:
            self.cell_type = "NSC"
        elif gfap > sox2 and gfap > 2.0:
            self.cell_type = "Astrocyte"
        elif dcx > 0.3 or asc > 1.0:
            self.cell_type = "TAP"
        else:
            self.cell_type = "NSC"
    
    def get_expression(self, genes=None):
        if self.state_vector is None:
            return {}
        if genes is None:
            return dict(zip(GENES, self.state_vector))
        return {g: float(self.state_vector[GENE2IDX[g]]) 
                for g in genes if g in GENE2IDX}


# ── 4. Niche Simulation ─────────────────────────────────────────────

class NicheSimulation:
    """Multi-cell niche simulation with morphogen gradients."""
    
    def __init__(self, n_cells=20):
        self.n_cells = n_cells
        self.cells = []
        self.gradient = MorphogenGradient()
        self.time = 0
        
        # SVZ-like gradients
        self.gradient.add_source("SHH", 0.1, 2.0, 0.12)
        self.gradient.add_source("BMP", 0.9, 2.5, 0.15)
        self.gradient.add_source("Wnt", 0.5, 1.5, 0.18)
        
        # Initialize cells in SVZ zones
        np.random.seed(42)
        zones = [0.15, 0.25, 0.40, 0.55, 0.70, 0.85]
        for i in range(n_cells):
            pos = zones[i % len(zones)] + 0.03 * np.random.randn()
            pos = np.clip(pos, 0.05, 0.95)
            cell = GradientCell(pos, self.gradient)
            cell.simulate()
            self.cells.append(cell)
    
    def run_step(self, migration=0.005):
        """One time step: migrate + divide + re-simulate."""
        self.time += 1
        new_cells = list(self.cells)
        
        for cell in self.cells:
            if cell.state_vector is None:
                continue
            
            sox2 = cell.state_vector[GENE2IDX["SOX2"]]
            dcx = cell.state_vector[GENE2IDX["DCX"]]
            
            # Chemotaxis: NSC move toward SHH (ventral, pos↓), TAP toward dorsal (pos↑)
            stem_force = sox2 / (sox2 + 0.5) * -migration  # toward SHH
            diff_force = dcx / (dcx + 0.3) * migration     # toward BMP
            cell.position = np.clip(cell.position + stem_force + diff_force, 0.02, 0.98)
            
            # Division if CCND1 high
            ccnd1 = cell.state_vector[GENE2IDX["CCND1"]]
            if ccnd1 > 0.5 and len(new_cells) < self.n_cells * 2:
                daughter = GradientCell(cell.position + 0.01 * np.random.randn(), self.gradient)
                daughter.simulate()
                new_cells.append(daughter)
            
            cell.simulate(t_span=(0, 100 if self.time > 1 else 200))
        
        self.cells = new_cells
        return self._get_stats()
    
    def _get_stats(self):
        types = {}
        dps_list = []
        pos_list = []
        for c in self.cells:
            types[c.cell_type] = types.get(c.cell_type, 0) + 1
            if c.dps_info:
                dps_list.append(c.dps_info["DPS"])
            pos_list.append(c.position)
        return {
            "n_cells": len(self.cells),
            "types": types,
            "mean_dps": float(np.mean(dps_list)) if dps_list else 0,
            "std_dps": float(np.std(dps_list)) if dps_list else 0,
        }
    
    def run_simulation(self, n_steps=15):
        print(f"  Step 0: {self._get_stats()}")
        history = [self._get_stats()]
        for s in range(1, n_steps + 1):
            h = self.run_step()
            history.append(h)
            if s % 5 == 0:
                print(f"  Step {s}: n={h['n_cells']}, types={h['types']}, DPS={h['mean_dps']:.3f}")
        return history
    
    def plot_niche(self, history):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1: Cell positions colored by DPS
        ax = axes[0, 0]
        positions = [c.position for c in self.cells]
        dps = [c.dps_info["DPS"] if c.dps_info else 0.5 for c in self.cells]
        sc = ax.scatter(positions, np.random.randn(len(self.cells))*0.05,
                       c=dps, cmap="RdYlGn", s=60, vmin=0, vmax=1,
                       edgecolors="black", alpha=0.8)
        ax.set_xlabel("Position (0=ventral, 1=dorsal)")
        ax.set_ylabel("Jitter")
        ax.set_title("Cell DPS vs Position")
        plt.colorbar(sc, ax=ax, label="DPS")
        
        # 2: Population + DPS timeline
        ax = axes[0, 1]
        steps = range(len(history))
        n = [h["n_cells"] for h in history]
        d = [h["mean_dps"] for h in history]
        ax.plot(steps, n, "o-", color="#2166ac", lw=2, label="Cells")
        ax_t = ax.twinx()
        ax_t.plot(steps, d, "s-", color="#d6604d", lw=2, label="Mean DPS")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cell count", color="#2166ac")
        ax_t.set_ylabel("Mean DPS", color="#d6604d")
        ax.grid(alpha=0.3)
        
        # 3: Cell type pie
        ax = axes[1, 0]
        types = self._get_stats()["types"]
        labels = list(types.keys())
        sizes = list(types.values())
        colors_ct = {"NSC": "#1b7837", "TAP": "#f4a582", 
                     "Astrocyte": "#d6604d", "Neuron": "#2166ac"}
        pie_colors = [colors_ct.get(t, "#999") for t in labels]
        ax.pie(sizes, labels=labels, colors=pie_colors, autopct="%1.0f%%",
               startangle=90, textprops={"fontsize": 9})
        ax.set_title("Cell Type Composition")
        
        # 4: DPS vs position
        ax = axes[1, 1]
        ax.scatter(positions, dps, c=[colors_ct.get(c.cell_type, "#999") for c in self.cells],
                  s=80, edgecolors="black", alpha=0.8)
        ax.set_xlabel("Position")
        ax.set_ylabel("DPS")
        ax.set_ylim(0, 1.1)
        ax.grid(alpha=0.3)
        ax.set_title("DPS vs Position (Ribes & Briscoe 2009)")
        
        fig.suptitle("SVZ Niche Simulation v5.0", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return _save_fig(fig, "niche_simulation_v5.png")


def run_niche_gradient_simulation(n_cells=20, n_steps=15):
    print("=" * 60)
    print("v5.0 — Morphogen Gradient Niche Simulation")
    print("Papers: Ribes & Briscoe 2009, Bollenbach 2007, Karr 2012")
    print("=" * 60)

    niche = NicheSimulation(n_cells=n_cells)
    
    for name in niche.gradient._morphogens:
        c = niche.gradient._morphogens[name]
        print(f"  {name}: min={c.min():.3f}, max={c.max():.3f}")
    
    init_types = {}
    for c in niche.cells:
        init_types[c.cell_type] = init_types.get(c.cell_type, 0) + 1
    print(f"  Initial: {init_types}, DPS: {np.mean([c.dps_info['DPS'] for c in niche.cells]):.3f}")
    
    history = niche.run_simulation(n_steps=n_steps)
    
    final = niche._get_stats()
    print(f"\n  Final: n={final['n_cells']}, types={final['types']}, DPS={final['mean_dps']:.3f}")
    
    grad_p = niche.gradient.plot_gradients()
    niche_p = niche.plot_niche(history)
    print(f"  ✅ {grad_p}")
    print(f"  ✅ {niche_p}")
    
    return {"niche": niche, "history": history, "figures": (grad_p, niche_p)}


if __name__ == "__main__":
    run_niche_gradient_simulation()
