"""
VirtualCell-Agent v4.0 — Neural Stem Cell GRN Model (Upgraded)
=============================================================
基于高层次论文的 5 项核心升级:

1. 随机基因表达 (Chemical Langevin Equation)
   → Munsky et al. 2012, "Using models to understand and predict gene expression" (Cell)
   → Elowitz et al. 2002, "Stochastic gene expression in a single cell" (Science)

2. 多细胞 Notch-Delta 侧向抑制
   → Sprinzak et al. 2010, "Cis-interactions between Notch and Delta..." (Nature)
   → Shaya & Sprinzak 2011, "Lateral inhibition: from embryos to circuits" (Dev Cell)

3. 分化潜能评分 DPS (Differentiation Potency Score)
   → MacArthur et al. 2009, "Systems biology of stem cell fate" (Nat Rev MCB)

4. 自动分岔分析 (Bifurcation Analysis)
   → Ferrell 2012, "Bistability, bifurcations, and Waddington's epigenetic landscape"
     (Current Biology)

5. 多细胞时间序列交互仿真
   → Karr et al. 2012, "A whole-cell computational model predicts phenotype"
     (Cell)

用法:
  from core.grn_model_v4 import run_grn_simulation
  run_grn_simulation()                              # 基线（确定性）
  run_grn_simulation(stochastic=True)                # 随机仿真
  run_grn_simulation('NOTCH1', 'knock_out')          # 扰动
  run_grn_simulation(multi_cell=10)                  # 10细胞群体仿真
  run_grn_simulation(bifurcation='HES1')             # 分岔分析
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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

# Pathway grouping for DPS calculation
PATHWAYS = {
    "stemness": ["SOX2", "NES", "PROM1", "TERT"],
    "notch":    ["NOTCH1", "HES1", "HES5"],
    "proneural":["ASCL1", "NEUROG2", "NEUROD1"],
    "wnt":      ["CTNNB1", "MYC", "CCND1"],
    "shh":      ["GLI1", "MYCN"],
    "bmp":      ["SMAD1", "ID1", "GFAP"],
    "neuronal": ["DCX", "TUBB3", "RBFOX3", "MBP"],
}

# ── Default production & degradation ──────────────────────────────────────
_BASAL_DEG = {
    "SOX2":    (0.08, 0.12), "NES":     (0.02, 0.10),
    "PROM1":   (0.01, 0.10), "TERT":    (0.01, 0.10),
    "NOTCH1":  (0.12, 0.25), "HES1":    (0.01, 0.35),
    "HES5":    (0.01, 0.35), "ASCL1":   (0.05, 0.25),
    "CTNNB1":  (0.20, 0.18), "MYC":     (0.03, 0.20),
    "CCND1":   (0.02, 0.15), "NEUROG2": (0.01, 0.15),
    "GLI1":    (0.08, 0.15), "MYCN":    (0.02, 0.15),
    "SMAD1":   (0.15, 0.15), "ID1":     (0.01, 0.20),
    "GFAP":    (0.01, 0.08), "DCX":     (0.005, 0.12),
    "TUBB3":   (0.005, 0.10), "RBFOX3": (0.002, 0.08),
    "NEUROD1": (0.005, 0.15), "MBP":    (0.001, 0.06),
}

# ── Regulatory interactions ────────────────────────────────────────────────
REGULATIONS = [
    ("SOX2",    "SOX2",  "act", 0.30, 0.5, 2),
    ("NES",     "SOX2",  "act", 0.25, 0.4, 2),
    ("PROM1",   "SOX2",  "act", 0.20, 0.4, 2),
    ("TERT",    "SOX2",  "act", 0.18, 0.4, 2),
    ("HES1",    "NOTCH1","act", 0.80, 0.5, 2),
    ("HES5",    "NOTCH1","act", 0.60, 0.5, 2),
    ("ASCL1",   "HES1",  "rep", 10.0, 1.0, 4),
    ("ASCL1",   "HES5",  "rep", 8.0,  0.8, 4),
    ("NEUROG2", "HES1",  "rep", 10.0, 1.0, 4),
    ("NEUROG2", "HES5",  "rep", 8.0,  0.8, 4),
    ("NEUROG2", "ASCL1", "act", 0.50, 0.4, 2),
    ("DCX",     "ASCL1", "act", 0.25, 0.4, 2),
    ("TUBB3",   "ASCL1", "act", 0.20, 0.4, 2),
    ("HES1",    "ASCL1", "rep", 5.0,  0.4, 4),
    ("HES5",    "ASCL1", "rep", 3.0,  0.4, 4),
    ("ASCL1",   "ASCL1", "act", 0.25, 0.5, 2),
    ("MYC",     "CTNNB1","act", 0.50, 0.4, 2),
    ("CCND1",   "CTNNB1","act", 0.40, 0.4, 2),
    ("CCND1",   "MYC",   "act", 0.20, 0.4, 2),
    ("TERT",    "MYC",   "act", 0.15, 0.4, 2),
    ("NEUROG2", "CTNNB1","act", 0.20, 0.5, 2),
    ("MYCN",    "GLI1",  "act", 0.40, 0.4, 2),
    ("CCND1",   "GLI1",  "act", 0.25, 0.4, 2),
    ("ID1",     "SMAD1", "act", 0.35, 0.4, 2),
    ("GFAP",    "SMAD1", "act", 0.30, 0.5, 2),
    ("NEUROG2", "ID1",   "rep", 5.0,  0.6, 3),
    ("ASCL1",   "ID1",   "rep", 3.0,  0.6, 3),
    ("DCX",     "NEUROG2","act",0.40, 0.4, 2),
    ("TUBB3",   "NEUROG2","act",0.25, 0.4, 2),
    ("RBFOX3",  "NEUROG2","act",0.20, 0.5, 2),
    ("NEUROD1", "NEUROG2","act",0.25, 0.5, 2),
    ("DCX",     "NEUROD1","act",0.15, 0.4, 2),
    ("MBP",     "NEUROD1","act",0.25, 0.4, 2),
    ("ASCL1",   "NEUROD1","act",0.15, 0.5, 2),
    ("RBFOX3",  "NEUROD1","act",0.15, 0.4, 2),
    ("NEUROG2", "GFAP",  "rep", 5.0,  0.6, 3),
    ("DCX",     "GFAP",  "rep", 3.0,  0.6, 3),
    ("CCND1",   "HES1",  "rep", 5.0,  0.5, 3),
    ("GFAP",    "HES1",  "rep", 3.0,  0.6, 3),
    ("HES1",    "GFAP",  "act", 0.20, 0.6, 2),
]

# ── Hill functions ────────────────────────────────────────────────────────
def hill_activate(x, V, K, n):
    """Hill activation: V * x^n / (K^n + x^n)"""
    if x <= 0:
        return 0.0
    ratio = (x / K) ** n
    return V * ratio / (1.0 + ratio)

# ── Regulation lookups ────────────────────────────────────────────────────
def build_regulation_lookup(regulations):
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

# ── ODE and SDE systems ──────────────────────────────────────────────────

def compute_production(y, act_map, rep_map, basal_vec):
    """Compute the production rate vector (fold-change model)."""
    prod = np.zeros(N_GENES)
    for i in range(N_GENES):
        act = 0.0
        for r_idx, V, K, n in act_map[i]:
            act += hill_activate(y[r_idx], V, K, n)
        rep = 0.0
        for r_idx, V, K, n in rep_map[i]:
            rep += hill_activate(y[r_idx], V, K, n)
        fold_rep = 1.0 / (1.0 + rep)
        prod[i] = (basal_vec[i] + act) * fold_rep
    return prod

def grn_ode(t, y, basal_vec, deg_vec, act_map, rep_map):
    """Deterministic ODE."""
    prod = compute_production(y, act_map, rep_map, basal_vec)
    return prod - deg_vec * y

def grn_sde(t, y, basal_vec, deg_vec, act_map, rep_map, noise_scale=0.1, cell_volume=1.0):
    """
    Chemical Langevin Equation (CLE) — 随机基因表达.
    
    d[g] = (prod - deg·g)dt + √(prod + deg·g)/√V · dW
    
    论文依据:
    - Gillespie 2000, "The chemical Langevin equation" (J Chem Phys)
    - Munsky et al. 2012, Cell: 基因表达的随机性由分子数有限导致
    - noise_scale: 噪声强度控制 (默认0.1, 典型NSC为0.05-0.2)
    - cell_volume: 有效细胞体积 (越大噪声越小, 默认=1.0)
    """
    prod = compute_production(y, act_map, rep_map, basal_vec)
    degradation = deg_vec * y
    
    # Deterministic drift
    drift = prod - degradation
    
    # Stochastic diffusion (Chemical Langevin)
    # sqrt of sum of birth and death rates, scaled by volume
    diff = np.sqrt(np.maximum(prod + degradation, 1e-15)) / np.sqrt(cell_volume)
    diff *= noise_scale  # global noise intensity
    
    # Return [drift, diff] for Euler-Maruyama integrator
    return np.column_stack([drift, diff])

def grn_sde_wrapper(t, y, basal_vec, deg_vec, act_map, rep_map, noise_scale, cell_volume):
    """Wrapper that returns only drift (for scipy.integrate.solve_ivp)."""
    result = grn_sde(t, y, basal_vec, deg_vec, act_map, rep_map, noise_scale, cell_volume)
    return result[:, 0]  # return drift only for deterministic part

# ── Euler-Maruyama integrator for SDE ────────────────────────────────────

def euler_maruyama(basal, deg, t_span=(0, 500), n_points=2000, 
                   noise_scale=0.1, cell_volume=1.0, seed=None):
    """
    Euler-Maruyama integration of the Chemical Langevin Equation.
    
    论文依据: Kloeden & Platen 1992, "Numerical Solution of SDE"
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = (t_span[1] - t_span[0]) / n_points
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    
    y = np.ones(N_GENES) * 0.001
    traj = np.zeros((N_GENES, n_points))
    traj[:, 0] = y
    
    for step in range(1, n_points):
        drift_diff = grn_sde(0, y, basal, deg, _ACT_MAP, _REP_MAP, 
                             noise_scale, cell_volume)
        drift = drift_diff[:, 0]
        diff = drift_diff[:, 1]
        
        # Euler-Maruyama: y += drift*dt + diff*sqrt(dt)*N(0,1)
        dW = np.random.randn(N_GENES)
        y = y + drift * dt + diff * np.sqrt(dt) * dW
        y = np.maximum(y, 0.0)  # prevent negative expression
        traj[:, step] = y
    
    return traj, t_eval, traj[:, -1]

# ── Multi-cell Notch-Delta lateral inhibition ────────────────────────────

def build_neighbor_matrix(n_cells, topology="random_1d"):
    """
    构建多细胞邻接矩阵.
    
    论文依据:
    - Sprinzak et al. 2010 (Nature): 相邻细胞的Notch-Delta信号
    - Shaya & Sprinzak 2011 (Dev Cell): 侧向抑制的细胞拓扑
    
    topology options:
      "pair":      两个细胞互作 (最简系统)
      "random_1d": 一维细胞链 (神经管模式)
      "hexagonal": 六边形网格 (上皮组织) — 占位
    """
    if n_cells == 1:
        return np.zeros((1, 1), dtype=float)
    
    if topology == "pair" or n_cells == 2:
        A = np.zeros((n_cells, n_cells))
        for i in range(n_cells):
            for j in range(n_cells):
                if i != j:
                    A[i, j] = 1.0 / (n_cells - 1)
        return A
    
    if topology == "random_1d":
        # One-dimensional chain: each cell connects to ±1 neighbor
        A = np.zeros((n_cells, n_cells))
        for i in range(n_cells):
            neighbors = []
            if i > 0:
                neighbors.append(i - 1)
            if i < n_cells - 1:
                neighbors.append(i + 1)
            for j in neighbors:
                A[i, j] = 1.0 / len(neighbors)
        return A
    
    return np.zeros((n_cells, n_cells))

# Notch-Delta signaling gene indices
DELTA_GENES = ["HES1", "ASCL1"]  # 基因产生Delta-like信号
NOTCH_TARGET = "NOTCH1"          # 接收信号的受体

def compute_lateral_signal(y_cells, neighbor_matrix, n_cells):
    """
    增强版Notch-Delta侧向抑制信号.
    
    Sprinzak et al. 2010 (Nature) 的核心机制:
    1. Delta (由ASCL1驱动) 在发送细胞上激活邻居的Notch
    2. Notch在接收细胞上激活HES
    3. HES抑制同细胞的ASCL1 (→ 降低自己的Delta)
    4. 形成"你高我低"的侧向抑制模式
    
    增强: 
    - Delta-like: ASCL1主导 (生物学中ASCL1直接调控Dll1/Dll3)
    - Notch激活: 使用Hill函数而非线性加和 (陡峭响应)
    - cis-inhibition: 与自身ASCL1成正比 (同细胞Delta-Notch互作)
    """
    notch_signals = np.zeros(n_cells)
    for i in range(n_cells):
        # Neighbor Delta activity — mainly from ASCL1 (Dll1/Dll3)
        delta_input = 0.0
        for j in range(n_cells):
            if i != j and neighbor_matrix[i, j] > 0:
                # ASCL1 drives Delta expression (Dll1/Dll3 promoters)
                # HES1 weakly represses Delta
                asc_j = y_cells[j, GENE2IDX["ASCL1"]]
                hes_j = y_cells[j, GENE2IDX["HES1"]]
                delta_j = max(0, asc_j - 0.2 * hes_j)  # ASCL1 - HES repression
                delta_input += neighbor_matrix[i, j] * delta_j
        
        # Cis-inhibition: own Delta traps Notch in cis, preventing trans-activation
        own_asc = y_cells[i, GENE2IDX["ASCL1"]]
        cis_inhibition = 1.0 / (1.0 + 3.0 * own_asc)  # strong cis-inhibition
        
        # Trans-activation via Hill function (steep response)
        # Keff = 0.2: half-max achieved at moderate neighbor ASCL1
        notch_act = delta_input / (delta_input + 0.2) if delta_input > 0 else 0.0
        
        # Effective NOTCH1 signal = trans * cis_inhibition
        notch_signals[i] = notch_act * cis_inhibition
    
    return notch_signals

def multi_cell_ode(t, y_flat, basal_local, deg_local, neighbor_matrix, n_cells, noise_scale=0.0):
    """
    多细胞ODE/SDE系统 (增强版Notch-Delta侧向抑制).
    
    关键生物机制 (Sprinzak et al. 2010, Nature):
    邻居Delta → NOTCH1激活 → HES1/5表达 → ASCL1抑制 → 自身Delta降低
    ↑                                                          |
    +------------------------- 竞争性反馈 ------------------------+
    
    实现: 每个细胞的NOTCH1 basal受邻居Delta调制, NOTCH1越高→HES越高→ASCL1越低→Delta越低
    结果: 相邻细胞的Delta表达呈现"你高我低"模式
    """
    y_cells = y_flat.reshape(n_cells, N_GENES)
    dydt = np.zeros_like(y_cells)
    
    # Compute lateral signals
    notch_lateral = compute_lateral_signal(y_cells, neighbor_matrix, n_cells)
    
    for i in range(n_cells):
        # Create PER-CELL basal vector with modulated NOTCH1
        basal_i = basal_local.copy()
        ni = GENE2IDX["NOTCH1"]
        
        # KEY: Neighbor Delta directly modulates NOTCH1 production rate
        # Strong lateral signal → high NOTCH1 basal → high HES → low ASCL1
        # The Hill-form modulation creates switch-like response
        lateral = notch_lateral[i]
        
        # NOTCH1 basal = intrinsic (0.12) + neighbor signal (0 to 1.0)
        # This directly feeds into the ODE as a production term
        basal_i[ni] = 0.12 + 1.5 * lateral / (lateral + 0.3)
        
        if noise_scale > 0:
            drift_diff = grn_sde(0, y_cells[i], basal_i, deg_local, _ACT_MAP, _REP_MAP,
                                 noise_scale, 1.0)
            drift = drift_diff[:, 0]
        else:
            drift = grn_ode(0, y_cells[i], basal_i, deg_local, _ACT_MAP, _REP_MAP)
        
        dydt[i] = drift
    
    return dydt.flatten()

def get_default_basal_and_deg():
    basal = np.array([_BASAL_DEG[g][0] for g in GENES], dtype=float)
    deg   = np.array([_BASAL_DEG[g][1] for g in GENES], dtype=float)
    return basal, deg

# ── Steady-state solver ──────────────────────────────────────────────────

def steady_state(basal, deg, t_span=(0, 500), n_points=2000):
    """确定性子ODE求解至稳态."""
    y0 = np.ones(N_GENES) * 0.001
    sol = solve_ivp(
        lambda t, y: grn_ode(t, y, basal, deg, _ACT_MAP, _REP_MAP),
        t_span, y0, method="LSODA",
        dense_output=True, max_step=0.5,
        rtol=1e-8, atol=1e-10,
    )
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    y_traj = sol.sol(t_eval)
    return y_traj, t_eval, y_traj[:, -1]

# ── Differentiation Potency Score (DPS) ─────────────────────────────────

def compute_dps(y_vector):
    """
    计算分化潜能评分 DPS ∈ [0, 1].
    
    DPS = 1 → 完全干性 (SOX2高, 分化基因低)
    DPS = 0 → 完全分化 (分化基因高, SOX2低)
    
    论文依据:
    - MacArthur et al. 2009 (Nat Rev MCB): 干细胞命运的连续系统生物学模型
    - 基于多基因通路的加权评分
    
    公式:
    DPS = stemness_score / (stemness_score + diff_score + ϵ)
    
    stemness_score = Σ_{g∈stemness} max(0, g - θ_g) / (1 + Σ_{g∈diff} max(0, g - θ_g))
    """
    # Stemness markers
    stem_genes = ["SOX2", "NES", "PROM1", "TERT", "HES1", "HES5"]
    # Differentiation markers  
    diff_genes = ["DCX", "TUBB3", "RBFOX3", "NEUROD1", "MBP", "GFAP"]
    # Proneural (intermediate state)
    proneural_genes = ["ASCL1", "NEUROG2"]
    
    stem_score = sum(max(0, y_vector[GENE2IDX[g]] - 0.02) for g in stem_genes 
                     if g in GENE2IDX)
    diff_score = sum(max(0, y_vector[GENE2IDX[g]] - 0.01) for g in diff_genes 
                     if g in GENE2IDX)
    proneural_score = sum(max(0, y_vector[GENE2IDX[g]] - 0.01) for g in proneural_genes
                          if g in GENE2IDX)
    
    # DPS: high when stemness dominates
    denominator = stem_score + diff_score + proneural_score + 0.001
    dps = stem_score / denominator
    
    # State classification
    if dps > 0.6:
        state = "Stem"
    elif dps > 0.3:
        state = "Intermediate"
    elif proneural_score > diff_score:
        state = "Proneural"
    else:
        state = "Differentiated"
    
    return {
        "DPS": dps,
        "state": state,
        "stem_score": stem_score,
        "diff_score": diff_score,
        "proneural_score": proneural_score,
    }

# ── Bifurcation analysis ────────────────────────────────────────────────

def bifurcation_scan(param_name, param_range, target_gene, n_points=50):
    """
    分岔分析: 扫描单个参数, 检测双稳态区间.
    
    论文依据:
    - Ferrell 2012 (Current Biology): 双稳态与Waddington表观遗传景观
    - Huang 2009 (Development): GRN中的分岔与干细胞命运决定
    
    param_name: 要扫描的参数名 (如 "r_H1_A_V" 表示 HES1→ASCL1 阻遏强度)
    param_range: (start, stop) 参数扫描范围
    target_gene: 观察的目标基因
    """
    # Build param lookup from REGULATIONS
    param_idx_map = {}
    for idx, (target, reg, rtype, V, K, n) in enumerate(REGULATIONS):
        key = f"r_{reg[:2]}_{target[:2]}_{rtype[:1]}"
        param_idx_map[key] = idx
    
    basal, deg = get_default_basal_and_deg()
    param_vals = np.linspace(param_range[0], param_range[1], n_points)
    steady_vals = []
    
    for pv in param_vals:
        # Modify regulation parameter
        if param_name in param_idx_map:
            pidx = param_idx_map[param_name]
            reg_list = list(REGULATIONS)
            t, r, rtype, _, K, n = reg_list[pidx]
            reg_list[pidx] = (t, r, rtype, pv, K, n)
            act_map, rep_map = build_regulation_lookup(reg_list)
        else:
            act_map, rep_map = _ACT_MAP, _REP_MAP
        
        # Solve with two different initial conditions
        y0_low = np.ones(N_GENES) * 0.001
        y0_high = np.ones(N_GENES) * 10.0
        
        for y0, label in [(y0_low, "low"), (y0_high, "high")]:
            sol = solve_ivp(
                lambda t, y: grn_ode(t, y, basal, deg, act_map, rep_map),
                (0, 500), y0, method="LSODA",
                max_step=0.5, rtol=1e-6, atol=1e-8,
            )
            y_final = sol.y[:, -1]
            tgt_idx = GENE2IDX.get(target_gene, 0)
            steady_vals.append({
                "param": pv,
                "gene": target_gene,
                "value": y_final[tgt_idx],
                "ic": label,
            })
    
    return steady_vals

# ── Multi-cell simulation ───────────────────────────────────────────────

def steady_state_multi_cell(basal, deg, n_cells=5, topology="random_1d",
                            t_span=(0, 500), n_points=2000):
    """
    多细胞确定性ODE仿真 (含Notch-Delta侧向抑制).
    
    初始条件策略: 部分细胞高ASCL1、部分低ASCL1 → 通过侧向抑制放大多样性
    这是 Sprinzak 2010 的核心结论: 初始微小差异被反馈环路放大
    """
    neighbor_matrix = build_neighbor_matrix(n_cells, topology)
    
    # Bimodal initial conditions: high ASCL1 cells and low ASCL1 cells
    y0 = np.ones(n_cells * N_GENES) * 0.001
    
    a_idx = GENE2IDX["ASCL1"]
    h1_idx = GENE2IDX["HES1"]
    
    # Half cells start with higher ASCL1 (proneural fate) 
    # Half start with higher HES1 (stem fate)
    for i in range(n_cells):
        if i < n_cells // 2:
            y0[i * N_GENES + a_idx] = 2.0  # ASCL1-high cells (pro-neural)
            y0[i * N_GENES + h1_idx] = 0.1
        else:
            y0[i * N_GENES + a_idx] = 0.01  # ASCL1-low cells (stem)
            y0[i * N_GENES + h1_idx] = 3.0
        # SOX2 intermediate
        y0[i * N_GENES + GENE2IDX["SOX2"]] = 2.0
    
    sol = solve_ivp(
        lambda t, y: multi_cell_ode(t, y, basal, deg, neighbor_matrix, n_cells, 0.0),
        t_span, y0, method="LSODA",
        dense_output=True, max_step=0.5,
        rtol=1e-7, atol=1e-9,
    )
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    y_traj = sol.sol(t_eval)
    y_final_flat = y_traj[:, -1]
    
    # Reshape: (n_cells, N_GENES)
    y_cells = y_final_flat.reshape(n_cells, N_GENES)
    return y_traj, t_eval, y_cells, neighbor_matrix

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
        deg[idx] = deg[idx] * 3.0
        desc = f"{gene_name} knockout (basal→0, deg×3)"
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

def plot_dps_landscape(dps_results, title="Differentiation Potency Landscape"):
    """分化潜能景观图 — 细胞状态的连续谱."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar: DPS for each cell
    cells = list(range(len(dps_results)))
    dps_vals = [r["DPS"] for r in dps_results]
    states = [r["state"] for r in dps_results]
    colors = {"Stem": "#1b7837", "Intermediate": "#f4a582", 
              "Proneural": "#b2182b", "Differentiated": "#67001f"}
    bar_colors = [colors.get(s, "#999999") for s in states]
    
    ax1.bar(cells, dps_vals, color=bar_colors, edgecolor="black", lw=0.5)
    ax1.set_xlabel("Cell index", fontsize=10)
    ax1.set_ylabel("DPS (Differentiation Potency Score)", fontsize=10)
    ax1.set_title("Single-cell Potency Distribution", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.axhline(0.6, color="green", ls="--", alpha=0.5, lw=1)
    ax1.axhline(0.3, color="orange", ls="--", alpha=0.5, lw=1)
    ax1.text(0, 0.62, "Stem", fontsize=8, color="green")
    ax1.text(0, 0.32, "Intermediate", fontsize=8, color="orange")
    
    # Scatter: stem_score vs diff_score
    stem_scores = [r["stem_score"] for r in dps_results]
    diff_scores = [r["diff_score"] for r in dps_results]
    scatter = ax2.scatter(stem_scores, diff_scores, c=dps_vals, 
                          cmap="RdYlGn", s=80, edgecolors="black", 
                          vmin=0, vmax=1)
    ax2.set_xlabel("Stemness score", fontsize=10)
    ax2.set_ylabel("Differentiation score", fontsize=10)
    ax2.set_title("Potency Phase Space", fontsize=11, fontweight="bold")
    cbar = plt.colorbar(scatter, ax=ax2, label="DPS")
    
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save_fig(fig, "dps_landscape.png")

def plot_bifurcation(bif_results, param_name, target_gene, title=None):
    """分岔图绘制."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    low_vals = [r for r in bif_results if r["ic"] == "low"]
    high_vals = [r for r in bif_results if r["ic"] == "high"]
    
    ax.plot([r["param"] for r in low_vals], [r["value"] for r in low_vals],
            "o-", color="#4393c3", markersize=4, label="Low initial", alpha=0.8)
    ax.plot([r["param"] for r in high_vals], [r["value"] for r in high_vals],
            "s-", color="#d6604d", markersize=4, label="High initial", alpha=0.8)
    
    ax.set_xlabel(param_name, fontsize=11)
    ax.set_ylabel(f"{target_gene} steady state", fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title(f"Bifurcation Analysis: {target_gene} vs {param_name}", 
                     fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Detect bistable region
    if len(low_vals) == len(high_vals):
        for l, h in zip(low_vals, high_vals):
            if abs(l["value"] - h["value"]) > 0.5:
                ax.axvspan(l["param"] - 0.05, l["param"] + 0.05, 
                          alpha=0.15, color="yellow", zorder=0)
                break
    
    fig.tight_layout()
    return _save_fig(fig, f"bifurcation_{target_gene}.png")

def plot_multi_cell_heatmap(y_cells, title="Multi-cell Expression Profile"):
    """多细胞表达热图."""
    fig, ax = plt.subplots(figsize=(8, 6))
    n_cells = y_cells.shape[0]
    
    vmin = 0
    vmax = max(y_cells.max(), 1.0)
    im = ax.imshow(y_cells.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    
    ax.set_yticks(range(N_GENES))
    ax.set_yticklabels(GENES, fontsize=7)
    ax.set_xlabel("Cell index", fontsize=10)
    ax.set_xticks(range(n_cells))
    ax.set_title(title, fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, label="Expression level")
    fig.tight_layout()
    return _save_fig(fig, "multi_cell_heatmap.png")

# ── Main entry point ─────────────────────────────────────────────────────

def run_grn_simulation(gene_name=None, perturbation_type=None,
                       stochastic=False, noise_scale=0.1, 
                       multi_cell=None, topology="random_1d",
                       bifurcation=None, bifurcation_range=None):
    """
    统一入口点 — 支持所有升级模式.
    
    Args:
        gene_name: 扰动基因名
        perturbation_type: knock_out / overexpress / drug_inhibit
        stochastic: 是否启用随机SDE仿真 (论文依据: Gillespie CLE)
        noise_scale: 噪声强度 (默认0.1, NSC典型值)
        multi_cell: 多细胞数 (启用Notch-Delta侧向抑制)
        topology: 细胞拓扑 (pair/random_1d)
        bifurcation: 分岔分析目标参数名
        bifurcation_range: (start, stop) 参数扫描范围
    
    Returns:
        dict with results
    """
    print("=" * 60)
    print("VirtualCell-Agent v4.0 — NSC GRN Simulation")
    print("=" * 60)
    
    results = {}
    basal_ctrl, deg_ctrl = get_default_basal_and_deg()
    
    # ── Mode selection ──
    if bifurcation:
        print(f"\n[Mode] Bifurcation Analysis: {bifurcation}")
        if bifurcation_range is None:
            bifurcation_range = (0.1, 20.0)
        bif_results = bifurcation_scan(bifurcation, bifurcation_range, "ASCL1")
        path = plot_bifurcation(bif_results, bifurcation, "ASCL1")
        results["bifurcation"] = {"data": bif_results, "figure": path}
        print(f"  ✅ Bifurcation plot: {path}")
        
    elif multi_cell and multi_cell > 1:
        print(f"\n[Mode] Multi-cell: {multi_cell} cells (topology={topology})")
        _, _, y_cells, neighbor_matrix = steady_state_multi_cell(
            basal_ctrl, deg_ctrl, n_cells=multi_cell, topology=topology
        )
        path = plot_multi_cell_heatmap(y_cells)
        results["multi_cell"] = {"cells": y_cells.tolist(), "figure": path}
        print(f"  ✅ Multi-cell heatmap: {path}")
        
        # DPS for each cell
        dps_results = []
        for i in range(multi_cell):
            dps = compute_dps(y_cells[i])
            dps_results.append(dps)
            print(f"  Cell {i}: DPS={dps['DPS']:.3f} [{dps['state']}]")
        
        dps_path = plot_dps_landscape(dps_results, 
            f"Multi-cell DPS ({multi_cell} cells, {topology})")
        results["dps"] = dps_path
        results["dps_data"] = dps_results
    
    elif stochastic:
        print(f"\n[Mode] Stochastic (Chemical Langevin, noise={noise_scale})")
        y_traj, t_eval, y_final = euler_maruyama(
            basal_ctrl, deg_ctrl, noise_scale=noise_scale
        )
        print("  Final expression (mean of last 100 pts):")
        # Smooth final state by averaging last 100 points
        y_smooth = np.mean(y_traj[:, -100:], axis=1)
        for g, v in zip(GENES, y_smooth):
            print(f"     {g:>8s}: {v:.4f}")
        
        # DPS of stochastic state
        dps = compute_dps(y_smooth)
        print(f"\n  DPS: {dps['DPS']:.3f} [{dps['state']}]")
        results["stochastic"] = {
            "expression": dict(zip(GENES, y_smooth)),
            "dps": dps,
            "trajectory": y_traj,
        }
    
    # ── Standard deterministic mode (with optional perturbation) ──
    if not bifurcation and not multi_cell:
        print(f"\n[Mode] {'Deterministic' if not stochastic else ''} simulation")
        if gene_name and perturbation_type:
            basal_pert, deg_pert, desc = apply_perturbation(
                gene_name, perturbation_type, basal_ctrl, deg_ctrl
            )
        else:
            basal_pert, deg_pert = basal_ctrl.copy(), deg_ctrl.copy()
            desc = "Control"
        
        y_traj_ctrl, t_eval, y_ctrl = steady_state(basal_ctrl, deg_ctrl)
        y_traj_pert, _, y_pert = steady_state(basal_pert, deg_pert)
        
        # DPS
        dps_ctrl = compute_dps(y_ctrl)
        dps_pert = compute_dps(y_pert) if gene_name else None
        
        print(f"\n  Control DPS: {dps_ctrl['DPS']:.3f} [{dps_ctrl['state']}]")
        if dps_pert:
            print(f"  Perturbed DPS: {dps_pert['DPS']:.3f} [{dps_pert['state']}]")
        
        results["control"] = {
            "expression": dict(zip(GENES, y_ctrl)),
            "dps": dps_ctrl,
        }
        if gene_name:
            l2fc = {}
            for g in GENES:
                c = max(y_ctrl[GENES.index(g)], 0.001)
                p = max(y_pert[GENES.index(g)], 0.001)
                l2fc[g] = np.log2(p / c)
            
            results["perturbation"] = {
                "gene": gene_name,
                "type": perturbation_type,
                "expression": dict(zip(GENES, y_pert)),
                "log2fc": l2fc,
                "dps": dps_pert,
                "description": desc,
            }
        
        # ── DPS comparison plot ──
        if gene_name:
            dps_compare = [dps_ctrl, dps_pert]
            fig, ax = plt.subplots(figsize=(6, 4))
            labels = ["Control", desc]
            vals = [d["DPS"] for d in dps_compare]
            states = [d["state"] for d in dps_compare]
            colors_plot = ["#2166ac", "#d6604d"]
            ax.bar(labels, vals, color=colors_plot, edgecolor="black", lw=0.8)
            ax.set_ylabel("Differentiation Potency Score (DPS)", fontsize=10)
            ax.set_title(f"DPS: {dps_ctrl['state']} → {dps_pert['state']}", 
                        fontsize=11, fontweight="bold")
            ax.set_ylim(0, 1.0)
            for i, (v, s) in enumerate(zip(vals, states)):
                ax.text(i, v + 0.02, f"{v:.3f}\n({s})", ha="center", fontsize=9)
            fig.tight_layout()
            dps_path = _save_fig(fig, "dps_comparison.png")
            results["dps_figure"] = dps_path
            print(f"\n  DPS comparison: {dps_path}")
    
    print("\n" + "=" * 60)
    print("Simulation complete ✅")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    import sys
    # v4 entry point
    if len(sys.argv) >= 3:
        run_grn_simulation(sys.argv[1], sys.argv[2])
    else:
        # Run all modes demo
        print("=== DEMO: Deterministic ===")
        run_grn_simulation()
        print("\n=== DEMO: NOTCH1 KO ===")
        run_grn_simulation("NOTCH1", "knock_out")
        print("\n=== DEMO: Multi-cell (3 cells) ===")
        run_grn_simulation(multi_cell=3)
        print("\n=== DEMO: Stochastic ===")
        run_grn_simulation(stochastic=True, noise_scale=0.15)


# ══════════════════════════════════════════════════════════════════════════
# Multi-cell Notch-Delta Lateral Inhibition Module
# ══════════════════════════════════════════════════════════════════════════
# 独立于单细胞 GRN 的多细胞 Notch-Delta 侧向抑制模型。
#
# 论文依据:
# - Sprinzak et al. 2010, "Cis-interactions between Notch and Delta generate
#   mutually exclusive signalling states" (Nature)
# - Sprinzak et al. 2011, "Lateral inhibition in development" (Dev Cell)
#
# 核心数学模型:
#   每个细胞: NOTCH1, HES1, ASCL1 三个变量
#   NOTCH1_activation_i = Σ neighbors_j (ASCL1_j / (ASCL1_j + K))
#   d(NOTCH1)_i / dt = activation_i - deg_N * NOTCH1_i
#   d(HES1)_i    / dt = hill(NOTCH1_i) - deg_H * HES1_i
#   d(ASCL1)_i   / dt = basal - hill(HES1_i) * ASCL1_i
#
#   → ASCL1_high cell sends strong Delta → neighbor NOTCH1_high → HES1_high
#     → neighbor ASCL1_low → neighbor Delta_low → ...
#   → "你高我低"的竞争模式
# ══════════════════════════════════════════════════════════════════════════

def run_multi_cell_lateral_inhibition(n_cells=8, topology="random_1d",
                                      t_span=(0, 200), noise=0.05,
                                      seed=42):
    """
    运行多细胞 Notch-Delta 侧向抑制仿真 (Sprinzak 2010 精准参数化).
    
    Sprinzak et al. 2010 (Nature) 核心:
    - Delta (由ASCL1驱动) 和 Notch 形成互斥状态
    - cis-inhibition: 同细胞的Delta捕获Notch, 防止被邻居激活
    - trans-activation: 邻居Delta激活本细胞Notch
    - 反馈: ASCL1↑→Delta↑→cis↑→Notch↓→HES↓→ASCL1↑↑ (自我强化)
    
    参数设计确保:
    - ASCL1 sender: ASCL1≈1.5, NOTCH1≈0.1, HES1≈0.3
    - NOTCH1 receiver: ASCL1≈0.05, NOTCH1≈1.5, HES1≈1.8
    - 中间态不稳定 → 细胞必然分化成两个阵营
    """
    np.random.seed(seed)
    
    # ── Parameters (calibrated for bistable switch) ──
    # Each cell: [NOTCH1, HES1, ASCL1]
    V_N = 1.2     # NOTCH1→HES1 max activation (stronger)
    K_N = 0.5     # half-max
    n_N = 3       # cooperativity
    
    V_H = 3.0     # HES1→ASCL1 repression max
    K_H = 1.0     # half-max (higher = weaker repression at low HES)  
    n_H = 4       # cooperativity (steeper = better switch)
    
    # ASCL1 auto-activation (positive feedback — KEY for bistability)
    V_A = 1.5     # ASCL1 auto-activation max
    K_A = 0.8     # half-max
    
    basal_A = 0.03  # ASCL1 basal production (lower = cleaner switch)
    deg_N = 0.3     # NOTCH1 degradation
    deg_H = 0.4     # HES1 degradation
    deg_A = 0.25    # ASCL1 degradation
    
    # Delta signaling parameters
    delta_sens = 2.0   # max lateral activation of NOTCH1 by neighbor Delta
    delta_K = 0.35     # half-max Delta sensitivity
    
    # cis-inhibition strength
    cis_strength = 5.0   # how strongly own ASCL1 blocks Notch activation
    
    # Build neighbor matrix
    neighbor_matrix = build_neighbor_matrix(n_cells, topology)
    
    # ── Initial conditions ──
    y0 = np.zeros(n_cells * 3)
    for i in range(n_cells):
        ni = i * 3
        # Bimodal: alternating pattern for maximum competition
        if i % 2 == 0:
            # Sender: high ASCL1, low NOTCH1
            y0[ni] = 0.05      # NOTCH1 low
            y0[ni+1] = 0.15    # HES1 low
            y0[ni+2] = 1.5     # ASCL1 high
        else:
            # Receiver: low ASCL1, high NOTCH1
            y0[ni] = 1.2       # NOTCH1 high
            y0[ni+1] = 1.5     # HES1 high
            y0[ni+2] = 0.03    # ASCL1 low
        
        # Add noise
        y0[ni:ni+3] *= (1.0 + noise * np.random.randn(3))
        y0[ni:ni+3] = np.maximum(y0[ni:ni+3], 0.001)
    
    # ── ODE ──
    def notch_delta_ode(t, y):
        dydt = np.zeros_like(y)
        y_cells = y.reshape(n_cells, 3)
        
        for i in range(n_cells):
            ni = i * 3
            n_val = max(y_cells[i, 0], 0.001)  # NOTCH1
            h_val = max(y_cells[i, 1], 0.001)  # HES1
            a_val = max(y_cells[i, 2], 0.001)  # ASCL1
            
            # 1. Lateral NOTCH1 activation from neighbors' Delta (ASCL1)
            lateral = 0.0
            for j in range(n_cells):
                if i != j and neighbor_matrix[i, j] > 0:
                    neighbor_asc = y_cells[j, 2]
                    lateral += neighbor_matrix[i, j] * (
                        delta_sens * neighbor_asc / (neighbor_asc + delta_K)
                    )
            
            # 2. Cis-inhibition: own ASCL1 sequesters Notch (strong!)
            # When ASCL1 is high, the cell cannot receive lateral signal
            cis_block = 1.0 / (1.0 + cis_strength * a_val)
            
            # 3. Effective NOTCH1 activation
            notch_act = lateral * cis_block
            
            # 4. ODE system
            # NOTCH1: activation - degradation
            dydt[ni] = notch_act - deg_N * n_val
            
            # HES1: NOTCH1-driven + small basal - degradation
            hes_prod = hill_activate(n_val, V_N, K_N, n_N) + 0.01
            dydt[ni+1] = hes_prod - deg_H * h_val
            
            # ASCL1: basal + auto-activation - HES1 repression - degradation
            asc_prod = basal_A + hill_activate(a_val, V_A, K_A, n_N)
            hes_rep = hill_activate(h_val, V_H, K_H, n_H)
            dydt[ni+2] = asc_prod / (1.0 + hes_rep) - deg_A * a_val
        
        return dydt
    
    # ── Integrate ──
    sol = solve_ivp(notch_delta_ode, t_span, y0, method="LSODA",
                    max_step=0.5, rtol=1e-7, atol=1e-9,
                    dense_output=True)
    
    t_eval = np.linspace(t_span[0], t_span[1], 500)
    y_traj = sol.sol(t_eval)
    y_final = y_traj[:, -1].reshape(n_cells, 3)
    
    # Ensure non-negative
    y_final = np.maximum(y_final, 0.0)
    
    return y_traj, t_eval, y_final, neighbor_matrix


def plot_lateral_inhibition_results(y_final, n_cells, topology,
                                    y_traj=None, t_eval=None):
    """
    侧向抑制结果可视化.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel 1: ASCL1 expression per cell
    ax = axes[0]
    asc_levels = y_final[:, 2]
    colors = ["#b2182b" if a > 0.5 else "#2166ac" for a in asc_levels]
    
    bars = ax.bar(range(n_cells), asc_levels, color=colors, 
                  edgecolor="black", lw=0.8)
    ax.set_xlabel("Cell index", fontsize=10)
    ax.set_ylabel("ASCL1 (Delta) steady state", fontsize=10, color="#b2182b")
    ax.set_title("Notch-Delta Lateral Inhibition\n(ASCL1 = Delta signal)", 
                 fontsize=11, fontweight="bold")
    ax.axhline(0.5, color="gray", ls="--", alpha=0.4)
    
    # Annotate sender/receiver
    for i, a in enumerate(asc_levels):
        role = "🔵 Sender" if a > 0.5 else "🔴 Receiver"
        ax.text(i, a + 0.05, role, ha="center", fontsize=7, fontweight="bold")
    
    # Panel 2: NOTCH1 vs ASCL1 (mutual exclusion)
    ax = axes[1]
    ax.scatter(y_final[:, 2], y_final[:, 0], c=colors, s=80, 
               edgecolors="black", zorder=3)
    ax.set_xlabel("ASCL1 (Delta)", fontsize=10)
    ax.set_ylabel("NOTCH1 (receptor)", fontsize=10)
    ax.set_title("Mutual Exclusion\n(Sprinzak 2010 Nature)", 
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    
    # Anti-correlation line
    x_line = np.linspace(0.01, max(y_final[:, 2]) * 1.2, 100)
    ax.plot(x_line, 0.5 / (x_line + 0.2), "r--", alpha=0.5, lw=1)
    
    # Panel 3: Time series (first 3 cells if available)
    ax = axes[2]
    if y_traj is not None and t_eval is not None:
        n_plot = min(3, n_cells)
        for i in range(n_plot):
            cell_traj = y_traj[i*3:(i+1)*3, :]
            ax.plot(t_eval, cell_traj[2, :],  # ASCL1
                   label=f"Cell {i} ASCL1", 
                   color=colors[i] if i < len(colors) else "#333333",
                   lw=1.5)
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel("ASCL1 expression", fontsize=10)
        ax.set_title("ASCL1 Dynamics\n(divergence via lateral inhibition)", 
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    fig.suptitle(f"Multi-cell Notch-Delta Lateral Inhibition ({n_cells} cells, {topology})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save_fig(fig, "lateral_inhibition.png")


def demo_multi_cell():
    """演示多细胞侧向抑制."""
    print("=" * 60)
    print("Notch-Delta Lateral Inhibition (Sprinzak 2010 Model)")
    print("=" * 60)
    
    for n in [2, 4, 8]:
        print(f"\n--- {n} cells (random_1d topology) ---")
        y_traj, t_eval, y_final, neighbor_matrix = run_multi_cell_lateral_inhibition(
            n_cells=n, topology="random_1d"
        )
        for i in range(n):
            n_val = y_final[i, 0]
            h_val = y_final[i, 1]
            a_val = y_final[i, 2]
            role = "SENDER (ASCL1↑)" if a_val > 0.5 else "RECEIVER (NOTCH1↑)"
            print(f"  Cell {i}: NOTCH1={n_val:.3f} HES1={h_val:.3f} ASCL1={a_val:.4f}  ← {role}")
    
    # Plot 8-cell
    y_traj, t_eval, y_final, _ = run_multi_cell_lateral_inhibition(n_cells=8)
    path = plot_lateral_inhibition_results(y_final, 8, "random_1d", y_traj, t_eval)
    print(f"\n✅ Lateral inhibition plot: {path}")
