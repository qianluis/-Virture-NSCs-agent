"""VirtualCell-Agent v5.0: 统一虚拟细胞引擎

整合六大模块：
  morphology — 4 种细胞状态的逼真形态生成
  grn_model — 22 基因的GRN (v4升级: 随机/侧向抑制/DPS/分岔)
  landscape — Waddington表观遗传景观 (Wang 2022, Bhatt 2020)
  morphogen_gradient — 形态发生素梯度生态位 (Ribes & Briscoe 2009)
  scseq_projection — scRNA-seq 投影验证 (Llorens-Bobadilla 2015)
  niche_model — SVZ 生态位 ABM 空间群体动力学
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("VirtualCell")

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── 1. Morphology Simulation ───────────────────────────────────────


def simulate_morphology(cell_state: str = "qNSC") -> dict:
    """
    Generate morphology image and return morphological features.

    Args:
        cell_state: "qNSC" | "aNSC" | "TAP" | "Neuron"

    Returns:
        morphology features dict
    """
    features = {
        "qNSC": {
            "state": "Quiescent NSC", "soma_area_um2": 102.1,
            "longest_process_um": 70.0, "num_short_processes": 5,
            "num_branches": 0, "description": "Small soma, 1 long radial fiber with endfoot",
        },
        "aNSC": {
            "state": "Activated NSC", "soma_area_um2": 251.3,
            "longest_process_um": 86.8, "num_short_processes": 10,
            "num_branches": 2, "description": "Large soma, apical+basal processes, short filopodia",
        },
        "TAP": {
            "state": "Transit Amplifying Progenitor", "soma_area_um2": 188.5,
            "longest_process_um": 7.5, "num_short_processes": 4,
            "num_branches": 0, "description": "Near-round, very short stub processes",
        },
        "Neuron": {
            "state": "Differentiated Neuron", "soma_area_um2": 311.0,
            "longest_process_um": 95.0, "num_short_processes": 12,
            "num_branches": 33, "description": "Large soma, axon + branched dendritic tree",
        },
    }

    # Generate image
    try:
        from core.morphology import plot_nsc_morphology
        png_path = str(OUTPUT_DIR / f"morphology_{cell_state}.png")
        plot_nsc_morphology(output_path=png_path)
        features["image_path"] = png_path
    except Exception as e:
        logger.warning(f"Morphology plot failed: {e}")
        features["image_path"] = ""

    return features.get(cell_state, features["qNSC"])


def plot_all_morphologies():
    """Generate combined 4-panel morphology figure."""
    try:
        from core.morphology import plot_nsc_morphology
        png_path = str(OUTPUT_DIR / "nsc_morphology_all.png")
        plot_nsc_morphology(output_path=png_path)
        return png_path
    except Exception as e:
        logger.warning(f"Full morphology plot failed: {e}")
        return ""


# ─── 2. GRN Simulation ──────────────────────────────────────────────


def simulate_grn(target_gene: str = "", perturbation: str = "") -> dict:
    """
    Run GRN ODE simulation and return results.

    Args:
        target_gene: e.g. "NOTCH1", "SOX2"
        perturbation: "knock_out" | "overexpression" | "drug_inhibit" | ""

    Returns:
        dict with steady state values, predicted up/down genes, image paths
    """
    result = {
        "steady_state_control": {},
        "steady_state_perturbed": {},
        "fold_changes": {},
        "upregulated": [],
        "downregulated": [],
        "heatmap_path": "",
        "timeseries_path": "",
        "bar_path": "",
        "success": False,
    }

    try:
        from core import grn_model

        # Run simulation (returns dict with control_expression, perturbed_expression, figure_paths)
        sim_result = grn_model.run_grn_simulation(
            gene_name=target_gene if target_gene else None,
            perturbation_type=perturbation if perturbation else None,
        )

        ctrl_expr = sim_result.get("control_expression", {})
        pert_expr = sim_result.get("perturbed_expression", {}) or {}
        result["steady_state_control"] = {k: round(float(v), 4) for k, v in ctrl_expr.items()}
        result["steady_state_perturbed"] = {k: round(float(v), 4) for k, v in pert_expr.items()}

        # Compute fold changes
        fold_changes = {}
        for gene in ctrl_expr:
            ctrl_val = ctrl_expr.get(gene, 1e-10)
            pert_val = pert_expr.get(gene, ctrl_val)
            if abs(ctrl_val) > 1e-10:
                fc = (pert_val - ctrl_val) / (abs(ctrl_val) + 1e-10)
                fold_changes[gene] = round(fc, 4)
        result["fold_changes"] = fold_changes

        # Classify — fold-change model produces biologically meaningful effect sizes
        # |FC|>0.10 is a reliable signal
        for gene, fc in fold_changes.items():
            if fc > 0.10:
                result["upregulated"].append(gene)
            elif fc < -0.10:
                result["downregulated"].append(gene)

        # Image paths from simulation
        fig_paths = sim_result.get("figure_paths", {})
        result["heatmap_path"] = fig_paths.get("diff_heatmap", str(OUTPUT_DIR / "diff_heatmap.png"))
        result["timeseries_path"] = fig_paths.get("time_series", str(OUTPUT_DIR / "time_series.png"))
        result["bar_path"] = fig_paths.get("bar_comparison", str(OUTPUT_DIR / "bar_comparison.png"))
        result["success"] = True

    except Exception as e:
        logger.warning(f"GRN simulation failed: {e}")

    return result


# ─── 3. Niche (ABM) Simulation ──────────────────────────────────────


def simulate_niche(steps: int = 100, grid_size: int = 64) -> dict:
    """
    Run SVZ niche ABM simulation.

    Returns:
        dict with population stats, image paths
    """
    result = {
        "population_timeline": {},
        "final_population": {},
        "initial_grid": "",
        "final_grid": "",
        "population_plot": "",
        "snapshots": "",
        "total_cells": 0,
        "success": False,
    }

    try:
        from core import niche_model

        # Run simulation
        stats = niche_model.run_niche_simulation(
            grid_size=grid_size,
            steps=steps,
            output_dir=str(OUTPUT_DIR),
        )

        if stats:
            result["initial_grid"] = str(OUTPUT_DIR / "initial_grid.png")
            result["final_grid"] = str(OUTPUT_DIR / "final_grid.png")
            result["snapshots"] = str(OUTPUT_DIR / "niche_snapshots.png")
            result["population_plot"] = str(OUTPUT_DIR / "population_timeline.png")

            if hasattr(stats, 'get'):
                result["final_population"] = stats
                result["total_cells"] = sum(stats.values()) if isinstance(stats, dict) else 0
            result["success"] = True

    except Exception as e:
        logger.warning(f"Niche simulation failed: {e}")
        logger.exception("Niche error details:")

    return result


# ─── 4. Unified Analysis ────────────────────────────────────────────


def analyze_perturbation(target_gene: str, perturbation: str) -> dict:
    """
    Run the full virtual cell analysis for a given perturbation.
    Integrates GRN + Morphology + Niche predictions.

    Returns:
        Complete analysis result dict
    """
    print(f"[VirtualCell] 🧬 虚拟细胞分析启动: {target_gene} {perturbation}")
    result = {
        "target_gene": target_gene,
        "perturbation": perturbation,
        "timestamp": datetime.now().isoformat(),
        "grn": {},
        "morphology": {},
        "niche": {},
        "cell_fate_prediction": {},
        "summary": "",
    }

    # 1. GRN
    print(f"[VirtualCell] 🔬 GRN 仿真中...")
    grn_result = simulate_grn(target_gene, perturbation)
    result["grn"] = grn_result

    # 2. Cell fate prediction from GRN
    up = set(grn_result.get("upregulated", []))
    down = set(grn_result.get("downregulated", []))

    # Map genes to cell states
    stemness_genes = {"SOX2", "NES", "PROM1", "TERT", "MYC", "CCND1"}
    neural_genes = {"DCX", "TUBB3", "RBFOX3", "NEUROD1", "MAP2", "STMN2"}
    astro_genes = {"GFAP", "S100B", "ALDH1L1"}
    oligo_genes = {"MBP", "OLIG2", "SOX10", "PLP1"}
    proliferation_genes = {"MKI67", "CCND1", "MYC", "CCND2", "PCNA"}

    fate_scores = {
        "self_renewal": len(stemness_genes & up) - len(stemness_genes & down),
        "neural_differentiation": len(neural_genes & up) - len(neural_genes & down),
        "astrocyte_differentiation": len(astro_genes & up) - len(astro_genes & down),
        "oligodendrocyte_differentiation": len(oligo_genes & up) - len(oligo_genes & down),
        "proliferation": len(proliferation_genes & up) - len(proliferation_genes & down),
    }

    max_fate = max(fate_scores, key=fate_scores.get)
    result["cell_fate_prediction"] = {
        "scores": fate_scores,
        "dominant_fate": max_fate,
        "interpretation": {
            "self_renewal": "自我更新",
            "neural_differentiation": "神经分化",
            "astrocyte_differentiation": "星形胶质分化",
            "oligodendrocyte_differentiation": "少突胶质分化",
            "proliferation": "增殖",
        }.get(max_fate, "不确定"),
    }

    # 3. Morphology prediction
    state_map = {
        "self_renewal": "qNSC" if any(g in down for g in proliferation_genes) else "aNSC",
        "neural_differentiation": "Neuron",
        "astrocyte_differentiation": "aNSC",  # astro diff from aNSC
        "oligodendrocyte_differentiation": "aNSC",
        "proliferation": "aNSC",
    }
    predicted_state = state_map.get(max_fate, "aNSC")
    result["morphology"] = simulate_morphology(predicted_state)
    result["morphology"]["predicted_state"] = predicted_state

    # 4. Niche (only if no perturbation or default)
    if not target_gene:
        print(f"[VirtualCell] 🧫 生态位仿真中...")
        niche_result = simulate_niche(steps=50)
        result["niche"] = niche_result

    # 5. Summary
    up_str = ", ".join(result["grn"].get("upregulated", [])[:8]) or "none"
    down_str = ", ".join(result["grn"].get("downregulated", [])[:8]) or "none"
    result["summary"] = (
        f"靶点 **{target_gene}** {perturbation} 分析完成。 "
        f"GRN 预测上调基因: {up_str}；下调: {down_str}。"
        f"细胞命运偏向: **{result['cell_fate_prediction']['interpretation']}**。"
        f"预测形态: **{predicted_state}**。"
    )

    print(f"[VirtualCell] ✅ 分析完成")
    return result


# ─── 5. v5.0 Waddington Landscape Analysis ────────────────────────────


def simulate_landscape() -> dict:
    """
    Run Waddington epigenetic landscape analysis.
    
    Paper: Wang 2022 (WIRES), Bhatt 2020 (Nat Rev Genet)
    """
    result = {"success": False, "figure": "", "attractors": [], "metrics": {}}
    try:
        from core import landscape
        landscape_results = landscape.run_landscape_analysis()
        result["figure"] = landscape_results.get("landscape_figure", "")
        result["bifurcation_figure"] = landscape_results.get("bifurcation_figure", "")
        result["attractors"] = landscape_results.get("attractors_by_condition", [])
        result["success"] = True
    except Exception as e:
        logger.warning(f"Landscape analysis failed: {e}")
    return result


# ─── 6. v5.0 Morphogen Gradient Niche Simulation ──────────────────────


def simulate_gradient_niche(n_cells: int = 20) -> dict:
    """
    Run morphogen gradient niche simulation.
    
    Paper: Ribes & Briscoe 2009 (Nat Rev Neurosci), Karr 2012 (Cell)
    """
    result = {"success": False, "figure": "", "final_stats": {}}
    try:
        from core import morphogen_gradient
        niche_results = morphogen_gradient.run_niche_gradient_simulation(
            n_cells=n_cells, n_steps=15
        )
        result["figure"] = niche_results.get("figures", ("", ""))[1]
        result["gradient_figure"] = niche_results.get("figures", ("", ""))[0]
        result["final_stats"] = niche_results["niche"]._get_stats()
        result["success"] = True
    except Exception as e:
        logger.warning(f"Gradient niche failed: {e}")
    return result


# ─── 7. v5.0 scRNA-seq Projection ─────────────────────────────────────


def simulate_scseq_projection(n_cells: int = 500) -> dict:
    """
    Run scRNA-seq projection analysis.
    
    Paper: Llorens-Bobadilla 2015 (Cell Stem Cell)
    """
    result = {"success": False, "figures": [], "metrics": {}}
    try:
        from core import scseq_projection
        scseq_results = scseq_projection.run_scseq_analysis(
            n_cells=n_cells
        )
        result["figures"] = scseq_results.get("figures", [])
        result["metrics"] = scseq_results.get("metrics", {})
        result["success"] = True
    except Exception as e:
        logger.warning(f"scRNA-seq projection failed: {e}")
    return result


def format_virtual_cell_report(result: dict) -> str:
    """Format virtual cell analysis result as Markdown report section."""
    lines = [
        "---",
        "## 🧬 虚拟细胞综合分析",
        "",
        f"> 靶点: **{result.get('target_gene', '?')}** | "
        f"干预: **{result.get('perturbation', '?')}**",
        "",
        "### 1️⃣ 基因调控网络 (GRN)",
        "",
    ]

    grn = result.get("grn", {})
    if grn.get("success"):
        lines.append(f"- **22 基因 ODE模型**, Hill函数调控 (n=2)")
        lines.append(f"- 控制稳态 vs 扰动稳态对比\n")
        lines.append("**上调基因**: " + ", ".join(grn.get("upregulated", [])[:10]) or "无")
        lines.append("**下调基因**: " + ", ".join(grn.get("downregulated", [])[:10]) or "无\n")

        if grn.get("heatmap_path"):
            lines.append(f"![差异热图]({Path(grn['heatmap_path']).name})\n")
        if grn.get("timeseries_path"):
            lines.append(f"![时间序列]({Path(grn['timeseries_path']).name})\n")
    else:
        lines.append("⚠️ GRN 仿真未运行\n")

    # Fate
    fate = result.get("cell_fate_prediction", {})
    if fate:
        lines += [
            "### 2️⃣ 细胞命运预测",
            "",
            "| 维度 | 偏向分数 |",
            "|------|---------|",
        ]
        for dim, score in fate.get("scores", {}).items():
            bar = "█" * max(0, min(10, score + 5))
            lines.append(f"| {dim:30s} | {score:+d} {bar} |")
        lines.append(f"\n**主导命运**: {fate.get('interpretation', '?')}\n")

    # Morphology
    morph = result.get("morphology", {})
    if morph:
        lines += [
            "### 3️⃣ 细胞形态预测",
            "",
            f"- **预测细胞状态**: {morph.get('predicted_state', '?')}",
            f"- **描述**: {morph.get('description', '')}",
            f"- **细胞体面积**: {morph.get('soma_area_um2', '?')} µm²",
            f"- **最长突起**: {morph.get('longest_process_um', '?')} µm",
            f"- **突起数**: {morph.get('num_short_processes', '?')}",
            "",
        ]

    # Niche
    niche = result.get("niche", {})
    if niche.get("success"):
        lines += [
            "### 4️⃣ SVZ 生态位群体动力学",
            "",
        ]
        if niche.get("snapshots"):
            lines.append(f"![生态位演化]({Path(niche['snapshots']).name})\n")
        if niche.get("population_plot"):
            lines.append(f"![群体变化]({Path(niche['population_plot']).name})\n")

    # Summary
    lines += [
        "### 📋 综合摘要",
        "",
        result.get("summary", ""),
        "",
    ]

    return "\n".join(lines)


# ─── CLI ────────────────────────────────────────────────────────────


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VirtualCell Engine")
    parser.add_argument("--target", "-t", default="", help="Target gene")
    parser.add_argument("--perturbation", "-p", default="", help="knock_out | overexpression | drug_inhibit")
    parser.add_argument("--mode", "-m", default="full",
                        choices=["full", "morphology", "grn", "niche",
                                 "landscape", "gradient", "scseq"])
    args = parser.parse_args()

    if args.mode == "morphology":
        plot_all_morphologies()
        print(f"Morphology saved to {OUTPUT_DIR}/nsc_morphology_all.png")
    elif args.mode == "grn":
        result = simulate_grn(args.target, args.perturbation)
        print(json.dumps({k: v for k, v in result.items() if k != 'fold_changes'}, indent=2))
    elif args.mode == "niche":
        result = simulate_niche(steps=50)
        print(json.dumps(result, indent=2, default=str))
    elif args.mode == "landscape":
        print("[VirtualCell] 🌄 Waddington Landscape Analysis...")
        result = simulate_landscape()
        print(f"  Figure: {result.get('figure', 'N/A')}")
    elif args.mode == "gradient":
        print("[VirtualCell] 🧪 Morphogen Gradient Niche...")
        result = simulate_gradient_niche()
        print(f"  Figure: {result.get('figure', 'N/A')}")
        print(f"  Stats: {result.get('final_stats', {})}")
    elif args.mode == "scseq":
        print("[VirtualCell] 🔬 scRNA-seq Projection...")
        result = simulate_scseq_projection()
        print(f"  Figures: {result.get('figures', [])}")
        metrics = result.get("metrics", {})
        print(f"  Pearson r: {metrics.get('pearson_r', 'N/A')}")
        print(f"  AUC: {metrics.get('auc', 'N/A')}")
    else:
        result = analyze_perturbation(args.target, args.perturbation)
        print(format_virtual_cell_report(result))


if __name__ == "__main__":
    main()
