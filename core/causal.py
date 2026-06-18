"""VirtualCell-Agent: 因果推断模块（轻量级 SCM-based）"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CausalEdge:
    source: str
    target: str
    sign: str  # "+" activation, "-" inhibition
    evidence: str  # literature evidence reference


# ─── Neuro stem cell causal graph (SCM) ────────────────────────────
# Edges represent known causal relationships from signaling pathways.

NSC_CAUSAL_GRAPH = {
    # Notch pathway
    "NOTCH1": [CausalEdge("NOTCH1", "HES1", "+", "Notch→RBPJ→HES1 promoter"),
               CausalEdge("NOTCH1", "HES5", "+", "Notch→RBPJ→HES5 promoter"),
               CausalEdge("NOTCH1", "HEY1", "+", "Notch→RBPJ→HEY1 promoter")],
    "HES1": [CausalEdge("HES1", "ASCL1", "-", "HES1 represses ASCL1 via N-box"),
             CausalEdge("HES1", "NEUROG2", "-", "HES1 represses NEUROG2")],
    "HES5": [CausalEdge("HES5", "ASCL1", "-", "HES5 represses ASCL1"),
             CausalEdge("HES5", "DLL1", "-", "HES5 represses DLL1 (lateral inhibition)")],
    "ASCL1": [CausalEdge("ASCL1", "NEUROG2", "+", "ASCL1 activates NEUROG2"),
              CausalEdge("ASCL1", "NEUROD1", "+", "ASCL1→NEUROD1 cascade"),
              CausalEdge("ASCL1", "DCX", "+", "ASCL1 activates neuronal differentiation program")],

    # Wnt pathway
    "WNT3A": [CausalEdge("WNT3A", "CTNNB1", "+", "Wnt→Frizzled→β-catenin stabilization")],
    "WNT7A": [CausalEdge("WNT7A", "CTNNB1", "+", "Wnt→Frizzled→β-catenin stabilization")],
    "GSK3B": [CausalEdge("GSK3B", "CTNNB1", "-", "GSK3β phosphorylates β-catenin→degradation")],
    "CTNNB1": [CausalEdge("CTNNB1", "MYC", "+", "β-catenin/TCF→MYC promoter"),
               CausalEdge("CTNNB1", "CCND1", "+", "β-catenin/TCF→CCND1 promoter"),
               CausalEdge("CTNNB1", "NEUROG2", "+", "β-catenin/TCF→NEUROG2"),
               CausalEdge("CTNNB1", "AXIN2", "+", "Negative feedback: β-catenin→AXIN2")],

    # SHH pathway
    "SHH": [CausalEdge("SHH", "PTCH1", "+", "SHH induces PTCH1 (transcriptional target)"),
            CausalEdge("SHH", "GLI1", "+", "SHH→SMO→GLI1 activation"),
            CausalEdge("PTCH1", "SMO", "-", "PTCH1 inhibits SMO in absence of SHH")],
    "SMO": [CausalEdge("SMO", "GLI1", "+", "SMO→SUFU release→GLI1 nuclear entry"),
            CausalEdge("SMO", "GLI2", "+", "SMO→GLI2 activator form")],
    "GLI1": [CausalEdge("GLI1", "MYCN", "+", "GLI1→MYCN promoter"),
             CausalEdge("GLI1", "CCND2", "+", "GLI1→CCND2 promoter"),
             CausalEdge("GLI1", "PTCH1", "+", "GLI1→PTCH1 (negative feedback)")],

    # BMP pathway
    "BMP4": [CausalEdge("BMP4", "SMAD1", "+", "BMP→BMPR→SMAD1 phosphorylation"),
             CausalEdge("BMP4", "SMAD5", "+", "BMP→BMPR→SMAD5 phosphorylation")],
    "BMPR1A": [CausalEdge("BMPR1A", "SMAD1", "+", "BMPR1A phosphorylates SMAD1/5")],
    "SMAD1": [CausalEdge("SMAD1", "ID1", "+", "pSMAD1/SMAD4→ID1 promoter"),
              CausalEdge("SMAD1", "ID3", "+", "pSMAD1/SMAD4→ID3 promoter")],
    "SMAD4": [CausalEdge("SMAD4", "ID1", "+", "Co-SMAD required for ID1 activation")],
    "ID1": [CausalEdge("ID1", "ASCL1", "-", "ID1 sequesters E proteins, blocks ASCL1"),
            CausalEdge("ID1", "NEUROG2", "-", "ID1 sequesters E proteins, blocks NEUROG2"),
            CausalEdge("ID1", "GFAP", "+", "ID1 promotes astrocyte differentiation")],

    # MAPK pathway
    "EGFR": [CausalEdge("EGFR", "MAPK1", "+", "EGFR→RAS→RAF→MEK→ERK")],
    "FGFR1": [CausalEdge("FGFR1", "MAPK1", "+", "FGFR→RAS→RAF→MEK→ERK")],
    "MAPK1": [CausalEdge("MAPK1", "MYC", "+", "ERK phosphorylates and stabilizes MYC"),
              CausalEdge("MAPK1", "ELK1", "+", "ERK→ELK1→immediate early genes")],

    # Hippo pathway
    "YAP1": [CausalEdge("YAP1", "CYR61", "+", "YAP/TEAD→CYR61 promoter"),
             CausalEdge("YAP1", "CTGF", "+", "YAP/TEAD→CTGF promoter"),
             CausalEdge("YAP1", "BIRC5", "+", "YAP/TEAD→BIRC5 (survivin)"),
             CausalEdge("YAP1", "SOX2", "+", "YAP maintains SOX2 expression")],
    "LATS1": [CausalEdge("LATS1", "YAP1", "-", "LATS phosphorylates YAP→cytoplasmic retention")],
    "TEAD1": [CausalEdge("TEAD1", "CYR61", "+", "TEAD is the DNA-binding partner for YAP")],

    # Pluripotency
    "SOX2": [CausalEdge("SOX2", "POU5F1", "+", "SOX2-OCT4 cooperative binding"),
             CausalEdge("SOX2", "NES", "+", "SOX2 maintains Nestin expression"),
             CausalEdge("SOX2", "GFAP", "-", "SOX2 represses GFAP (maintains stemness)")],
    "PAX6": [CausalEdge("PAX6", "SOX2", "+", "PAX6 regulates SOX2 in cortical NSCs"),
             CausalEdge("PAX6", "GFAP", "-", "PAX6 represses astrocyte fate")],
    "NEUROG2": [CausalEdge("NEUROG2", "NEUROD1", "+", "NEUROG2→NEUROD1 cascade"),
                CausalEdge("NEUROG2", "DCX", "+", "NEUROG2 activates DCX"),
                CausalEdge("NEUROG2", "HES5", "-", "NEUROG2 represses HES5")],
    "NEUROD1": [CausalEdge("NEUROD1", "DCX", "+", "NEUROD1→DCX"),
                CausalEdge("NEUROD1", "TUBB3", "+", "NEUROD1→TUBB3"),
                CausalEdge("NEUROD1", "RBFOX3", "+", "NEUROD1→NeuN")],

    # Disease related
    "PTEN": [CausalEdge("PTEN", "MTOR", "-", "PTEN dephosphorylates PIP3→AKT↓→mTOR↓")],
    "MTOR": [CausalEdge("MTOR", "CCND1", "+", "mTOR→S6K→CCND1 translation"),
             CausalEdge("MTOR", "MYC", "+", "mTOR→4E-BP→MYC translation")],
    "TP53": [CausalEdge("TP53", "CDKN1A", "+", "p53→p21 transcription"),
             CausalEdge("TP53", "CCND1", "-", "p53 represses CCND1"),
             CausalEdge("TP53", "MYC", "-", "p53 represses MYC")],
    "TGFB1": [CausalEdge("TGFB1", "SMAD2", "+", "TGF-β→TβR→SMAD2 phosphorylation"),
              CausalEdge("TGFB1", "SMAD3", "+", "TGF-β→TβR→SMAD3 phosphorylation")],
}

# ─── Downstream markers for each cell state ─────────────────────────

CELL_STATE_MARKERS = {
    "self_renewal": ["SOX2", "NES", "PROM1", "TERT", "MYC", "CCND1"],
    "neural_differentiation": ["DCX", "TUBB3", "RBFOX3", "NEUROD1", "MAP2", "STMN2"],
    "astrocyte_differentiation": ["GFAP", "S100B", "ALDH1L1", "AQP4"],
    "oligodendrocyte_differentiation": ["MBP", "OLIG2", "SOX10", "PDGFRA", "PLP1"],
    "apoptosis": ["BAX", "BBC3", "CASP3", "CASP9"],
    "proliferation": ["MKI67", "CCND1", "MYC", "PCNA"],
    "quiescence": ["GFAP", "NES", "SOX2", "EGFR(low)", "CDKN1A"],
}


def infer_causal_pathways(target_gene: str, perturbation: str) -> list[dict]:
    """
    Walk the causal graph from the target gene outward and predict
    upstream and downstream effects.

    Returns a list of predicted causal chains.
    """
    visited = set()
    chains = []

    def dfs(gene: str, depth: int = 0, path: Optional[list[str]] = None):
        if depth > 3 or gene in visited:
            return
        if path is None:
            path = []
        visited.add(gene)
        current_path = path + [gene]

        if gene in NSC_CAUSAL_GRAPH:
            for edge in NSC_CAUSAL_GRAPH[gene]:
                if edge.target not in visited:
                    chains.append({
                        "path": " → ".join(current_path + [edge.target]),
                        "sign": edge.sign,
                        "source_gene": gene,
                        "target_gene": edge.target,
                        "evidence": edge.evidence,
                        "predicted_effect": (
                            "upregulated" if edge.sign == "+" else "downregulated"
                        ) if perturbation in ("overexpression", "drug_agonist") else (
                            "downregulated" if edge.sign == "+" else "upregulated"
                        ) if perturbation == "knock_out" else "unknown",
                    })
                    dfs(edge.target, depth + 1, current_path)

    dfs(target_gene)
    return chains


def predict_cell_fate_shift(chains: list[dict], perturbation: str) -> dict:
    """
    Based on causal chains, predict the overall cell fate shift
    (self-renewal vs differentiation).
    """
    scores = {
        "self_renewal": 0,
        "neural_diff": 0,
        "astro_diff": 0,
        "oligo_diff": 0,
        "apoptosis": 0,
    }

    for chain in chains:
        target = chain["target_gene"]
        effect = chain["predicted_effect"]
        sign = chain["sign"]

        for fate, markers in CELL_STATE_MARKERS.items():
            if target in markers:
                delta = 1.0 if effect == "upregulated" else -1.0
                if fate == "self_renewal":
                    scores["self_renewal"] += delta
                elif fate == "neural_differentiation":
                    scores["neural_diff"] += delta
                elif fate == "astrocyte_differentiation":
                    scores["astro_diff"] += delta
                elif fate == "oligodendrocyte_differentiation":
                    scores["oligo_diff"] += delta
                elif fate == "apoptosis":
                    scores["apoptosis"] += delta

    # Normalize
    max_score = max(abs(v) for v in scores.values()) or 1
    normalized = {k: round(v / max_score, 2) for k, v in scores.items()}

    # Determine major shift
    top_fate = max(normalized, key=normalized.get) if max(normalized.values()) > 0 else "uncertain"
    fate_labels = {
        "self_renewal": "自我更新维持 ↑",
        "neural_diff": "神经分化 ↑",
        "astro_diff": "星形胶质细胞分化 ↑",
        "oligo_diff": "少突胶质细胞分化 ↑",
        "apoptosis": "凋亡 ↑",
    }

    return {
        "scores": normalized,
        "top_fate": fate_labels.get(top_fate, "不确定"),
        "summary": _generate_fate_summary(normalized, top_fate, perturbation),
    }


def _generate_fate_summary(scores: dict, top_fate: str, perturbation: str) -> str:
    """Generate a human-readable summary of cell fate prediction."""
    up_fates = [k for k, v in scores.items() if v > 0.3]
    down_fates = [k for k, v in scores.items() if v < -0.3]

    labels = {
        "self_renewal": "自我更新",
        "neural_diff": "神经分化",
        "astro_diff": "星形胶质分化",
        "oligo_diff": "少突胶质分化",
        "apoptosis": "凋亡",
    }

    parts = []
    if up_fates:
        parts.append("促进: " + ", ".join(labels.get(f, f) for f in up_fates))
    if down_fates:
        parts.append("抑制: " + ", ".join(labels.get(f, f) for f in down_fates))
    if not parts:
        parts.append("无明显偏倚")

    return " | ".join(parts)


def format_causal_summary(target_gene: str, perturbation: str) -> str:
    """Format causal inference results as Markdown."""
    chains = infer_causal_pathways(target_gene, perturbation)
    fate = predict_cell_fate_shift(chains, perturbation)

    lines = [
        "### 🔗 因果通路推断",
        "",
        f"靶点 **{target_gene}** 的因果下游效应链（{len(chains)} 条）：\n",
    ]

    # Show top 15 chains
    for i, ch in enumerate(chains[:15], 1):
        arrow = "→" if ch["sign"] == "+" else "⊣"
        lines.append(f"{i}. {ch['path']}  [{arrow}] {ch['predicted_effect']}")
    if len(chains) > 15:
        lines.append(f"... 还有 {len(chains)-15} 条因果链（折叠）")

    lines += [
        "",
        "### 🧫 细胞命运预测",
        "",
        "| 命运维度 | 偏向分数 |",
        "|---------|---------|",
    ]
    for fate_name, score in fate["scores"].items():
        bar = "█" * int(abs(score) * 10) if score > 0 else "░" * int(abs(score) * 10)
        if score < 0:
            bar = "-" + bar
        elif score == 0:
            bar = "·"
        lines.append(f"| {fate_name:20s} | {score:+.2f} {bar} |")

    lines += [
        "",
        f"**主要趋势**: {fate['summary']}",
        "",
        "> ⚠️ 因果推断基于已知文献构建的因果图，仅指示方向性，不提供定量预测。",
        "",
    ]

    return "\n".join(lines)
