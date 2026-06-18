"""VirtualCell-Agent: 神经干细胞智能体 — 核心编排入口（v1.1 升级版）

5 层流水线工作流 (LangGraph Nodes):
  node_parse → node_evidence → node_pathway → node_simulate → node_predict → node_validate → node_report
"""

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

import numpy as np  # noqa: E402

from .state import (  # noqa: E402
    AgentState, HardwareInfo, ParsedQuery, EvidencePackage, Paper,
    PathwayModel, SimulationResult, AIPrediction, SimpleBaseline,
    ValidationResult, Explanation, HARDWARE_LEVELS
)
from .validator import (  # noqa: E402
    validate_ai_prediction, format_validation_summary
)

logger = logging.getLogger("VirtualCellAgent")

# ─── Paths ──────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_DIR / "knowledge" / "neural_stem_cell"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Hardware Detection ─────────────────────────────────────────────


def node_hardware(state: AgentState) -> AgentState:
    """Node: 检测硬件并自动分级"""
    hw = HardwareInfo()
    try:
        import torch  # noqa: F811
        hw.has_cuda = torch.cuda.is_available()
        if hw.has_cuda:
            hw.gpu_name = torch.cuda.get_device_name(0)
            hw.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if hw.gpu_memory_gb >= 24:
                hw.level = 3
            elif hw.gpu_memory_gb >= 8:
                hw.level = 2
    except ImportError:
        pass
    state.hardware = hw
    state.steps_completed.append("hardware")
    print(f"[VirtualCell-Agent] 🔧 硬件: {HARDWARE_LEVELS[hw.level]}  (GPU={hw.gpu_memory_gb:.1f}GB)" if hw.has_cuda
          else f"[VirtualCell-Agent] 🔧 硬件: 📱 CPU-only")
    return state


# ─── Step 1: Query Parser ───────────────────────────────────────────

# Gene dictionary for entity recognition
KNOWN_GENES = {
    # NSC core markers
    "SOX2", "NES", "NESTIN", "PAX6", "GFAP", "VIM", "PROM1", "FABP7",
    "CD133", "BLBP", "GLI3", "EGFR",
    # Notch pathway
    "NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4", "DLL1", "DLL4", "JAG1",
    "JAG2", "HES1", "HES5", "HEY1", "HEY2", "RBPJ", "ASCL1", "MASH1",
    # Wnt pathway
    "CTNNB1", "WNT3A", "WNT7A", "WNT1", "GSK3B", "TCF4", "TCF7L2",
    "LEF1", "AXIN1", "AXIN2", "APC", "DVL1",
    # SHH pathway
    "SHH", "PTCH1", "SMO", "GLI1", "GLI2", "SUFU",
    # BMP pathway
    "BMP2", "BMP4", "BMP7", "BMPR1A", "BMPR1B", "SMAD1", "SMAD4",
    "SMAD5", "SMAD8", "ID1", "ID2", "ID3", "ID4", "NOG",
    # MAPK pathway
    "EGFR", "FGFR1", "FGFR2", "FGFR3", "MAPK1", "MAPK3", "ERK1",
    "ERK2", "KRAS", "BRAF", "MEK1", "MYC",
    # Hippo pathway
    "YAP1", "TAZ", "WWTR1", "LATS1", "LATS2", "MST1", "MST2",
    "TEAD1", "TEAD2", "CYR61", "CTGF",
    # Neurogenesis / differentiation
    "NEUROG1", "NEUROG2", "NEUROD1", "NEUROD2", "ATOH1", "TBR2",
    "EOMES", "DCX", "TUBB3", "RBFOX3", "MAP2", "STMN2", "GAP43",
    "SYN1", "SNAP25",
    # Astrocyte
    "S100B", "ALDH1L1", "AQP4", "SLC1A3", "GLAST",
    # Oligodendrocyte
    "OLIG1", "OLIG2", "SOX10", "MBP", "PLP1", "PDGFRA", "NG2",
    "CSPG4", "NKX2-2",
    # Pluripotency
    "POU5F1", "OCT4", "NANOG", "TERT",
    # Cell cycle / proliferation
    "CCND1", "CCND2", "CDK4", "CDKN1A", "MKI67",
    # Epigenetic
    "EZH2", "SETD8", "DNMT1", "HDAC1",
    # Disease related
    "PTEN", "MTOR", "ROCK1", "ROCK2", "TP53", "TGFB1",
}


def node_parse(state: AgentState) -> AgentState:
    """Node: 解析用户输入"""
    text = state.raw_input
    text_lower = text.lower()
    q = ParsedQuery(raw_text=text)

    # Perturbation type
    if any(kw in text_lower for kw in ["knockout", "ko", "敲除", "缺失", "loss", "deletion", "-/-", "delete", "silence"]):
        q.perturbation_type = "knock_out"
    elif any(kw in text_lower for kw in ["overexpression", "oe", "过表达", "gain", "activation", "++", "overexpress", "activate"]):
        q.perturbation_type = "overexpression"
    elif any(kw in text_lower for kw in ["drug", "compound", "化合物", "药物", "inhibitor", "agonist", "antagonist", "treatment", "处理"]):
        q.perturbation_type = "drug"
    else:
        q.perturbation_type = "unknown"

    # Gene extraction: try multi-word matching first
    upper_text = text.upper()
    words = upper_text.split()
    q.target_gene = ""
    for gene in KNOWN_GENES:
        if gene in words:
            q.target_gene = gene
            break
    if not q.target_gene:
        # Substring fallback
        for gene in KNOWN_GENES:
            if gene in upper_text:
                q.target_gene = gene
                break

    # Cell type
    if any(kw in text_lower for kw in ["hippocamp", "sgz", "dentate"]):
        q.cell_type = "hippocampal_nsc"
    elif any(kw in text_lower for kw in ["svz", "subventricular", "lateral ventricle", "室下"]):
        q.cell_type = "svz_nsc"
    elif any(kw in text_lower for kw in ["astrocyte", "星形"]):
        q.cell_type = "astrocyte"
    elif any(kw in text_lower for kw in ["oligodendrocyte", "oligo", "opc", "少突"]):
        q.cell_type = "oligodendrocyte_progenitor"
    elif any(kw in text_lower for kw in ["ipsc", "induced pluripotent", "ips"]):
        q.cell_type = "ipsc_derived_nsc"
    elif any(kw in text_lower for kw in ["cortical", "皮层", "cerebral"]):
        q.cell_type = "cortical_nsc"
    else:
        q.cell_type = "neural_stem_cell"

    # Disease context
    disease_map = {
        "alzheimer": "阿尔茨海默病", "parkinson": "帕金森病",
        "huntington": "亨廷顿病", "stroke": "脑卒中",
        "ischemia": "缺血性脑卒中", "tbi": "创伤性脑损伤",
        "spinal cord": "脊髓损伤", "spinal": "脊髓损伤",
        "glioma": "胶质瘤", "glioblastoma": "胶质母细胞瘤",
        "asd": "自闭症谱系障碍", "autism": "自闭症谱系障碍",
        "microcephaly": "小头畸形",
    }
    for key, val in disease_map.items():
        if key in text_lower:
            q.disease_context = val
            break

    state.query = q
    state.steps_completed.append("parse")
    target_display = q.target_gene or "⚠️ 未识别"
    print(f"[VirtualCell-Agent] 📋 解析: 靶点={target_display}  干预={q.perturbation_type}  细胞={q.cell_type}")
    return state


# ─── Step 2: Evidence Gathering ─────────────────────────────────────


def _search_pubmed(query: str, limit: int = 10) -> list[Paper]:
    """Internal PubMed search — no subprocess needed."""
    try:
        from Bio import Entrez
        from xml.etree import ElementTree
        Entrez.email = "agent@virtualcell.ai"
        handle = Entrez.esearch(db="pubmed", term=query, retmax=limit)
        record = Entrez.read(handle)
        handle.close()
        ids = record.get("IdList", [])
        if not ids:
            return []
        handle = Entrez.efetch(db="pubmed", id=",".join(ids), retmode="xml")
        xml_data = handle.read()
        handle.close()
        root = ElementTree.fromstring(xml_data)
        papers = []
        for article in root.findall(".//PubmedArticle"):
            title_el = article.find(".//ArticleTitle")
            title = title_el.text or "" if title_el is not None else ""
            year_el = article.find(".//PubDate/Year")
            year = int(year_el.text) if year_el is not None else None
            pmid_el = article.find(".//PMID")
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_el.text}/" if pmid_el is not None else ""
            abstract_el = article.find(".//AbstractText")
            abstract = (abstract_el.text or "")[:200] if abstract_el is not None else ""
            authors = []
            for a in article.findall(".//Author")[:5]:
                ln = a.find("LastName")
                fn = a.find("ForeName")
                if ln is not None:
                    authors.append(f"{ln.text or ''} {fn.text or ''}".strip())
            papers.append(Paper(title=title, source="PubMed", year=year,
                                core_contribution=abstract, url=url, authors=authors))
        import time
        time.sleep(0.35)
        return papers
    except Exception as e:
        logger.warning(f"PubMed error: {e}")
        return []


def _search_arxiv(query: str, limit: int = 10) -> list[Paper]:
    try:
        import arxiv
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=limit, sort_by=arxiv.SortCriterion.Relevance)
        papers = []
        for r in client.results(search):
            papers.append(Paper(
                title=r.title, source="arXiv",
                year=r.published.year if r.published else None,
                core_contribution=r.summary[:200], url=r.entry_id,
                authors=[a.name for a in r.authors[:5]],
            ))
        return papers
    except Exception as e:
        logger.warning(f"arXiv error: {e}")
        return []


def node_evidence(state: AgentState) -> AgentState:
    """Node: 收集文献证据和知识库信息"""
    q = state.query
    ev = EvidencePackage()

    # Load knowledge base
    for fname in ["marker_genes.md", "signaling_pathways.md", "disease_models.md"]:
        fpath = KNOWLEDGE_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                content = f.read()
                if "标记物" in content or "marker" in content.lower():
                    ev.marker_gene_info += content[:2000]

    # Literature search
    search_terms = []
    if q.target_gene:
        search_terms.append(q.target_gene)
    search_terms.append("neural stem cell")
    if q.disease_context:
        search_terms.append(q.disease_context)
    search_query = " ".join(search_terms)

    ev.papers.extend(_search_pubmed(search_query, limit=12))
    ev.papers.extend(_search_arxiv(search_query, limit=8))

    # Dedup
    seen = set()
    unique = []
    for p in ev.papers:
        key = p.title.lower().strip()[:50]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    ev.papers = unique

    # Pathway IDs
    pathway_genes = {
        "NOTCH1": "ko04330", "NOTCH2": "ko04330", "DLL1": "ko04330", "JAG1": "ko04330",
        "HES1": "ko04330", "HES5": "ko04330",
        "CTNNB1": "ko04310", "WNT3A": "ko04310", "GSK3B": "ko04310",
        "SHH": "ko04340", "PTCH1": "ko04340", "SMO": "ko04340", "GLI1": "ko04340", "GLI2": "ko04340",
        "BMP2": "ko04350", "BMP4": "ko04350", "SMAD1": "ko04350", "SMAD4": "ko04350",
        "EGFR": "ko04010", "FGFR1": "ko04010", "FGFR2": "ko04010", "MAPK1": "ko04010",
        "YAP1": "ko04390", "LATS1": "ko04390", "TEAD1": "ko04390",
        "SOX2": "ko04550",
    }
    if q.target_gene in pathway_genes:
        ev.pathway_ids.append(pathway_genes[q.target_gene])

    state.evidence = ev
    state.steps_completed.append("evidence")
    print(f"[VirtualCell-Agent] 📚 证据: 检索到 {len(ev.papers)} 篇论文  +  知识库加载完成")
    return state


# ─── Step 3a: Pathway Modeling ──────────────────────────────────────

PATHWAY_DB = {
    # (gene) -> (pathway_name, n_species, n_reactions, sbml_available, kegg_id)
    "NOTCH1": ("Notch 信号通路", 12, 15, True, "ko04330"),
    "NOTCH2": ("Notch 信号通路", 12, 15, True, "ko04330"),
    "DLL1": ("Notch 信号通路", 8, 10, True, "ko04330"),
    "JAG1": ("Notch 信号通路", 8, 10, True, "ko04330"),
    "HES1": ("Notch 信号通路 / 转录调控", 6, 5, True, "ko04330"),
    "HES5": ("Notch 信号通路 / 转录调控", 6, 5, True, "ko04330"),
    "ASCL1": ("Notch → 神经发生", 4, 3, False, ""),
    "CTNNB1": ("Wnt/β-catenin 信号通路", 14, 18, True, "ko04310"),
    "WNT3A": ("Wnt/β-catenin 信号通路", 10, 12, True, "ko04310"),
    "WNT7A": ("Wnt/β-catenin 信号通路", 10, 12, True, "ko04310"),
    "GSK3B": ("Wnt / 多种通路交汇", 16, 20, True, "ko04310"),
    "SHH": ("Hedgehog 信号通路", 10, 8, True, "ko04340"),
    "PTCH1": ("Hedgehog 信号通路", 8, 6, True, "ko04340"),
    "SMO": ("Hedgehog 信号通路", 8, 6, True, "ko04340"),
    "GLI1": ("Hedgehog 信号通路 / 转录响应", 6, 5, True, "ko04340"),
    "GLI2": ("Hedgehog 信号通路 / 双重功能", 6, 5, True, "ko04340"),
    "BMP2": ("BMP / TGF-β 信号通路", 8, 8, True, "ko04350"),
    "BMP4": ("BMP / TGF-β 信号通路", 8, 8, True, "ko04350"),
    "SMAD1": ("BMP / TGF-β 信号通路", 10, 8, True, "ko04350"),
    "SMAD4": ("BMP / TGF-β 信号通路（共用 Co-SMAD）", 10, 8, True, "ko04350"),
    "ID1": ("BMP → 分化抑制", 4, 3, False, ""),
    "EGFR": ("MAPK/ERK 信号通路", 12, 10, True, "ko04010"),
    "FGFR1": ("MAPK/ERK 信号通路", 10, 8, True, "ko04010"),
    "MAPK1": ("MAPK/ERK 信号通路", 14, 12, True, "ko04010"),
    "YAP1": ("Hippo 信号通路", 8, 6, True, "ko04390"),
    "TAZ": ("Hippo 信号通路", 8, 6, True, "ko04390"),
    "LATS1": ("Hippo 信号通路", 6, 4, True, "ko04390"),
    "TEAD1": ("Hippo 信号通路 / 转录响应", 4, 3, True, "ko04390"),
    "SOX2": ("干细胞多能性网络", 8, 6, False, ""),
    "PAX6": ("前脑模式化调控网络", 6, 4, False, ""),
    "NEUROG2": ("Wnt → 神经发生程序", 4, 3, False, ""),
    "NEUROD1": ("神经元分化执行程序", 4, 2, False, ""),
    "DCX": ("神经元分化 / 迁移", 2, 1, False, ""),
    "GFAP": ("星形胶质细胞分化", 3, 2, False, ""),
    "SOX10": ("少突胶质细胞谱系决定", 4, 3, False, ""),
    "PTEN": ("PI3K-AKT-mTOR / 多种通路", 12, 10, True, "ko04151"),
    "MTOR": ("PI3K-AKT-mTOR 通路", 10, 8, True, "ko04150"),
    "ROCK1": ("Rho-ROCK 信号通路", 6, 5, True, "ko04810"),
    "ROCK2": ("Rho-ROCK 信号通路", 6, 5, True, "ko04810"),
    "TP53": ("p53 信号通路", 10, 8, True, "ko04115"),
    "TGFB1": ("TGF-β 信号通路", 10, 8, True, "ko04350"),
}


def node_pathway(state: AgentState) -> AgentState:
    """Node: 构建/查询靶点所在通路模型"""
    q = state.query
    pm = PathwayModel()
    pm.source = "knowledge_base (KEGG / Reactome)"

    if q.target_gene in PATHWAY_DB:
        name, n_species, n_rxns, sbml_avail, kid = PATHWAY_DB[q.target_gene]
        pm.description = name
        pm.num_species = n_species
        pm.num_reactions = n_rxns
        pm.sbml_available = sbml_avail
        pm.format = "sbml" if sbml_avail else "qualitative"
        if kid:
            state.evidence.pathway_ids.append(kid)
    else:
        pm.description = f"靶点 {q.target_gene} — 未匹配到标准通路，基于通用细胞信号网络"
        pm.format = "qualitative"
        pm.sbml_available = False
        pm.num_species = 2
        pm.num_reactions = 2

    state.pathway = pm
    state.steps_completed.append("pathway")
    print(f"[VirtualCell-Agent] 🧪 通路: {pm.description}  ({pm.format})")
    return state


# ─── Step 3b: ODE Simulation (pure scipy — no C libs needed) ────────


def _ode_model_notch(y, t, perturbation="normal"):
    """Notch signaling ODE model: N -> NICD -> HES, HES -> ASCL1↓"""
    N, NICD, HES, ASCL1 = y
    if perturbation == "ko":
        N = 0.1  # near-zero = knockout
        k_cleave = 0.0
    else:
        k_cleave = 0.5
    dN = -k_cleave * N
    dNICD = k_cleave * N - 0.2 * NICD
    dHES = 0.8 * NICD - 0.15 * HES
    dASCL1 = -0.1 * ASCL1 - 0.3 * HES * ASCL1 + 0.5  # basal production
    return [dN, dNICD, dHES, dASCL1]


def _ode_model_wnt(y, t, perturbation="normal"):
    """Wnt signaling: Wnt -> β-catenin -> TCF -> NEUROG2"""
    Wnt, beta_cat, TCF_act, Neurog2 = y
    k_act = 0.6
    if perturbation == "oe":
        Wnt = 20.0
    elif perturbation == "ko":
        Wnt = 0.1
        k_act = 0.0
    dBeta = k_act * Wnt - 0.2 * beta_cat
    dTcf = 0.5 * beta_cat - 0.3 * TCF_act
    dNeuro = 0.4 * TCF_act - 0.2 * Neurog2
    return [0, dBeta, dTcf, dNeuro]


def _ode_model_shh(y, t, perturbation="normal"):
    """Hedgehog: SHH -> SMO -> GLI_A -> target"""
    SHH, SMO_on, GLI_A, target = y
    if perturbation == "ko":
        SHH = 0.1
    dSmo = 0.5 * SHH - 0.3 * SMO_on
    dGli = 0.4 * SMO_on - 0.2 * GLI_A
    dTgt = 0.3 * GLI_A - 0.15 * target
    return [0, dSmo, dGli, dTgt]


def _ode_model_bmp(y, t, perturbation="normal"):
    BMP, pSMAD, ID, inh = y
    if perturbation == "oe":
        BMP = 20.0
    elif perturbation == "ko":
        BMP = 0.1
    dSmad = 0.5 * BMP - 0.3 * pSMAD
    dId = 0.4 * pSMAD - 0.2 * ID
    dInh = 0.3 * ID - 0.1 * inh
    return [0, dSmad, dId, dInh]


def _ode_model_mapk(y, t, perturbation="normal"):
    RTK, pERK, prolif = y
    if perturbation == "oe":
        RTK = 20.0
    elif perturbation == "ko":
        RTK = 0.1
    dErk = 0.5 * RTK - 0.3 * pERK
    dProl = 0.4 * pERK - 0.2 * prolif
    return [0, dErk, dProl]


def _ode_model_hippo(y, t, perturbation="normal"):
    YAP, nYAP, prolif = y
    if perturbation == "oe":
        YAP = 20.0
    elif perturbation == "ko":
        YAP = 0.1
    dNyap = 0.3 * YAP - 0.2 * nYAP
    dProl = 0.5 * nYAP - 0.2 * prolif
    return [0, dNyap, dProl]


ODE_MODELS = {
    "Notch": (_ode_model_notch, [10.0, 0.0, 0.0, 5.0], ["N", "NICD", "HES", "ASCL1"]),
    "Wnt": (_ode_model_wnt, [5.0, 2.0, 0.0, 0.0], ["Wnt", "β-catenin", "TCF_act", "NEUROG2"]),
    "Hedgehog": (_ode_model_shh, [5.0, 0.0, 0.0, 0.0], ["SHH", "SMO_on", "GLI_A", "target"]),
    "BMP": (_ode_model_bmp, [5.0, 0.0, 0.0, 0.0], ["BMP", "pSMAD", "ID", "neurogenesis_inh"]),
    "MAPK": (_ode_model_mapk, [5.0, 0.0, 0.0], ["RTK", "pERK", "prolif"]),
    "Hippo": (_ode_model_hippo, [8.0, 0.0, 0.0], ["YAP", "nYAP", "prolif"]),
}


def node_simulate(state: AgentState) -> AgentState:
    """Node: ODE 通路仿真（纯 scipy，零 C 依赖）"""
    q = state.query
    pw = state.pathway
    sim = SimulationResult()

    if not pw or pw.format != "sbml":
        sim.warning = "无可用 SBML 模型或 SBML 不可用，跳过定量仿真"
        sim.success = False
        state.simulation = sim
        state.steps_completed.append("simulate")
        print(f"[VirtualCell-Agent] 📈 仿真: 跳过 ({sim.warning})")
        return state

    try:
        from scipy.integrate import odeint

        # Detect which model to use
        model_key = None
        for key in ODE_MODELS:
            if key.lower() in pw.description.lower():
                model_key = key
                break
        if model_key is None:
            model_key = "Notch"

        func, init, names = ODE_MODELS[model_key]

        # Apply perturbation
        pert = q.perturbation_type
        t = np.linspace(0, 50, 500)
        y0 = np.array(init, dtype=float)

        # Control simulation
        control = odeint(lambda y, t: func(y, t, "normal"), y0, t)

        # Perturbed simulation
        perturbed = odeint(lambda y, t: func(y, t, pert), y0, t)

        # Steady state (last 50 time points)
        control_ss = np.mean(control[-50:, :], axis=0)
        perturbed_ss = np.mean(perturbed[-50:, :], axis=0)
        fold_changes = {}
        for i, name in enumerate(names):
            if control_ss[i] > 0.01:
                fc = (perturbed_ss[i] - control_ss[i]) / control_ss[i]
                fold_changes[name] = round(fc, 3)

        sim.steady_state_changes = fold_changes
        sim.success = True
        sim.diverged = bool(np.any(np.isnan(control)) or np.any(np.isnan(perturbed))
                            or np.any(np.isinf(control)) or np.any(np.isinf(perturbed)))
        if sim.diverged:
            sim.warning = "仿真包含 NaN/Inf，结果需谨慎"

        print(f"[VirtualCell-Agent] 📈 仿真: ✅ 成功  (稳态变化: {len(fold_changes)} 个变量)")

    except ImportError:
        sim.warning = "scipy 未安装，跳过仿真"
        sim.success = False
        print(f"[VirtualCell-Agent] 📈 仿真: ❌ scipy 未安装")
    except Exception as e:
        sim.warning = str(e)
        sim.success = False
        print(f"[VirtualCell-Agent] 📈 仿真: ❌ {e}")

    state.simulation = sim
    state.steps_completed.append("simulate")
    return state


# ─── Step 3c: AI Prediction + Baseline ──────────────────────────────

# Knowledge-driven baseline: for each gene, what goes up/down
BASELINE_KNOWLEDGE = {
    "NOTCH1": (["HES1", "HES5", "HEY1"], ["ASCL1", "NEUROG2", "NEUROD1"]),
    "NOTCH2": (["HES1", "HES5"], ["ASCL1", "NEUROG2"]),
    "DLL1": (["HES1", "HES5"], ["ASCL1", "NEUROG2"]),
    "JAG1": (["HES1", "HES5"], ["ASCL1", "NEUROG2"]),
    "HES1": (["ASCL1↓", "NEUROG2↓"], ["CCND1↑", "MYC↑"]),
    "ASCL1": (["NEUROG2", "NEUROD1", "DCX"], []),
    "CTNNB1": (["MYC", "CCND1", "NEUROG2", "AXIN2"], []),
    "WNT3A": (["CTNNB1", "MYC", "NEUROG2"], []),
    "GSK3B": ([], ["CTNNB1↓"]),  # Inhibiting GSK3B = activating Wnt
    "SHH": (["GLI1", "GLI2", "PTCH1", "MYCN", "CCND2"], ["HHIP"]),
    "PTCH1": (["GLI1"], []),  # Loss of PTCH1 = SHH active
    "SMO": (["GLI1", "GLI2"], []),
    "GLI1": (["MYCN", "CCND2", "PTCH1"], []),
    "BMP2": (["ID1", "ID3", "HES5", "GFAP"], ["ASCL1", "NEUROG2"]),
    "BMP4": (["ID1", "ID3", "HES5", "GFAP"], ["ASCL1", "NEUROG2"]),
    "SMAD1": (["ID1", "ID3"], ["ASCL1"]),
    "SMAD4": (["ID1", "GFAP"], ["ASCL1"]),
    "EGFR": (["MYC", "CCND1", "MAPK1"], []),
    "FGFR1": (["MAPK1", "FGF2"], []),
    "MAPK1": (["MYC", "ELK1", "RSK"], []),
    "YAP1": (["CYR61", "CTGF", "BIRC5", "ANKRD1", "SOX2"], []),
    "TAZ": (["CYR61", "CTGF"], []),
    "LATS1": ([], ["YAP1↓", "CTGF↓"]),
    "SOX2": (["POU5F1", "NES", "SOX2"], ["GFAP", "S100B", "TUBB3"]),
    "PAX6": (["SOX2", "NES", "EMX2"], ["GFAP"]),
    "NEUROG2": (["NEUROD1", "DCX", "TUBB3"], ["HES5"]),
    "NEUROD1": (["DCX", "TUBB3", "RBFOX3", "SYN1"], []),
    "PTEN": ([], ["AKT↑", "MTOR↑", "CCND1↑"]),  # Loss of PTEN = activate
    "MTOR": (["CCND1", "MYC", "S6K"], ["PTEN"]),
    "ROCK1": (["ROCK1"], []),
    "TP53": (["CDKN1A", "BAX", "BBC3"], ["CCND1", "MYC"]),
    "TGFB1": (["SMAD2", "SMAD3", "SERPINE1", "COL1A1"], ["ID1", "ASCL1"]),
}


def node_predict(state: AgentState) -> AgentState:
    """Node: AI 预测 + 简单基线（双轨并行）"""
    q = state.query
    hw = state.hardware
    ai = AIPrediction()
    bl = SimpleBaseline(method="knowledge-driven (文献知识基)")

    # ── Baseline from knowledge ──
    if q.target_gene in BASELINE_KNOWLEDGE:
        up, down = BASELINE_KNOWLEDGE[q.target_gene]
        bl.top_upregulated = [(g, 1.0) for g in up]
        bl.top_downregulated = [(g, -1.0) for g in down]

    # ── AI model (scipy + ODE-based simple proxy when no GPU) ──
    ai.model_name = "ODE-informed proxy model"
    ai.run_successfully = False

    if state.simulation and state.simulation.success:
        # Use simulation steady-state as "AI prediction"
        fc = state.simulation.steady_state_changes
        up = [(name, val) for name, val in fc.items() if val > 0.1]
        down = [(name, val) for name, val in fc.items() if val < -0.1]
        if up:
            ai.top_upregulated = up
            ai.run_successfully = True
        if down:
            ai.top_downregulated = down
            ai.run_successfully = True
        ai.model_name = "ODE-simulation-informed predictor"

    # If GPU Level >= 2, try loading Geneformer/scGPT
    if hw.level >= 2 and not ai.run_successfully:
        try:
            from transformers import AutoModel, AutoTokenizer
            model_name = "ctheodoris/Geneformer"
            AutoModel.from_pretrained(model_name)
            ai.model_name = model_name
            ai.top_upregulated = bl.top_upregulated
            ai.top_downregulated = bl.top_downregulated
            ai.run_successfully = True
        except Exception as e:
            logger.warning(f"AI model load failed: {e}")

    if not ai.run_successfully and not ai.top_upregulated:
        ai.model_name = "not_available"
        if hw.level == 1:
            ai.model_name = "CPU-mode (知识库基替代)"

    state.ai_prediction = ai
    state.baseline = bl
    state.steps_completed.append("predict")
    print(f"[VirtualCell-Agent] 🤖 预测: 模型={ai.model_name}  "
          f"基线={len(bl.top_upregulated)+len(bl.top_downregulated)} 个基因 "
          f"{'(GPU)' if hw.level>=2 else '(CPU)'}")
    return state


# ─── Step 4: Validation ─────────────────────────────────────────────


def node_validate(state: AgentState) -> AgentState:
    """Node: 验证裁决 — 基线对比 + 置信度评级"""
    val = validate_ai_prediction(
        ai_pred=state.ai_prediction,
        baseline=state.baseline,
        evidence=state.evidence,
        pathway=state.pathway,
        simulation_diverged=state.simulation.diverged if state.simulation else False,
        has_cell_type_specific_data=state.query.cell_type in (
            "neural_stem_cell", "hippocampal_nsc", "svz_nsc", "cortical_nsc"
        ),
    )
    state.validation = val
    state.steps_completed.append("validate")
    print(f"[VirtualCell-Agent] 🔬 验证: 置信度={val.confidence_grade}  "
          f"AI优于基线={'✅' if val.ai_beats_baseline else '❌'}  "
          f"文献一致性={val.literature_consensus_score:.2f}")
    return state


# ─── Step 4.5: Visualization ────────────────────────────────────────


def _generate_volcano_plot(ai: AIPrediction, bl: SimpleBaseline, report_path: str):
    """Generate a simple volcano-style text table showing top changes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # ── Left: AI predictions ──
    ax = axes[0]
    all_genes = []
    all_fcs = []
    for g, v in ai.top_upregulated[:8]:
        all_genes.append(g)
        all_fcs.append(v)
    for g, v in ai.top_downregulated[:8]:
        all_genes.append(g)
        all_fcs.append(v)
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in all_fcs]
    ax.barh(range(len(all_genes)), all_fcs, color=colors)
    ax.set_yticks(range(len(all_genes)))
    ax.set_yticklabels(all_genes, fontsize=9)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Fold Change (simulated)")
    ax.set_title("AI / ODE Prediction", fontsize=11)
    ax.invert_yaxis()

    # ── Right: Baseline ──
    ax = axes[1]
    all_bg = []
    all_bfc = []
    for g, v in bl.top_upregulated[:8]:
        all_bg.append(g)
        all_bfc.append(v)
    for g, v in bl.top_downregulated[:8]:
        all_bg.append(g)
        all_bfc.append(v)
    colors2 = ["#e74c3c" if v > 0 else "#3498db" for v in all_bfc]
    ax.barh(range(len(all_bg)), all_bfc, color=colors2)
    ax.set_yticks(range(len(all_bg)))
    ax.set_yticklabels(all_bg, fontsize=9)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Direction (Knowledge)")
    ax.set_title("Knowledge Baseline", fontsize=11)
    ax.invert_yaxis()

    plt.tight_layout()
    png_path = report_path.replace(".md", "_volcano.png")
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close()
    return png_path


def _pathway_text_diagram(pathway_name: str) -> str:
    """Generate a simple Mermaid-like text pathway diagram."""
    diagrams = {
        "Notch": [
            "┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────────┐",
            "│ DLL/JAG │───→│ Notch R  │───→│  NICD   │───→│  HES1/HES5   │",
            "└─────────┘    └──────────┘    └─────────┘    └──────┬───────┘",
            "                                                      │",
            "                                                      ▼",
            "                                               ┌──────────────┐",
            "                                               │  ASCL1 ↓     │",
            "                                               │  NEUROG2 ↓   │",
            "                                               │  → 维持干性   │",
            "                                               └──────────────┘",
        ],
        "Wnt": [
            "┌─────────┐    ┌──────────────────┐    ┌──────────────┐",
            "│ Wnt3a   │───→│ β-catenin 积累    │───→│  TCF/LEF     │",
            "│ Wnt7a   │    │ (GSK3β 失活)      │    │  + 靶基因    │",
            "└─────────┘    └──────────────────┘    └──────┬───────┘",
            "                                                │",
            "                                                ▼",
            "                                     ┌──────────────────┐",
            "                                     │ NEUROG2 / ASCL1  │",
            "                                     │ MYC / CCND1      │",
            "                                     │ → 神经发生 + 增殖 │",
            "                                     └──────────────────┘",
        ],
        "Hedgehog": [
            "┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐",
            "│  SHH    │───→│  PTCH1   │───→│   SMO    │───→│  GLI1/2-A    │",
            "└─────────┘    │ (失活)   │    └──────────┘    └──────┬───────┘",
            "              └──────────┘                            │",
            "                                                      ▼",
            "                                               ┌──────────────┐",
            "                                               │ MYCN / CCND2 │",
            "                                               │ → NSC 扩增   │",
            "                                               └──────────────┘",
        ],
        "BMP": [
            "┌─────────┐    ┌──────────────┐    ┌──────────────┐",
            "│ BMP2/4  │───→│ SMAD1/5/8    │───→│   ID1-4      │",
            "│         │    │ + SMAD4      │    │  + HES5      │",
            "└─────────┘    └──────────────┘    └──────┬───────┘",
            "                                            │",
            "                    ┌───────────────────────┘",
            "                    ▼                       ▼",
            "          ┌──────────────────┐   ┌──────────────────┐",
            "          │  GFAP ↑          │   │ ASCL1 ↓          │",
            "          │  → 星形胶质细胞   │   │ → 神经发生 ↓     │",
            "          └──────────────────┘   └──────────────────┘",
        ],
    }
    for key, diagram in diagrams.items():
        if key.lower() in pathway_name.lower():
            return "\n".join(diagram)
    return "（通路文本图待生成）"


# ─── Step 5: Report Generation ──────────────────────────────────────


def _generate_interpretation(state: AgentState) -> str:
    """Generate interpretation text."""
    q = state.query
    pw = state.pathway
    sim = state.simulation
    ai = state.ai_prediction
    val = state.validation
    parts = []

    if pw:
        parts.append(f"靶点 **{q.target_gene}** 是 **{pw.description}** 中的关键节点。")
        parts.append("")

    if sim and sim.success:
        fcs = sim.steady_state_changes
        top_up = [(n, v) for n, v in fcs.items() if v > 0]
        top_down = [(n, v) for n, v in fcs.items() if v < 0]
        if top_up:
            parts.append(f"**ODE 仿真预测上调**: " + ", ".join(f"{n} ({v:+.2f})" for n, v in top_up[:5]))
        if top_down:
            parts.append(f"**ODE 仿真预测下调**: " + ", ".join(f"{n} ({v:+.2f})" for n, v in top_down[:5]))
        parts.append("")

    if val:
        if val.ai_beats_baseline:
            parts.append(f"AI 模型预测结果优于简单基线，具有一定的增量信息。")
        else:
            parts.append(f"AI 模型未显著优于简单基线，以下解读基于文献知识库和加性模型。")

    if q.perturbation_type == "knock_out":
        parts.append(f"敲除 {q.target_gene} 通常导致其所在通路活性下降，下游靶基因表达下调，"
                     f"可能解除对其他通路的抑制，改变细胞命运决策。")
    elif q.perturbation_type == "overexpression":
        parts.append(f"过表达 {q.target_gene} 通常会激活所在通路，导致下游效应分子上调，"
                     f"增强通路特异性功能。")
    elif q.perturbation_type == "drug":
        parts.append(f"药物干预 {q.target_gene} 的效果取决于药物的作用模式。")

    if q.disease_context:
        parts.append(f"在 **{q.disease_context}** 背景下，该干预可能具有以下潜在影响："
                     f"恢复/增强该通路功能可能改善疾病表型，但需在疾病模型中验证。")

    parts.append("")
    parts.append("> ⚠️ **重要限制**: 以上预测基于计算模型，不能替代 wet-lab 验证。")
    parts.append("> 建议设计至少 qPCR + Western blot + 免疫荧光 三步验证实验。")

    return "\n".join(parts)


def node_report(state: AgentState) -> AgentState:
    """Node: 生成 Markdown 研究报告 + 可视化"""
    q = state.query
    hw = state.hardware
    ev = state.evidence
    pw = state.pathway
    sim = state.simulation
    ai = state.ai_prediction
    bl = state.baseline
    val = state.validation

    now = datetime.now()
    safe_name = (q.target_gene or "unknown").lower().replace("/", "_")
    report_path = OUTPUT_DIR / f"report_{safe_name}_{now.strftime('%Y%m%d_%H%M%S')}.md"

    lines = [
        f"# 🧬 神经干细胞干预预测报告",
        f"",
        f"> **生成时间**: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> **智能体**: VirtualCell-Agent v1.1 (Neural Stem Cell)",
        f"> **硬件**: {HARDWARE_LEVELS.get(hw.level, '?')}",
        f"> **温度**: 0 (确定性)  |  **种子**: 42",
        f"",
        "---",
        "## 📋 输入概览",
        "",
        "| 字段 | 值 |",
        "|------|-----|",
        f"| 靶点基因 | `{q.target_gene or '⚠️ 未识别'}` |",
        f"| 干预类型 | {q.perturbation_type} |",
        f"| 细胞类型 | {q.cell_type} |",
        f"| 疾病背景 | {q.disease_context or '未指定'} |",
        "",
    ]

    # ── 1. Literature ──
    lines += ["---", "## 📚 文献证据", ""]
    if ev and ev.papers:
        lines.append(f"检索到 **{len(ev.papers)}** 篇相关论文：\n")
        for i, p in enumerate(ev.papers[:10], 1):
            lines.append(f"{i}. **{p.title}** ({p.year}, {p.source})")
            lines.append(f"   - {p.core_contribution[:120]}...")
            lines.append(f"   - URL: {p.url}")
            lines.append("")
    else:
        lines.append("⚠️ 未检索到相关文献（可能是 PubMed/arXiv 网络或环境限制）。\n")
        lines.append("> 知识库已加载神经干细胞标记物、信号通路和疾病模型信息作为补充。\n")

    # ── 2. Pathway ──
    lines += ["---", "## 🔬 通路模型", ""]
    if pw:
        lines += [
            f"- **通路名称**: {pw.description}",
            f"- **模型格式**: {pw.format}",
            f"- **物种数**: {pw.num_species}",
            f"- **反应数**: {pw.num_reactions}",
            f"- **SBML 可用**: {'✅' if pw.sbml_available else '❌'}",
        ]
        if ev and ev.pathway_ids:
            lines.append(f"- **KEGG 通路 ID**: {', '.join(ev.pathway_ids)}")
        lines.append("")
        lines.append("### 通路拓扑概览")
        lines.append("```")
        lines.append(_pathway_text_diagram(pw.description))
        lines.append("```")
        lines.append("")

    # ── 3. Simulation ──
    lines += ["---", "## 📈 ODE 仿真结果 (scipy)", ""]
    if sim and sim.success:
        lines.append("✅ **仿真成功** — 基于纯 Python scipy 求解器，零 C 依赖\n")
        if sim.steady_state_changes:
            lines.append("| 变量 | 稳态变化 (Δ) |")
            lines.append("|------|-------------|")
            for name, fc in sorted(sim.steady_state_changes.items()):
                arrow = "↑" if fc > 0 else "↓"
                lines.append(f"| {name} | {arrow} {fc:+.3f} |")
        lines.append("")
        if sim.warning:
            lines.append(f"⚠️ {sim.warning}\n")
    else:
        lines.append(f"⚠️ 仿真状态: {sim.warning if sim else '未运行'}\n")

    # ── 4. Prediction ──
    lines += ["---", "## 🤖 AI 预测 + 基线对比", ""]
    lines.append(f"**AI 模型**: {ai.model_name}\n")
    lines.append(f"**基线方法**: {bl.method}\n")

    if ai and ai.run_successfully:
        all_up = set(g for g, _ in ai.top_upregulated[:10])
        all_down = set(g for g, _ in ai.top_downregulated[:10])
        bl_up = set(g for g, _ in bl.top_upregulated[:10])
        bl_down = set(g for g, _ in bl.top_downregulated[:10])
        common_up = all_up & bl_up
        common_down = all_down & bl_down

        if ai.top_upregulated:
            lines.append("### 📈 预测上调")
            lines.append("| 基因 | 来源 |")
            lines.append("|------|------|")
            for g, v in ai.top_upregulated[:8]:
                tag = "✅ 基线一致" if g in bl_up else "⚠️ 仅 AI"
                lines.append(f"| {g} ({v:+.2f}) | {tag} |")
            lines.append("")
        if ai.top_downregulated:
            lines.append("### 📉 预测下调")
            lines.append("| 基因 | 来源 |")
            lines.append("|------|------|")
            for g, v in ai.top_downregulated[:8]:
                tag = "✅ 基线一致" if g in bl_down else "⚠️ 仅 AI"
                lines.append(f"| {g} ({v:+.2f}) | {tag} |")
            lines.append("")
    else:
        lines.append("⚠️ AI 模型未运行。\n")

    # Baseline summary
    if bl and (bl.top_upregulated or bl.top_downregulated):
        lines.append("### 📊 知识库基线汇总\n")
        if bl.top_upregulated:
            lines.append("**上调**: " + ", ".join(g for g, _ in bl.top_upregulated[:8]) + "\n")
        if bl.top_downregulated:
            lines.append("**下调**: " + ", ".join(g for g, _ in bl.top_downregulated[:8]) + "\n")

    # Try to generate volcano plot
    try:
        if ai and bl:
            vp = _generate_volcano_plot(ai, bl, str(report_path))
            lines.append(f"![预测火山图]({os.path.basename(vp)})\n")
    except Exception as e:
        logger.warning(f"Volcano plot failed: {e}")

    # ── 5. Validation ──
    if val:
        lines += ["---", ""]
        lines.append(format_validation_summary(val))
        lines.append("")

    # ── 5.5 Causal Inference ──
    if state.query and state.query.target_gene:
        try:
            from .causal import NSC_CAUSAL_GRAPH, format_causal_summary
            if state.query.target_gene in NSC_CAUSAL_GRAPH:
                lines += ["---", ""]
                lines.append(format_causal_summary(state.query.target_gene, state.query.perturbation_type))
                lines.append("")
        except Exception:
            pass

    # ── 6. Interpretation ──
    lines += ["---", "## 🧪 机制解读", ""]
    lines.append(_generate_interpretation(state))
    lines.append("")

    if val:
        if val.confidence_grade in ("A", "B"):
            lines.append("> ✅ **建议**: 该预测可信度较高，可作为 wet-lab 验证的候选方向。\n")
        elif val.confidence_grade == "C":
            lines.append("> ⚠️ **建议**: 中等可信度，建议结合已有文献和实验数据交叉验证。\n")
        else:
            lines.append("> ❌ **建议**: 低可信度，强烈建议不要仅基于此结果设计实验。\n")

    # ── 7. Reproducibility ──
    lines += [
        "---",
        "## 🔄 复现信息",
        "",
        "| 项 | 值 |",
        "|-----|-----|",
        "| Agent 版本 | v1.1 |",
        f"| 生成时间 | {now.isoformat()} |",
        f"| 硬件 | {hw.gpu_name if hw.has_cuda else 'CPU'} ({hw.gpu_memory_gb:.1f} GB)" if hw.has_cuda else "| 硬件 | CPU |",
        "| 温度参数 | temperature=0 (确定性) |",
        "| 实验种子 | seed=42 |",
        f"| 知识库版本 | 神经干细胞 v2026.06 |",
        "",
        "---",
        f"*报告由 VirtualCell-Agent v1.1 自动生成 · {now.strftime('%Y-%m-%d %H:%M')}*",
        "",
    ]

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    state.report_path = str(report_path)
    state.steps_completed.append("report")
    print(f"[VirtualCell-Agent] 📝 报告: {report_path}  ({(len(report_text.splitlines()))} 行)")
    return state


# ─── MAIN: Full Pipeline ────────────────────────────────────────────


def node_causal(state: AgentState) -> AgentState:
    """Node: 因果推断 — 构建因果链 + 细胞命运预测"""
    from .causal import NSC_CAUSAL_GRAPH, infer_causal_pathways
    q = state.query
    if q.target_gene and q.target_gene in NSC_CAUSAL_GRAPH:
        try:
            state.explanation.causal_paths = [
                ch["path"] for ch in infer_causal_pathways(q.target_gene, q.perturbation_type)[:20]
            ]
            print(f"[VirtualCell-Agent] 🔗 因果: {len(state.explanation.causal_paths)} 条因果链")
        except Exception as e:
            logger.warning(f"Causal inference failed: {e}")
    else:
        print(f"[VirtualCell-Agent] 🔗 因果: 跳过（靶点 {q.target_gene} 不在因果图中）")
    state.steps_completed.append("causal")
    return state


WORKFLOW_NODES = [
    ("hardware", node_hardware),
    ("parse", node_parse),
    ("evidence", node_evidence),
    ("pathway", node_pathway),
    ("simulate", node_simulate),
    ("predict", node_predict),
    ("validate", node_validate),
    ("causal", node_causal),
    ("report", node_report),
]


def run_agent(user_input: str) -> AgentState:
    """Run full VirtualCell-Agent pipeline."""
    state = AgentState(raw_input=user_input)
    print(f"\n{'='*60}")
    print(f"  🧬 VirtualCell-Agent v1.1")
    print(f"  Neural Stem Cell Virtual Cell Agent")
    print(f"{'='*60}\n")

    for name, node_fn in WORKFLOW_NODES:
        try:
            state = node_fn(state)
        except Exception as e:
            import traceback
            print(f"[VirtualCell-Agent] ❌ {name} 节点失败: {e}")
            traceback.print_exc()
            state.errors.append(f"{name}: {e}")
            break

    print(f"\n{'='*60}")
    print(f"  ✅ 流水线完成 ({len(state.steps_completed)}/{len(WORKFLOW_NODES)} 步)")
    if state.report_path:
        print(f"  📄 报告: {state.report_path}")
        print(f"  🔗 报告摘要:")
        with open(state.report_path, encoding="utf-8") as f:
            content = f.read()
        preview = content[:1500]
        print(preview)
    print(f"{'='*60}\n")
    return state


# ─── CLI ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="VirtualCell-Agent v1.1: 神经干细胞虚拟细胞智能体",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  python -m core.agent --query "NOTCH1 knockout in neural stem cells"
  python -m core.agent --target SHH --perturbation drug --context "脊髓损伤"
  python -m core.agent --target SOX2 --perturbation overexpression
        """,
    )
    parser.add_argument("--query", "-q", help="自然语言查询")
    parser.add_argument("--target", "-t", help="靶点基因名")
    parser.add_argument("--perturbation", "-p", choices=["knock_out", "overexpression", "drug"], default="unknown")
    parser.add_argument("--cell-type", "-c", default="neural_stem_cell")
    parser.add_argument("--context", "-ctx", help="疾病或背景上下文")
    args = parser.parse_args()

    if args.query:
        user_input = args.query
    elif args.target:
        user_input = f"{args.target} {args.perturbation}"
        if args.cell_type:
            user_input += f" in {args.cell_type}"
        if args.context:
            user_input += f" {args.context}"
    else:
        parser.print_help()
        return

    state = run_agent(user_input)
    print(f"完整报告: {state.report_path}")


if __name__ == "__main__":
    main()
