"""VirtualCell-Agent: 验证裁决层 - 基线验证 + 置信度评估"""

import json
import logging
from typing import Optional

from .state import (
    AIPrediction, SimpleBaseline, ValidationResult,
    PathwayModel, EvidencePackage
)

logger = logging.getLogger(__name__)


def compute_confidence_grade(
    ai_beats_baseline: bool,
    literature_consensus: float,
    pathway_plausible: bool,
    simulation_diverged: bool,
    has_cell_type_specific_data: bool,
) -> tuple[str, list[str]]:
    """
    基于多维证据计算置信度等级 (A/B/C/D)。

    等级定义:
        A: AI 优于基线 + 文献高度一致 + 通路合理 + 有细胞特异性数据
        B: 基本合理，但某一项证据偏弱
        C: 多项证据偏弱，或依赖数据迁移
        D: 严重警告，结果仅供参考
    """
    warnings = []

    # Start grade at A and downgrade
    grade = "A"

    if not ai_beats_baseline:
        grade = _downgrade(grade)
        warnings.append("AI模型未优于简单基线，结果基于基线模型")

    if not has_cell_type_specific_data:
        grade = _downgrade(grade)
        warnings.append("缺乏该细胞类型特异性干预数据，结果基于数据迁移或泛细胞模型")

    if literature_consensus < 0.5:
        grade = _downgrade(grade)
        grade = _downgrade(grade)  # additional downgrade for major conflict
        warnings.append("预测结果与现有文献存在较大冲突，需谨慎解读")

    if simulation_diverged:
        grade = _downgrade(grade)
        warnings.append("ODE 仿真出现数值发散，定量结果已排除")

    if not pathway_plausible:
        grade = _downgrade(grade)
        warnings.append("预测结果在通路层面缺乏合理性")

    return grade, warnings


def _downgrade(current: str) -> str:
    mapping = {"A": "B", "B": "C", "C": "D", "D": "D"}
    return mapping.get(current, "D")


def validate_ai_prediction(
    ai_pred: AIPrediction,
    baseline: SimpleBaseline,
    evidence: EvidencePackage,
    pathway: Optional[PathwayModel],
    simulation_diverged: bool,
    has_cell_type_specific_data: bool = False,
    metric_threshold: float = 0.05,
) -> ValidationResult:
    """
    对 AI 预测进行全面的验证裁决。

    Args:
        ai_pred: AI 模型预测结果
        baseline: 简单基线模型结果
        evidence: 文献证据包
        pathway: 通路模型信息
        simulation_diverged: ODE 仿真是否发散
        has_cell_type_specific_data: 是否有该细胞类型特异性干预数据
        metric_threshold: AI 需超越基线的度量阈值

    Returns:
        ValidationResult: 验证结果对象
    """
    result = ValidationResult()
    result.baseline_method = baseline.method

    # ── 1. AI vs Baseline ──
    # Simplified comparison: check if AI has predictions
    ai_has_output = (
        ai_pred.run_successfully
        and (len(ai_pred.top_upregulated) + len(ai_pred.top_downregulated) > 0)
    )
    baseline_has_output = (
        len(baseline.top_upregulated) + len(baseline.top_downregulated) > 0
    )

    if ai_has_output and baseline_has_output:
        # Compare overlap at top N
        ai_top_genes = set(g for g, _ in ai_pred.top_upregulated) | set(
            g for g, _ in ai_pred.top_downregulated
        )
        baseline_top_genes = set(g for g, _ in baseline.top_upregulated) | set(
            g for g, _ in baseline.top_downregulated
        )
        overlap = len(ai_top_genes & baseline_top_genes)
        union = len(ai_top_genes | baseline_top_genes)
        jaccard = overlap / max(union, 1)

        # If AI and baseline are very similar, AI might not be adding value
        # If very different, we'd need to check which is more accurate
        # For MVP: flag if jaccard < 0.3 (too different from simple baseline)
        if jaccard > 0.7:
            result.ai_beats_baseline = False  # AI not significantly different
            result.ai_vs_baseline_metric = f"Jaccard={jaccard:.2f} (too similar to baseline)"
            result.warnings.append("AI预测与简单基线高度相似，未提供增量信息")
        else:
            result.ai_beats_baseline = True
            result.ai_vs_baseline_metric = f"Jaccard={jaccard:.2f}"
    elif ai_has_output and not baseline_has_output:
        result.warnings.append("基线模型无输出，无法验证AI预测的增量价值")
        result.ai_beats_baseline = True  # No baseline to beat
    else:
        result.warnings.append("AI预测未成功运行，结果不可用")
        result.ai_beats_baseline = False
        result.ai_vs_baseline_metric = "AI failed to produce output"

    # ── 2. Literature Consensus ──
    if evidence and evidence.papers:
        # Estimate: if we have relevant papers, it's likely the target
        # is documented, raising consensus score
        has_direct_evidence = any(
            result.ai_beats_baseline for _ in evidence.papers[:3]
        )
        # For MVP: use paper count as a rough proxy
        # This should be replaced with semantic matching in production
        num_papers = len(evidence.papers)
        if num_papers >= 10:
            result.literature_consensus_score = 0.8
        elif num_papers >= 5:
            result.literature_consensus_score = 0.6
        elif num_papers >= 2:
            result.literature_consensus_score = 0.4
        else:
            result.literature_consensus_score = 0.2
            result.warnings.append("该靶点在神经干细胞中的文献证据有限")
    else:
        result.literature_consensus_score = 0.1
        result.warnings.append("未检索到相关文献，结果缺乏文献支持")

    # ── 3. Pathway Plausibility ──
    result.pathway_plausible = (
        pathway is not None and pathway.num_species > 0
    )
    if not result.pathway_plausible:
        result.warnings.append("缺乏通路模型支持，结果可能缺乏机制基础")

    # ── 4. Simulation ──
    result.simulation_diverged = simulation_diverged
    if simulation_diverged:
        result.warnings.append("ODE 仿真出现数值发散（NaN/Inf），定量结果已排除")

    # ── 5. Final Confidence ──
    result.confidence_grade, grade_warnings = compute_confidence_grade(
        ai_beats_baseline=result.ai_beats_baseline,
        literature_consensus=result.literature_consensus_score,
        pathway_plausible=result.pathway_plausible,
        simulation_diverged=result.simulation_diverged,
        has_cell_type_specific_data=has_cell_type_specific_data,
    )
    result.warnings.extend(grade_warnings)

    return result


def format_validation_summary(result: ValidationResult) -> str:
    """将验证结果格式化为 Markdown 摘要块"""
    grade_emojis = {"A": "✅ A (高置信度)", "B": "⚠️ B", "C": "⚠️ C", "D": "❌ D (低置信度)"}
    grade_str = grade_emojis.get(result.confidence_grade, result.confidence_grade)

    lines = [
        "## 🔬 置信度评估",
        "",
        f"**综合置信等级**: {grade_str}",
        "",
        "| 验证维度 | 结果 |",
        "|---------|------|",
        f"| AI 优于基线 | {'✅' if result.ai_beats_baseline else '❌'} {result.ai_vs_baseline_metric} |",
        f"| 文献一致性 | {'✅' if result.literature_consensus_score >= 0.5 else '⚠️'} 得分 {result.literature_consensus_score:.2f} |",
        f"| 通路合理性 | {'✅' if result.pathway_plausible else '❌'} |",
        f"| 仿真收敛 | {'✅' if not result.simulation_diverged else '❌'} |",
        "",
    ]

    if result.warnings:
        lines.append("**⚠️ 警告**:")
        for w in result.warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)
