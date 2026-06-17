# VirtualCell-Agent 测试套件

"""测试核心验证器"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.state import (
    AIPrediction, SimpleBaseline, EvidencePackage, Paper,
    PathwayModel, ValidationResult
)
from core.validator import validate_ai_prediction, compute_confidence_grade


def test_ai_better_than_baseline():
    """Case: AI prediction is meaningfully different from baseline"""
    ai = AIPrediction(
        model_name="geneformer",
        run_successfully=True,
        top_upregulated=[("HES1", 2.0), ("HES5", 1.8), ("HEY1", 1.5)],
        top_downregulated=[("ASCL1", -2.0), ("NEUROG2", -1.5)],
    )
    baseline = SimpleBaseline(
        method="additive",
        top_upregulated=[("HES1", 1.0), ("HES5", 1.0)],
        top_downregulated=[("ASCL1", -1.0)],
    )
    evidence = EvidencePackage(papers=[Paper("Test", "PubMed", 2024, "Test", "url")])
    pathway = PathwayModel(format="sbml", description="Notch", num_species=4, num_reactions=3, sbml_available=True)

    result = validate_ai_prediction(ai, baseline, evidence, pathway, False, True)
    assert result.ai_beats_baseline
    print("✅ test_ai_better_than_baseline passed")


def test_ai_no_output():
    """Case: AI prediction failed"""
    ai = AIPrediction(model_name="geneformer", run_successfully=False)
    baseline = SimpleBaseline(method="additive")
    evidence = EvidencePackage()
    pathway = PathwayModel()

    result = validate_ai_prediction(ai, baseline, evidence, pathway, False, False)
    assert not result.ai_beats_baseline
    assert result.confidence_grade == "D"
    print("✅ test_ai_no_output passed")


def test_confidence_grade():
    """Test confidence grade computation"""
    # A grade
    grade, warns = compute_confidence_grade(True, 0.9, True, False, True)
    assert grade == "A"
    assert len(warns) == 0

    # D grade (all bad)
    grade, warns = compute_confidence_grade(False, 0.1, False, True, False)
    assert grade == "D"
    print("✅ test_confidence_grade passed")


def test_hardware_levels():
    """Test hardware level interpretation"""
    from core.state import HARDWARE_LEVELS
    assert "🚀" in HARDWARE_LEVELS[3]
    assert "📱" in HARDWARE_LEVELS[1]
    print("✅ test_hardware_levels passed")


def test_parse_query():
    """Test query parsing"""
    from core.agent import parse_query

    q = parse_query("NOTCH1 knockout in neural stem cells")
    assert q.target_gene == "NOTCH1"
    assert q.perturbation_type == "knock_out"
    assert q.cell_type == "neural_stem_cell"

    q2 = parse_query("分析SOX2过表达对皮层神经干细胞的影响")
    assert q2.target_gene == "SOX2"
    assert q2.perturbation_type == "overexpression"

    print("✅ test_parse_query passed")


if __name__ == "__main__":
    test_ai_better_than_baseline()
    test_ai_no_output()
    test_confidence_grade()
    test_hardware_levels()
    test_parse_query()
    print("\n🎉 All tests passed!")
