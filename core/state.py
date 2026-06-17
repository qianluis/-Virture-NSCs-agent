"""VirtualCell-Agent: 状态数据结构定义"""

from dataclasses import dataclass, field
from typing import Optional, Any

# ─── Hardware Level ─────────────────────────────────────────────────
HARDWARE_LEVELS = {
    3: "🚀 Full (24GB+ GPU, all AI models enabled)",
    2: "⚡ Mid (8-16GB GPU, scGPT/Geneformer only)",
    1: "📱 CPU-only (literature + simulation + baseline only)",
}


@dataclass
class HardwareInfo:
    level: int = 1  # 1, 2, or 3
    has_cuda: bool = False
    gpu_name: str = "none"
    gpu_memory_gb: float = 0.0


# ─── Query Parsing ──────────────────────────────────────────────────

@dataclass
class ParsedQuery:
    raw_text: str
    target_gene: str = ""
    perturbation_type: str = ""  # knock_out | overexpression | drug | unknown
    cell_type: str = "neural_stem_cell"
    disease_context: str = ""
    output_format: str = "report"


# ─── Evidence ───────────────────────────────────────────────────────

@dataclass
class Paper:
    title: str
    source: str
    year: int
    core_contribution: str
    url: str
    authors: list[str] = field(default_factory=list)


@dataclass
class EvidencePackage:
    papers: list[Paper] = field(default_factory=list)
    pathway_ids: list[str] = field(default_factory=list)
    known_interactions: list[str] = field(default_factory=list)
    marker_gene_info: str = ""


# ─── Modeling ───────────────────────────────────────────────────────

@dataclass
class PathwayModel:
    format: str = ""  # "sbml" | "kegg" | "qualitative"
    source: str = ""
    num_species: int = 0
    num_reactions: int = 0
    sbml_available: bool = False
    description: str = ""


@dataclass
class SimulationResult:
    success: bool = False
    steady_state_changes: dict = field(default_factory=dict)
    time_series: Optional[Any] = None  # numpy array path or None
    warning: str = ""
    diverged: bool = False


@dataclass
class AIPrediction:
    model_name: str = ""
    run_successfully: bool = False
    top_upregulated: list[tuple[str, float]] = field(default_factory=list)
    top_downregulated: list[tuple[str, float]] = field(default_factory=list)
    effect_vector_path: str = ""


@dataclass
class SimpleBaseline:
    """Simple additive/null baseline for perturbation prediction"""
    method: str = "additive"  # additive | mean | null
    top_upregulated: list[tuple[str, float]] = field(default_factory=list)
    top_downregulated: list[tuple[str, float]] = field(default_factory=list)


# ─── Validation ─────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    ai_beats_baseline: bool = False
    baseline_method: str = ""
    ai_vs_baseline_metric: str = ""  # e.g. "R² delta: +0.03"
    literature_consensus_score: float = 0.0  # 0.0-1.0
    pathway_plausible: bool = True
    simulation_diverged: bool = False
    confidence_grade: str = "D"  # A/B/C/D
    warnings: list[str] = field(default_factory=list)


# ─── Explanation ────────────────────────────────────────────────────

@dataclass
class Explanation:
    mechanism_text: str = ""
    causal_paths: list[str] = field(default_factory=list)
    experimental_suggestions: list[str] = field(default_factory=list)


# ─── Final Output ───────────────────────────────────────────────────

@dataclass
class AgentState:
    """Complete agent state passed through workflow nodes."""

    # Step 0: Input
    raw_input: str = ""

    # Step 1: Parse
    query: Optional[ParsedQuery] = None

    # Step 2: Evidence
    evidence: Optional[EvidencePackage] = None

    # Step 3: Modeling
    pathway: Optional[PathwayModel] = None
    simulation: Optional[SimulationResult] = None
    ai_prediction: Optional[AIPrediction] = None
    baseline: Optional[SimpleBaseline] = None

    # Step 4: Validation
    validation: Optional[ValidationResult] = None

    # Step 5: Explain
    explanation: Optional[Explanation] = None

    # Meta
    hardware: Optional[HardwareInfo] = None
    steps_completed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Final report
    report_path: str = ""
