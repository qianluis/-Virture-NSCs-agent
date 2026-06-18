# VirtualCell-Agent System Architecture

_Generated: 2026-06-19_

## Overview

VirtualCell-Agent is a computational platform for **Neural Stem Cell Gene Regulatory Network (GRN) simulation and perturbation analysis**. It integrates ODE-based modeling, parameter fitting, AI prediction, and literature-driven validation into a unified pipeline.

## Architecture (3-Tier)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  TIER 1: DATA LAYER                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ scRNA-seq│  │Literature│  │Perturb-seq│  │SBML Mod.│  │Processor │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       └──────────────┴─────────────┴──────────────┘            │        │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
┌──────────────────────────────────────────────────────────────────────────┐
│  TIER 2: MODEL & SIMULATION                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│  │GRN Engine│◄──►│VirtualCel│◄──►│Parameter │    │AI Predict│            │
│  │22 genes  │    │Core      │    │Fitting   │    │or        │            │
│  │Hill ODEs │    │Perturb   │   │MCMC+NM   │    │DL-based  │            │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘            │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
┌──────────────────────────────────────────────────────────────────────────┐
│  TIER 3: VALIDATION & OUTPUT                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│  │Baseline   │    │Benchmark │    │Figures   │    │Output    │            │
│  │Check      │───►│Suite     │───►│(300 DPI) │───►│JSON/PNG  │            │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘            │
└──────────────────────────────────────────────────────────────────────────┘
```

## Skill Stack & Evaluation

Five VirtualCell skills form the agent interface to this system:

| Skill | Score | Role | Status |
|-------|:-----:|------|--------|
| `virtual-cell-perturbation` | **68/100** 🥇 | Perturbation design & validation | ✅ Leader |
| `virtual-cell-simulator` | **53/100** 🥈 | ODE simulation & parameter scan | ✅ Solid |
| `virtual-cell-literature` | **48/100** 🥉 | Literature search & curation | ✅ Solid |
| `virtual-cell-ai` | **40/100** | AI-based prediction | 🟡 Improving |
| `virtual-cell-data` | **40/100** | Data ingestion (scRNA-seq/SBML) | 🟡 Improving |

**Evaluation Framework:** [Darwin Skill](https://github.com/alchaincyf/darwin-skill) — 8-dimension rubric (structure + effectiveness), hill-climbing optimization.

## Key Results

**10 perturbations × 22 genes benchmark** — fold-change repression model achieves biological correctness:

| Perturbation | Correct Predictions | Key Finding |
|-------------|:------------------:|-------------|
| NOTCH1 KO | ✅ 5/5 | HES↓ → ASCL1↑ → NEUROGENESIS↑ |
| HES1 KO | ✅ 4/4 | ASCL1↑, neuronal diff↑ |
| ASCL1 KO | ✅ 4/4 | HES↑, neuronal diff↓ |
| CTNNB1 KO | ✅ 4/4 | MYC↓, CCND1↓, SOX2↓ |

## Diagrams (editable .drawio files)

Located in `diagrams/` — open in [app.diagrams.net](https://app.diagrams.net):

| File | Description |
|------|-------------|
| `system_architecture.drawio` | 3-tier system architecture |
| `grn_topology.drawio` | 22-gene GRN network with regulations |
| `mcmc_workflow.drawio` | Parameter estimation & validation pipeline |

## Publication Figures (300 DPI)

| File | Description |
|------|-------------|
| `parameter_fitting/fig1_perturbation_atlas.png` | Heatmap: 22 genes × 10 perturbations |
| `parameter_fitting/fig2_key_results.png` | NOTCH1 KO + Toggle + Validation |
| `parameter_fitting/fig3_grn_topology.png` | GRN network diagram |
| `parameter_fitting/benchmark_heatmap.png` | Full benchmark heatmap |

## Quick Start

```bash
# Baseline simulation
python -c "from core.grn_model import run_grn_simulation; run_grn_simulation()"

# Perturbation
python -c "from core.grn_model import run_grn_simulation; run_grn_simulation('NOTCH1', 'knock_out')"

# VirtualCell agent
python core/virtual_cell.py --gene NOTCH1 --perturbation knock_out

# Run full benchmark
python parameter_fitting/refine_and_benchmark.py
```

## Repository Structure

```
VirtualCell-Agent/
├── core/
│   ├── grn_model.py          # GRN ODE model (22 genes, Hill functions)
│   └── virtual_cell.py       # VirtualCell agent interface
├── parameter_fitting/
│   ├── mcmc_fit.py            # MCMC Bayesian fitting
│   ├── refine_and_benchmark.py# Nelder-Mead + benchmark runner
│   └── fig*.png              # Publication figures
├── scripts/
│   ├── run_ai_model.py        # AI prediction model
│   ├── parse_scdata.py        # scRNA-seq parser
│   ├── parse_sbml.py          # SBML parser
│   └── search_literature.py   # Literature search
├── diagrams/                   # Draw.io editable diagrams
│   ├── system_architecture.drawio
│   ├── grn_topology.drawio
│   └── mcmc_workflow.drawio
├── docs/
│   └── architecture.md         # This file
├── tests/
│   └── test_core.py
└── skills/                     # VirtualCell skills (5)
    ├── virtual-cell-ai/
    ├── virtual-cell-data/
    ├── virtual-cell-literature/
    ├── virtual-cell-perturbation/
    └── virtual-cell-simulator/
```
