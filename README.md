# VirtualCell-Agent

> 🧬 **神经干细胞（NSC）虚拟细胞智能体** — 输入靶点基因/药物，输出细胞级干预预测报告

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![GPU](https://img.shields.io/badge/GPU-8GB%2B%20Recommended-green)]()

---

## 📖 概述

**VirtualCell-Agent** 是一个开源的生物医学 AI 智能体，聚焦于**神经干细胞**（Neural Stem Cell, NSC）信号通路建模与干预效果预测。

它能帮你回答这样的问题：

> _"如果我敲除 NOTCH1，神经干细胞会发生什么变化？"_
> _"SHH 药物激活在脊髓损伤中是否有治疗潜力？"_
> _"SOX2 过表达对成体神经发生有何影响？"_

### 核心能力

| 能力 | 说明 |
|------|------|
| 📚 **文献检索** | 自动检索 PubMed/arXiv/Semantic Scholar |
| 🧪 **通路建模** | 内置 6 大 NSC 关键通路知识库（Notch/Wnt/SHH/BMP/MAPK/Hippo） |
| 📈 **ODE 仿真** | Tellurium 驱动的微分方程通路动力学仿真 |
| 🤖 **AI 预测** | 支持 scGPT/Geneformer 推理（需 GPU） |
| 🔬 **基线验证** | 强制比较 AI vs 简单加性模型，防止过拟合 |
| 📊 **置信度评级** | A/B/C/D 四级置信度评估 |
| 📝 **自动报告** | 生成格式化 Markdown 研究报告 |

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/yourname/VirtualCell-Agent.git
cd VirtualCell-Agent

# Core dependencies
pip install -r requirements.txt

# Optional: AI models (GPU recommended)
pip install torch transformers scgpt
```

### 运行

**方式 1：自然语言查询**
```bash
python -m core.agent --query "分析NOTCH1敲除对神经干细胞的影响"
```

**方式 2：结构化参数**
```bash
python -m core.agent --target SHH --perturbation drug --context "脊髓损伤"
```

### 示例输出

运行后，在 `output/` 目录生成完整的 Markdown 报告：
```
output/
├── report_notch1_20260618_103000.md
└── report_sox2_20260618_103500.md
```

---

## 🏗️ 架构

```
用户输入 (靶点 + 干预类型)
        │
        ▼
┌─────────────────────────────────────┐
│ ① 查询解析 (Query Parser)           │
│ → 识别靶点基因、干预类型、细胞类型    │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│ ② 证据收集 (Evidence Gathering)     │
│ → PubMed/arXiv 文献检索, 知识库查询  │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│ ③ 建模层 (Modeling Layer)           │
│ ├─ ③a 通路构建 (Pathway Modeling)   │
│ ├─ ③b ODE 仿真 (Simulation)         │
│ └─ ③c AI 预测 + 基线 (Prediction)   │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│ ④ 验证裁决 (Validation Layer)       │  ← 核心差异设计
│ → 基线对比 / 文献共识 / 置信度评级   │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│ ⑤ 解释报告 (Report Generation)      │
│ → Markdown 研究报告 + 实验建议      │
└─────────────────────────────────────┘
```

### 多级硬件自适应

| 级别 | 硬件 | 可用能力 |
|------|------|---------|
| 🚀 **L3** | 24GB+ GPU | 全部 AI 模型 + 仿真 + 文献 |
| ⚡ **L2** | 8-16GB GPU | scGPT/Geneformer + 仿真 + 文献 |
| 📱 **L1** | CPU-only | 文献 + 仿真 + 基线模型 |

---

## 📂 目录结构

```
VirtualCell-Agent/
├── README.md                    # 项目首页（本文）
├── LICENSE                      # MIT
├── requirements.txt             # Python 依赖
│
├── core/
│   ├── agent.py                 # Agent 主入口 + 完整工作流
│   ├── state.py                 # 状态数据结构
│   └── validator.py             # 基线验证 + 置信度评估
│
├── knowledge/
│   └── neural_stem_cell/        # 神经干细胞知识库
│       ├── marker_genes.md      # 标记物基因大全
│       ├── signaling_pathways.md # 6 大核心信号通路
│       └── disease_models.md    # 相关疾病模型
│
├── skills/                      # 5 个核心 Skill 封装
│   ├── literature/              # 文献检索
│   ├── data_parser/             # 数据解析（scRNA-seq / SBML）
│   ├── simulator/               # 通路仿真
│   ├── ai_predictor/            # AI 模型推理
│   └── perturbation/            # 扰动预测验证
│
├── tools/                       # 外部工具封装
│   ├── pubmed_tool.py           # PubMed API
│   ├── string_db_tool.py        # STRING DB
│   └── kegg_tool.py             # KEGG API
│
├── config/
│   ├── agent_config.yaml        # Agent 配置
│   └── skills_config.yaml       # Skill 启停
│
├── tests/                       # 单元测试
├── examples/                    # Jupyter Notebook 示例
├── output/                      # 报告输出目录
└── docs/                        # 文档
```

---

## 🧠 内置知识库

### 神经干细胞 6 大核心信号通路

| 通路 | 核心效应 | NSC 中的角色 |
|------|---------|-------------|
| **Notch** | HES1/HES5 → ASCL1↓ | 维持 NSC 池，侧向抑制 |
| **Wnt/β-catenin** | β-catenin → TCF/LEF → NEUROG2 | 神经发生的驱动器 |
| **SHH** | GLI1/2 → MYCN, CCND2 | 腹侧模式化，NSC 维持 |
| **BMP** | SMAD1/5+SMAD4 → ID1-4 | 维持静息态，胶质分化 |
| **MAPK/ERK** | RAS→RAF→MEK→ERK → c-Myc | 增殖与生存 |
| **Hippo/YAP** | YAP/TAZ → TEAD → CTGF | 机械力响应，接触抑制 |

### 核心标记物

| 类别 | 标记物 |
|------|--------|
| 经典 NSC | NES (Nestin), SOX2, PAX6, GFAP, PROM1 |
| 激活态 NSC | EGFR, ASCL1, Ki-67 |
| 静息态 NSC | GFAP+, Nestin+, EGFR− |
| 神经元分化 | DCX, TUBB3, NEUROD1, RBFOX3 |
| 星形胶质细胞 | S100B, ALDH1L1, AQP4 |
| 少突胶质细胞 | PDGFRA, OLIG2, SOX10, MBP |

---

## ⚠️ 设计哲学：诚实 > 漂亮

1. **AI 不比基线好 → 用基线**：不强行包装大模型结果
2. **无数据 → 明确标注数据缺失**：不假装有数据
3. **置信度透明**：A/B/C/D 四级始终展示
4. **硬件自适应降级**：无 GPU 不会报错，自动降级为文献+仿真模式
5. **所有预测标注"需 wet-lab 验证"**

---

## 🛠️ 依赖

### 核心（必须）
- Python 3.10+
- requests
- biopython
- arxiv
- numpy
- matplotlib
- python-libsbml
- tellurium
- networkx
- pysb

### 可选（AI 模型）
- torch
- transformers
- scgpt

---

## 📚 引用

如果您在研究中使用了 VirtualCell-Agent，请引用：

```bibtex
@software{VirtualCellAgent2026,
  author = {VirtualCell-Agent Contributors},
  title = {VirtualCell-Agent: A Neural Stem Cell Virtual Cell Agent},
  year = {2026},
  url = {https://github.com/yourname/VirtualCell-Agent}
}
```

---

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](docs/CONTRIBUTING.md) 了解指南。

### 路线图

- [x] v1.0 MVP: 核心工作流 + 神经干细胞知识库
- [ ] v1.1: 多细胞类型支持（小胶质细胞、星形胶质细胞）
- [ ] v1.2: 类器官建模集成
- [ ] v2.0: 完整 GPU 推理管线 + 因果推断

---

## 📄 许可证

[MIT License](LICENSE)

---

> **🧬 虚拟细胞，触手可及 — Virtual cells, within reach.**
