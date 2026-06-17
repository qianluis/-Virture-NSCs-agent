# 🧬 VirtualCell-Agent 架构设计 v1.0

> 基于髓核细胞（Nucleus Pulposus）智能体场景 · 2026.06
> 目标：GitHub 发布开源智能体，实现靶点发现 → 通路建模 → 干预预测 → 报告生成的端到端自动化

---

## 一、Core Identity：这个 Agent 是谁？

| 属性 | 定义 |
|------|------|
| **名称** | VirtualCell-Agent |
| **定位** | 生物医学领域的"虚拟细胞实验员"——输入靶点/药物，输出细胞级干预预测报告 |
| **首批聚焦** | **髓核细胞（Nucleus Pulposus Cell）** — 椎间盘退变（IVDD）的核心细胞类型 |
| **能力边界** | 知识检索 ✅ 数据解析 ✅ 通路仿真 ✅ AI 预测 ✅ 因果解释 ✅ 图文报告 ✅ |
| **不做的事** | 不替代 wet-lab 验证，不进行分子动力学模拟，不处理临床数据合规 |
| **发布平台** | GitHub（MIT License） |
| **Agent 框架** | LangGraph（Python） |

---

## 二、核心工作流（Workflow）

### 2.1 总架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                        用户入口                                    │
│  [输入：靶点基因/药物名 + 细胞类型（默认髓核细胞）]                  │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                         ① Query Parser                            │
│  LLM 解析：靶点是什么？要预测敲除/过表达/药物？期望什么输出？      │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
            ╔═══════════════════════════════════╗
            ║      ② 证据收集层 (Evidence)      ║
            ╠═══════════════════════════════════╣
            ║  文献检索 → 数据库查询 → 知识图谱  ║
            ╚═══════════════════════════════════╝
                           ▼
            ╔═══════════════════════════════════╗
            ║      ③ 数据建模层 (Modeling)      ║
            ╠═══════════════════════════════════╣
            ║  通路构建 → 仿真模拟 → AI 预测    ║
            ╚═══════════════════════════════════╝
                           ▼
            ╔═══════════════════════════════════╗
            ║      ④ 验证裁决层 (Validation)    ║
            ╠═══════════════════════════════════╣
            ║  基线对比 → 置信度评估 → 冲突检测  ║
            ╚═══════════════════════════════════╝
                           ▼
            ╔═══════════════════════════════════╗
            ║      ⑤ 解释输出层 (Explain)       ║
            ╠═══════════════════════════════════╣
            ║  因果推理 → 机制解释 → 报告生成    ║
            ╚═══════════════════════════════════╝
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                        输出：Markdown 研究报告                     │
│  [含文献证据 / 通路图 / 预测结果 / 置信度评级 / 实验建议]          │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Step-by-Step 详细工作流

#### Step 0：用户输入解析

```python
# 输入示例
input = {
    "target": "TGFB1",                # 靶点基因/蛋白
    "perturbation_type": "overexpression",  # knock_out | overexpression | drug
    "cell_type": "nucleus_pulposus",   # 默认髓核细胞
    "context": "椎间盘退变模型",       # 疾病背景
    "output_format": "report"          # report | json | visualization
}
```

LLM 对输入进行意图分类和参数补全，确保下游模块获得结构化输入。

#### Step 1：证据收集层（Evidence Gathering）

**并行执行 3 个子任务：**

| 子任务 | 调用 Skill | 输出 |
|--------|-----------|------|
| **文献检索** | `virtual-cell-literature` | PubMed/arXiv/Semantic Scholar 中该靶点在髓核细胞中的研究论文摘要列表 |
| **数据库查询** | 内置工具 | 从 STRING、GeneCards、KEGG 获取蛋白质互作、通路注释、已知功能 |
| **知识图谱匹配** | 内置知识库 | 髓核细胞特有的 ECM 基因集（ACAN、COL2A1、SOX9 等）和退变标记物 |

**输出标准**：结构化为 `EvidencePackage` 对象，包含论文关键发现、通路 ID、共表达网络。

#### Step 2：数据建模层（Modeling）

**串行执行，后一步依赖前一步：**

```
  检索到的通路信息
        │
        ▼
  ┌──────────────┐
  │ ②a 通路构建   │  ← 调用 virtual-cell-data Skill
  │               │     构建包含靶点的髓核细胞信号通路模型
  │               │     （TGF-β / Wnt / MAPK / NF-κB 等）
  └──────┬───────┘
         │ SBML 通路模型
         ▼
  ┌──────────────┐
  │ ②b ODE 仿真   │  ← 调用 virtual-cell-simulator Skill
  │               │     运行 Tellurium/RoadRunner 仿真
  │               │     模拟干预前后各节点浓度/活性变化
  └──────┬───────┘
         │ 仿真曲线 + 稳态变化
         ▼
  ┌──────────────┐
  │ ②c AI 预测    │  ← 调用 virtual-cell-ai Skill
  │               │     运行 scGPT/Geneformer 做
  │               │     全基因组范围扰动表达预测
  └──────┬───────┘
         │ 全基因表达变化向量
```

#### Step 3：验证裁决层（Validation）— **核心差异化设计**

这是 VirtualCell-Agent 区别于"套壳 AI"的关键：

| 验证项 | 方法 | 作用 |
|--------|------|------|
| **简单基线对比** | 计算加性模型（双基因 = 单基因效应之和）作为基线 | 防止大模型给出比简单规则更差的预测 |
| **文献共识检查** | 将 AI 预测结果与文献检索发现做一致性匹配 | 检测与已知知识严重冲突的预测 |
| **通路合理性检查** | 仿真结果是否违背热力学/质量守恒 | 检测 ODE 模型数值发散或稳态异常 |
| **置信度评级** | 综合以上 3 项给出 A/B/C/D 四级 | 让用户知道结果可信度 |

**裁决规则**：
```python
if ai_prediction not significantly better than baseline:
    use baseline + flag "AI未优于简单基线"
if prediction contradicts known literature (conflict_score > 0.7):
    downgrade confidence + 注明文献冲突
if ODE simulation diverges (NaN/inf in results):
    exclude simulation results + warn
```

#### Step 4：解释输出层（Explain & Report）

| 模块 | 内容 |
|------|------|
| **机制解释** | LLM 基于通路图 + 预测结果，用自然语言解释"为什么这个靶点会产生这个效果" |
| **因果推断** | （可选）基于 DoWhy 尝试拆解关键因果路径 |
| **实验建议** | 推荐验证实验：qPCR 验证哪些基因、WB 验证哪些蛋白、建议使用的细胞系 |
| **风险标注** | 标注置信度低、文献冲突、或基线未超越的部分 |
| **可视化** | 通路图（可选 Mermaid/Graphviz 文本拓扑图），预测火山图（matplotlib） |

---

## 三、外部工具 / Plugins 清单

### 3.1 必需工具（部署时必须配置）

| 类别 | 工具名称 | 用途 | 许可证 | 配置需求 |
|------|---------|------|--------|---------|
| **文献检索** | BioPython Entrez | PubMed 检索 | ✅ 开源 | 邮箱（用于 NCBI） |
| | arXiv API | 预印本检索 | ✅ 免费 | 无 |
| | Semantic Scholar API | 论文引用/影响力分析 | ✅ 免费 | API Key（可选） |
| **数据解析** | scanpy + anndata | 单细胞数据读取/处理 | ✅ BSD | Python 3.10+ |
| | python-libsbml | SBML 格式解析 | ✅ LGPL | C 编译依赖 |
| | pysb | BioNetGen 规则建模 | ✅ BSD | 无 |
| | networkx | 通路拓扑分析 | ✅ BSD | 无 |
| **通路仿真** | tellurium (RoadRunner) | ODE/随机仿真引擎 | ✅ MIT | JIT 编译器 |
| | matplotlib | 仿真曲线/火山图 | ✅ BSD | 无 |
| **AI 模型** | torch + transformers | 深度学习框架 | ✅ BSD | GPU（推荐） |
| | scGPT (huggingface) | 单细胞基础模型 | ✅ MIT | 8GB+ GPU |
| | Geneformer (huggingface) | 小参数量单细胞模型 | ✅ MIT | 4GB+ GPU |
| **因果推断** | dowhy | 因果效应估算 | ✅ MIT | 无 |
| **报告生成** | markdown + matplotlib | 最终输出 | ✅ 内置 | 无 |

### 3.2 可选工具（按需接入）

| 工具 | 场景 | 安装复杂度 |
|------|------|-----------|
| **BioSimulators** | 统一的仿真调度层（同时跑 RoadRunner + COPASI） | ⭐⭐⭐ 中等 |
| **Neo4j + BioPAX** | 持久化知识图谱（大规模通路整合） | ⭐⭐⭐⭐⭐ 高 |
| **Squidpy** | 空间组学分析（当用户输入包含空间坐标时） | ⭐⭐ 低 |
| **GEARS** | 图神经网络扰动预测（组合扰动场景） | ⭐⭐⭐ 中 |
| **Lingshu-Cell** | 细胞扩散生成模型（虚拟细胞生成） | ⭐⭐⭐⭐ 高，需 24GB+ GPU |

### 3.3 数据源（必须配置的远程服务）

| 数据源 | URL | 用途 | 限额 |
|--------|-----|------|------|
| PubMed E-utilities | https://eutils.ncbi.nlm.nih.gov | 文献检索 | 3 req/s（无密钥） |
| STRING DB API | https://string-db.org/api | 蛋白互作网络 | 10 req/s |
| GeneCards | https://www.genecards.org | 基因功能注释 | 限制较严格 |
| KEGG API | https://rest.kegg.jp | 通路注释 | 无硬限 |
| CellMarker 2.0 | http://cellmarker.biocuckoo.cn | 细胞类型标记物数据库 | 免费 |

---

## 四、当前最大技术瓶颈 & Prompt 规避策略

### 🚨 瓶颈 1：AI 模型在扰动预测上未必优于简单基线

**严重程度**：⭐⭐⭐⭐⭐（最高）

**问题本质**：多个独立 2025-2026 基准测试反复证实——在基因扰动预测任务中，scGPT、Geneformer 等复杂模型的预测质量，**常常不优于**一个简单的加性模型（即双基因扰动效应 = 两单基因效应之和）。这意味如果 Agent 盲目信任"大模型更优"，会给出不如简单规则的预测结果。

**Prompt 规避策略**（在 System Prompt 中固化）：

```yaml
## ⚠️ 强制规则：基线验证
- 在任何 AI 模型运行前，Agent 必须先计算简单基线（加性模型/平均值模型）
- 只有当 AI 预测显著优于基线（R² 提升 > 0.05 或 MSE 下降 > 10%）时，才能使用 AI 结果
- 如果 AI 未超越基线，输出中必须明确标注："AI预测未优于简单基线，以下结果基于加性模型"
- 绝不能向用户呈现"大模型更优越"的误导性叙事
```

---

### 🚨 瓶颈 2：干预数据极度稀缺（髓核细胞更是重灾区）

**严重程度**：⭐⭐⭐⭐

**问题本质**：虚拟细胞预测的核心是"干预后的全基因组响应"，但绝大多数公开数据是**观测性**的（稳态单细胞图谱），**干预性**数据（Perturb-seq / CRISPR 敲除后的表达谱）非常有限。髓核细胞尤其——几乎没有公开的 Perturb-seq 数据。

**Prompt 规避策略**：

```yaml
## ⚠️ 强制规则：数据质量透明度
- 每次生成预测时，Agent 必须报告所用数据的来源和类型：
  - ✅ "基于髓核细胞真实 Perturb-seq 数据" → 置信度 A
  - ⚠️ "基于近缘细胞类型（如软骨细胞）数据迁移" → 置信度 B，标注迁移来源
  - ❌ "基于通用细胞基础模型（无髓核特异性微调）" → 置信度 C，标注无髓核特异数据
- 如果缺乏该细胞类型的干预数据，Agent 必须在报告开头明确警示："注意：当前预测基于泛细胞模型，未经过髓核细胞特异性验证"
- Agent 被明确禁止：在缺乏数据时假装有数据（例如不许把软骨细胞数据冒充为髓核细胞数据）
```

---

### 🚨 瓶颈 3：SBML / 通路建模的标准化成本高

**严重程度**：⭐⭐⭐

**问题本质**：同一个通路在不同数据库（KEGG / Reactome / WikiPathways）中的表示格式、命名空间、甚至物种名称都不一致。从文献中手动构建 SBML 模型需要领域专家介入，无法全自动化。

**Prompt 规避策略**：

```yaml
## ⚠️ 强制规则：通路建模的诚实性原则
- Agent 必须优先使用已有公开 SBML 模型（BioModels Database），禁止虚构通路模型
- 如果在 BioModels 中找不到目标通路和细胞类型的 SBML 模型，Agent 必须：
  1. 告知用户"该通路的标准化 SBML 模型尚未公开可用"
  2. 退而使用 KEGG 通路拓扑（路径图）做定性分析
  3. 明确指出："以下为定性通路分析，非定量 ODE 仿真结果"
- 定量仿真（ODE）必须明确标注：模型来源、参数来源、与真实数据的拟合度
```

---

### 🚨 瓶颈 4：可重复性危机——大模型推理随机性

**严重程度**：⭐⭐⭐

**问题本质**：同样的输入到 LLM Agent，可能因为模型采样温度、随机种子等因素产生不同的推理路径和最终报告，这在科研场景下是不可接受的。

**Prompt 规避策略**：

```yaml
## ⚠️ 强制规则：推理确定性
- 对所有 LLM 调用设置 temperature = 0（确定性模式）
- 所有 AI 模型推理设置固定随机种子（seed = 42）
- Agent 必须记录每一步的调用参数（模型名、版本、seed）到输出的元数据中
- 重要输出必须包含可复现摘要：
  ```markdown
  ## 复现信息
  - Agent 版本：v1.0
  - 文献检索时间：2026-06-18
  - AI 模型：scGPT (commit xxx), seed=42
  - 仿真引擎：RoadRunner v2.x, SBML 模型来自 BioModels: MODEL_xxxx
  ```
```

---

### 🚨 瓶颈 5：GPU 资源门槛与部署成本

**严重程度**：⭐⭐⭐

**问题本质**：scGPT 推理需要 8GB+ GPU，Lingshu-Cell 需要 24GB+，X-Cell 需要 80GB。如果用户没有 GPU 环境，大部分 AI 能力无法运行。

**Prompt 规避策略**：

```yaml
## ⚠️ 强制规则：硬件自适应降级
- Agent 启动时必须检测 GPU 可用性并自动分级：
  - 🚀 Level 3（24GB+ GPU）：启用全部 AI 模型
  - ⚡ Level 2（8-16GB GPU）：仅启用 scGPT / Geneformer 推理
  - 📱 Level 1（无 GPU / CPU only）：禁用 AI 模型，仅使用文献检索 + 通路仿真 + 基线模型
- 在 Level 1 模式下，Agent 必须告知用户："当前环境无 GPU，AI 模型预测已禁用。输出仅基于文献证据和通路仿真。"
- 不能在没有 GPU 时假装加载了模型
```

---

## 五、系统 Prompt 核心设计（System Prompt 骨架）

以下是该 Agent 的 System Prompt 关键段落，已嵌入上述所有规避策略：

```markdown
# VirtualCell-Agent System Prompt

## 角色定义
你是一位虚拟细胞（Virtual Cell）智能体，专精于髓核细胞（Nucleus Pulposus）信号通路建模与干预预测。你的用户通常是生物医学研究员，需要你帮助他们回答"如果我对某个靶点进行干预，细胞会怎样变化？"

## 核心工作流
1. 解析用户输入 → 2. 检索文献证据 → 3. 构建/查询通路模型 → 4. 运行仿真/AI预测 → 5. 基线验证 → 6. 生成报告

## 强制约束（必须严格遵守）

### C1：置信度透明
- 必须明确标注预测结果基于何种数据源：髓核细胞真实数据 / 近缘细胞迁移 / 泛细胞通用模型
- 缺乏数据时必须如实告知，禁止假装有数据

### C2：基线验证优先
- AI模型运行前必须计算简单基线
- 仅当AI显著优于基线时才能使用AI结果
- 未超越基线时必须使用基线结果并标注

### C3：可复现性
- temperature = 0（确定性输出）
- 记录所有模型调用的版本和随机种子
- 输出包含完整的复现信息块

### C4：硬件自适应
- 检测GPU可用性并自动降级
- 无GPU时禁用AI模型，仅做文献+仿真+基线
- 不能在没有GPU时假装加载了模型

### C5：诚实的科学边界
- 不虚构没有公开来源的通路模型
- 定量仿真必须标注模型和参数来源
- 所有预测标注"需wet-lab验证"，不替代实验
```

---

## 六、GitHub 仓库结构建议

```
VirtualCell-Agent/
├── README.md                    # 项目介绍 + 快速开始 + 架构图
├── LICENSE                      # MIT License
├── requirements.txt             # 依赖总表
├── setup.py / pyproject.toml    # 安装脚本
│
├── config/
│   ├── agent_config.yaml        # Agent 配置（模型路径、API Key 等）
│   └── skills_config.yaml       # Skill 启停配置
│
├── core/
│   ├── agent.py                 # Agent 主类（LangGraph 编排）
│   ├── workflow.py              # 工作流节点定义
│   ├── state.py                 # Agent 状态数据结构
│   └── validator.py             # 基线验证 + 置信度评估
│
├── skills/                      # 5 个 Skill 的实现
│   ├── literature/              # virtual-cell-literature
│   ├── data_parser/             # virtual-cell-data
│   ├── simulator/               # virtual-cell-simulator
│   ├── ai_predictor/            # virtual-cell-ai
│   └── perturbation/            # virtual-cell-perturbation
│
├── knowledge/                   # 内置知识库
│   ├── nucleus_pulposus/        # 髓核细胞专项知识
│   │   ├── marker_genes.md      # 髓核细胞标记物基因列表
│   │   ├── signaling_pathways.md # 主要信号通路汇总
│   │   └── degeneration_markers.md # 退变标记物
│   └── sbml_models/             # 预制 SBML 模型
│
├── tools/                       # 外部工具封装
│   ├── pubmed_tool.py           # PubMed 检索封装
│   ├── string_db_tool.py        # STRING DB 查询
│   ├── kegg_tool.py             # KEGG API 封装
│   └── biogrid_tool.py          # BioGRID 蛋白互作
│
├── output/                      # 输出目录
│   └── templates/               # 报告模板 Markdown
│
├── tests/
│   ├── test_workflow.py         # 工作流测试
│   ├── test_validator.py        # 基线验证测试
│   └── test_data_parser.py      # 数据解析测试
│
├── examples/
│   ├── example_tgfb1.ipynb      # TGFB1 靶点分析示例
│   └── example_col2a1.ipynb     # COL2A1 靶点分析示例
│
└── docs/
    ├── architecture.md          # 架构文档（本文）
    ├── skill_reference.md       # Skill 调用参考
    └── quickstart.md            # 快速开始指南
```

---

## 七、髓核细胞专项知识快照（内置知识库核心内容）

| 知识点 | 内容 | 来源 |
|--------|------|------|
| **髓核细胞标记物** | ACAN, COL2A1, COL9A2, SOX9, KRT19, FOXF1, CA12, PAX1 | CellMarker 2.0 |
| **退变上调基因** | MMP3, MMP13, ADAMTS4, ADAMTS5, IL1B, TNF, NGF, BDNF | PubMed 综述 |
| **退变下调基因** | ACAN, COL2A1, SOX9, SHH, FOXF1 | PubMed 综述 |
| **关键信号通路** | TGF-β/BMP（ECM 合成调控）、Wnt/β-catenin（退变关键驱动）、MAPK/ERK（炎症响应）、NF-κB（炎症核心通路）、Hedgehog（发育维持） | KEGG + Reactome |
| **常用细胞模型** | 人原代髓核细胞（NP cells）、大鼠髓核细胞系、牛尾椎间盘器官培养 | 文献 |
| **常用动物模型** | 大鼠尾椎针刺模型、小鼠腰椎不稳模型、兔间盘退变模型 | 文献 |

---

## 八、快速启动命令（MVP 原型）

```bash
# 1. 克隆仓库
git clone https://github.com/yourname/VirtualCell-Agent.git
cd VirtualCell-Agent

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置
cp config/agent_config.example.yaml config/agent_config.yaml
# 编辑：填入 ENTREZ_EMAIL 等

# 4. 运行示例
python core/agent.py --target TGFB1 --cell-type nucleus_pulposus --perturbation overexpress

# 5. 预期输出
# output/report_tgfb1_20260618.md  ← 完整研究报告
```

---

## 九、总结：架构设计决策树

```
用户输入一个靶点
        │
        ▼
┌─────────────────────────────────────┐
│ 有 GPU 吗？                          │
├────────────┬────────────────────────┤
│  是 (L3)   │  否 (L1)               │
│  跑 AI 模型 │  仅文献 + 仿真 + 基线   │
└──────┬─────┴───────────┬────────────┘
       ▼                 ▼
┌──────────────────────────────┐
│ AI 预测优于基线吗？            │
├──────────┬───────────────────┤
│  是 ✅   │  否 → 用基线      │
└────┬─────┴───────────────────┘
     ▼
┌──────────────────────────────┐
│ 有髓核细胞特异性干预数据吗？  │
├──────────┬───────────────────┤
│  是 ✅   │  否 → 降置信度    │
└────┬─────┴───────────────────┘
     ▼
┌──────────────────────────────┐
│ SBML 模型可用吗？            │
├──────────┬───────────────────┤
│  是 ✅   │  否 → 用定性通路  │
│  定量仿真 │   + 标注限制     │
└────┬─────┴───────────────────┘
     ▼
   最终报告（含置信度评级）

---

> **版本**：v1.0 · 2026.06
> **场景**：髓核细胞智能体 · GitHub 开源发布
> **作者建议**：MVP 阶段建议先发 L1（无 GPU）版本，以文献+仿真+基线为主力，逐步加入 AI 预测能力。这样用户无需 GPU 即可体验核心工作流，降低准入门槛。
