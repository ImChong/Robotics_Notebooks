# Awesome RSI（Prism-Shadow）

- **URL**：<https://github.com/Prism-Shadow/awesome-rsi>
- **类型**：精选列表 / 交互式策展站点配套仓库
- **维护方**：Prism-Shadow
- **收录日期**：2026-09-15
- **Tags**：#recursive-self-improvement #llm-agents #agent-harness #self-evolution #benchmarks #literature-review

## 一句话

围绕 **Recursive Self-Improvement（RSI）** 维护的 **方法、基准与系统** 策展清单：按 **RSI artifact**（改什么）与 **RSI mode**（在线/离线评测协议）组织 50+ 方法论文、29 基准与 2 个系统，配套双语静态站与引用图谱。

## 为什么值得保留

- **taxonomy 清晰**：把「自改进」从口号拆成可检索维度 — **Model parameters / Harness code / Context / Memory / Skill / Other artifacts**，避免把 prompt 进化、harness 自改与权重更新混为一谈。
- **方法与基准对齐**：基准按 **Online / Offline / Offline→Online** 分组，并标注被评测 workflow 中演化的 artifact；适合对照 [HarnessBank](../../wiki/entities/paper-harnessbank.md)、[MetaRSI-v1](../../wiki/entities/paper-metarsi-v1.md) 等单篇工作。
- **站点可筛选**：<https://prism-shadow.github.io/awesome-rsi/#methods> 支持多维度 tag 组合（artifact、topology、feedback 等）、引用图与中英切换；仓库 README 为生成式目录快照。
- **与本库 RSI 叙事互补**：[递归自改进概念页](../../wiki/concepts/recursive-self-improvement.md) 侧重 Anthropic 宏观 RSI 与具身跟随假设；本清单聚焦 **agent 层可操作的自进化文献**。

## 核心内容（结构级）

### RSI 闭环定义（站点 / README 共识）

Agent 执行任务 → 从轨迹与反馈学习 → **更新自身状态** → 在后续任务中使用更新后的状态。状态可以是：模型参数、harness 代码、上下文、记忆或技能。

### Methods & Systems（按 RSI artifact）

| Artifact | 含义 | 代表条目（README 摘录） |
|----------|------|-------------------------|
| Model parameters | 权重参与后续任务 | Self-Adapting Language Models；Reef |
| Harness code | 可执行 agent / 控制流 / 工具代码被修改 | Gödel Agent；Darwin Gödel Machine；Proteus；HarnessEvolve |
| Context | prompt / 工作上下文更新 | GEPA；ReasoningBank；Agentic Context Engineering |
| Memory | 跨步/跨任务存储与检索经验 | A-MEM；Agent Workflow Memory；Dynamic Cheatsheet |
| Skill | 可复用策略/流程/技能资源演化 | SkillSmith；TRACE；WikiSkill |
| Other artifacts | 数据策略、实验配置、任务解等 | 主要在 benchmark 侧体现 |

### Benchmarks（按 RSI mode）

| Mode | 协议要点 | 代表条目 |
|------|----------|----------|
| Online | 任务流中累积经验并用于后续 | LifelongAgentBench；Evo-Memory；ContinualSkillBench |
| Offline | 演化与 held-out 评测分离 | HarnessDev；Evo-Bench；SEAGym；MemoryBench |
| Offline → Online | 离线构建后继续在线演化 | SkillFlow |

### 站点能力（2026-09-15 核查）

- **Methods 页**：<https://prism-shadow.github.io/awesome-rsi/#methods> — 多 tag 筛选、排序、摘要展开。
- **Citation graph**：methods / benchmarks 分图，可搜索节点。
- **Understanding RSI** 导读博客（中英）。
- **规模（生成计数）**：50 method papers · 29 benchmark papers · 2 systems。

## 开源边界（步骤 2.5）

| 已发布 | 不适用 |
|--------|--------|
| Markdown 策展 + 静态站源码（GitHub Pages） | 统一训练/评测代码栈（清单性质） |
| 部分条目链到各自官方仓库（如 Proteus、Reef） | 单篇论文权重/数据（逐条核项目页） |

清单为 **文献与基准导航**；复现须跳转各论文/项目页。

## 对 wiki 的映射

- 主沉淀：[Awesome RSI（Prism-Shadow 精选集）](../../wiki/entities/awesome-rsi.md)
- 概念交叉：[递归自改进（RSI）](../../wiki/concepts/recursive-self-improvement.md) · [AI Auto-Research](../../wiki/concepts/ai-auto-research.md)
- 站点镜像：[awesome-rsi-github-io.md](../sites/awesome-rsi-github-io.md)
- 相关实体：[RSI-Harness](../../wiki/entities/rsi-harness.md) · [HarnessBank](../../wiki/entities/paper-harnessbank.md) · [MetaRSI-v1](../../wiki/entities/paper-metarsi-v1.md) · [karpathy/autoresearch](../../wiki/entities/karpathy-autoresearch.md)

## 参考来源（原始）

- 仓库 README：<https://github.com/Prism-Shadow/awesome-rsi>（2026-09-15 结构级摘录）
- Methods 页：<https://prism-shadow.github.io/awesome-rsi/#methods>
