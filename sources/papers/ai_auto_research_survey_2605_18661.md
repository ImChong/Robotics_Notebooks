# AI for Auto-Research: Roadmap & User Guide（综述预印本）

- **类型**：论文（survey / roadmap）
- **收录日期**：2026-06-20
- **arXiv**：<https://arxiv.org/abs/2605.18661>
- **PDF**：<https://arxiv.org/pdf/2605.18661>
- **配套资源**：[Awesome AI Auto-Research 仓库](../repos/awesome-ai-auto-research.md) · [静态站点](../sites/awesome-ai-auto-research.md)

## 一句话

对 **AI 辅助学术研究全生命周期** 的端到端综述：按四阶段八阶段（Creation → Writing → Validation → Dissemination）整理工具、基准与方法族，并指出 **产物生成快于科学验证**、**人机共治协作** 比完全自主更可信的跨阶段规律。

## 为什么值得保留

- 给出可操作的 **生命周期 taxonomy**（8 stages / 4 phases），覆盖从 ideation 到 Paper2X 的完整链路，而非孤立讨论写作或 coding agent。
- 汇总 **250+ 论文、52+ 基准** 与代表性系统（The AI Scientist、FARS、ARIS、OpenScholar、PaperBench 等），适合作为本库 **LLM Wiki 维护模式** 与 **科研 agent 栈** 的外部索引。
- 明确 **阶段依赖的能力边界**：结构化、可检索、工具可验证的任务 AI 表现强；真正新颖的想法、研究级代码实现与科学判断仍脆弱。
- 强调 **治理已从「检测 AI」转向「披露、归因与问责」**——与本库 `schema/ingest-workflow.md` 的溯源要求同构。

## 核心摘录（面向 wiki 编译）

### 四阶段八阶段框架

| 阶段 | 子阶段 | 核心问题 |
|------|--------|----------|
| **P1 Creation** | S1 Idea · S2 Lit. Review · S3 Coding & Exp. · S4 Tables & Figures | 贡献是什么？证据如何产生？ |
| **P2 Writing** | S5 Paper Writing | 如何把产物组织成可审阅稿件？ |
| **P3 Validation** | S6 Peer Review · S7 Rebuttal & Revision | 社区如何质疑、辩护与修订？ |
| **P4 Dissemination** | S8 Paper2X（poster / slides / video / social / web / agent） | 如何向更广受众忠实传播？ |

生命周期 **非严格线性**：审稿可能要求回到实验；传播材料可能暴露写作中的歧义。

### 五类方法族（跨阶段复用）

1. **Prompt engineering** — 轻量、无训练；对措辞敏感、缺持久 grounding。
2. **RAG** — 文献/代码/日志 grounding；不保证源正确或忠实转述。
3. **Training-free agentic** — 规划、工具、记忆、自反思；风险在错误传播。
4. **Training-based** — 审稿/写作/代码等领域微调；依赖数据质量。
5. **Hybrid** — RAG + agent + 微调子模块；当前主流端到端系统形态。

### 五项核心发现（摘要口径）

1. **结构化、可外部核验** 的任务 AI 强；开放新颖、隐式领域知识、长程推理与科学判断任务骤降。
2. **产物生成 consistently 快于验证** — 想法、代码、文稿、审稿意见都可「看起来对」而实质错误。
3. **人机共治协作** 是最可信部署：AI 减机械摩擦，人保留判断、实验设计、论证与问责。
4. **有效系统收敛于分层架构**：探索（search）+ 执行（tools）+ 验证（execution/citation/critique/human）。
5. **AI 使用已是治理问题**：从检测转向披露、归因、责任界定。

### 阶段成熟度快照（综述 Table 1 口径）

| 子阶段 | 成熟度（★） | 主要张力 |
|--------|------------|----------|
| S1 Idea Gen. | ★★★★★ 工具多 | 新颖性–可行性鸿沟；实施后 idea 退化 |
| S2 Lit. Review | ★★★★★ 最快成熟 | 引用忠实度、多论文关系推理 |
| S3 Coding & Exp. | ★★★★★ SWE 强、研究代码弱 | ResearchCodeBench ~37%；语义错误占多数 |
| S4 Tables & Figures | ★★★★★ 工具相对欠发达 | 视觉可信 ≠ 科学正确 |
| S5 Writing | ★★★★★ 采用广 | 流畅 ≠ 论证深度；端到端自主难达顶会标准 |
| S6 Peer Review | ★★★★★ 有风险 | 单独 AI 审稿 lenient；ICLR 2025 AI 反馈改稿更可信 |
| S7 Rebuttal | ★★★★★ 相对欠服务 | 承诺修订 vs 实际兑现（commitment gap） |
| S8 Dissemination | ★★★★★ 信任瓶颈 | 公众材料可能过度简化或夸大 |

### 端到端系统四类架构（§7.1）

- **Sequential pipeline** — AI Scientist 类；简单但阶段间错误传播。
- **Search / self-improving** — AI Scientist v2、进化搜索；需可靠 evaluator 防 Goodhart。
- **Skill / tool-integrated** — ARIS、ResearchClaw；模块化但需共享可更新状态。
- **Multi-agent / community-scale** — VirSci、FARS；角色分离有助批判，但协调与共识错误仍存。

### 开放挑战（§7.4 归纳）

阶段边界忠实度 · 科学判断与新颖性 · 验证与可复现 · 引用版本一致性 · 治理与披露 · 跨域泛化 · 研究者认知所有权。

## 对 wiki 的映射

- 升格概念页：[AI Auto-Research（学术研究自动化）](../../wiki/concepts/ai-auto-research.md) — 生命周期框架、能力边界、分层架构与本库维护模式对照。
- 交叉补强：
  - [LLM Wiki（Karpathy 模式）](../../wiki/references/llm-wiki-karpathy.md) — 本库 ingest/query/lint 与综述 S2 文献综合、知识编译同源。
  - [schema/ingest-workflow.md](../../schema/ingest-workflow.md) — 运维规范即「人机共治」研究自动化实例。
  - [Hermes Agent](../../wiki/entities/hermes-agent.md)、[Agent Reach](../../wiki/entities/agent-reach.md)、[Superpowers（obra）](../../wiki/entities/superpowers-obra.md) — 工具/技能/外网读搜层实例。

## 参考来源（原始）

- arXiv 摘要与 PDF：<https://arxiv.org/abs/2605.18661> · <https://arxiv.org/pdf/2605.18661>（2026-06-20 抓取要点）
- 项目页：<https://worldbench.github.io/awesome-ai-auto-research>
- GitHub：<https://github.com/worldbench/awesome-ai-auto-research>
