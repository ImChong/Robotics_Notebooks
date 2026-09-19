# Recursive Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research Loops

> 来源归档（ingest）

- **标题：** Recursive Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research Loops
- **类型：** paper / survey / recursive-self-improvement / llm-agents / ai-auto-research / governance
- **arXiv：** <https://arxiv.org/abs/2607.07663>（v2 2026-09-06；PDF：<https://arxiv.org/pdf/2607.07663>）
- **作者：** Mingguang Chen*（DeepGrounding）、Licheng Wang（AlphaAvatar）、Bo Qu（Illinois Institute of Technology）
- **对应作者：** deepgroundingai@gmail.com
- **页数 / 图：** 44 pages, 6 figures
- **配套资源：** [GitHub 语料与复现脚本](../repos/recursive-self-improvement-deepgrounding.md)
- **入库日期：** 2026-09-19
- **一句话说明：** 系统梳理 **1,250 篇** arXiv（2024–2026）自改进文献：按 **改进对象 × 闭环程度** 两轴 taxonomy 区分 **有界 self-refinement** 与 **开放式 RSI**；单列 **自评估（evaluator）** 为第四技术类，并给出 **验证层级** 与治理测量空白。

## 开源状态（步骤 2.5）

- **项目页：** 无独立 `*.github.io`；以 arXiv + GitHub 为复现入口。
- **官方仓库（2026-09-19 核查）：** <https://github.com/deepgrounding/recursive-self-improvement> — **已开源**（语料 CSV、分类脚本、图表与 `draft/main.md` 手稿源）。
- **旧链更正：** arXiv v1 Data availability 曾写 `github.com/bamboodrift/recursive_self_improvement`（私有/不可解析）；README 声明 **deepgrounding 仓为 canonical release**，下一版 arXiv 将更正引用。
- **结论：** **已开源（语料 + 脚本 + 手稿）** — 非端到端训练/Agent 系统代码。

## 为什么值得保留

- **消歧 self-X 词汇：** 把 self-refine / self-reward / self-play / self-evolve 等机制按 **改什么**（部署行为、训练策略、evaluator、研究过程）与 **谁验证**（human-in / human-on / closed）拆开 — 与本库 [rsi-four-tier-five-pushes](../../wiki/queries/rsi-four-tier-five-pushes.md) 四层标准、[Awesome RSI](../../wiki/entities/awesome-rsi.md) artifact 索引 **互补**：本文偏 **全谱系 survey + 验证层级**。
- **语料可审计：** 释放 `corpus_v2.csv`（1,250 条 taxonomy 标注）、种子 871 条 + 定向补充 379 条、分类规则与 per-paper overrides — 适合作为 RSI 文献 **lint / 遗漏检查** 的外部基准。
- **evaluator 作为一等公民：** 主张每条改进环都是「某信号可替代人类判断」的 claim；自改进强度 **跟踪验证层级**（形式验证器 → 执行反馈 → 学习 judge → 内在自评）。
- **与 Auto-Research 综述分工：** [AI Auto-Research（2605.18661）](./ai_auto_research_survey_2605_18661.md) 按 **学术生命周期八阶段**；本文按 **自改进机制与闭环** — 二者在 §6 Auto Research 与 [AI Auto-Research 概念页](../../wiki/concepts/ai-auto-research.md) 交叉。

## 核心摘录（面向 wiki 编译）

### 两轴 taxonomy（§2.2）

**轴 1 — 改什么（四类 + foundations）：**

| 类别 | § | 子线程（摘要） | 语料 |
|------|---|----------------|------|
| Deployment-time self-evolution | §3 | 输出 refine · TTT · harness/skill 进化 | 393（74% 2026） |
| Training-time self-iteration | §4 | self-reward RL · CoT 自训 · 自蒸馏 · self-play（含 zero-data）· 具身 | 340（69%） |
| Self-evaluation | §5 | judges · PRM · verifiers · rubrics · meta-eval | 318（82%） |
| Auto Research | §6 | AI scientist · 进化程序发现 | 139（76%） |
| Foundations, limits & safety | §7 | 理论 · 极限 · 安全 | 60（57%） |

**轴 2 — 闭环程度：**

- **Human-in-the-loop** — 人审每次改动（AI 辅助编码、co-scientist）。
- **Human-on-the-loop** — 自动 improvement signal + 人审计/门控部署（**语料主体**）。
- **Closed loop** — 系统自生成、自验证、自应用改进（Anthropic「closing the loop」极限情形）。

**中心切分：** **Bounded self-refinement**（固定外部 evaluator，可收敛可评测） vs **open-ended RSI**（连 evaluator 与改进 machinery 一并改写，原则上发散）。

### 验证层级（§5，survey 核心 regularity）

自上而下：**形式验证器**（sound）→ **执行反馈**（测试/编译器/基准，可靠但不完备）→ **学习 judge / RM**（受 judge 能力上限，可被 Goodhart）→ **内在信号**（最便宜、最易被 hack）。语料 **定性规律**：Demonstrated self-improvement strength **跟踪**该层级；FunSearch / AlphaEvolve 居上两层；大量 self-refine 依赖 **外部** execution/retrieval — 「intrinsic self-correction」已变 rare。

**方向设定瓶颈 = 验证瓶颈：** 研究 **执行** 可验证（代码跑、基准分）；**选题/品味** 属不可验证任务 — 与 Anthropic RSI essay 人侧瓶颈同构；层级 **索引** verification，**不** 索引「什么值得评估」。

### 语料与方法（§2.3）

- **种子：** 7 条 arXiv 检索线 → 871 篇 + OpenAlex 元数据；规则重分类移 89 篇 off default。
- **补充：** 379 篇（self-eval、TTT、zero-data self-play 等 taxonomy 一等方向）。
- **局限（作者自述）：** 非 census；2026 占 74%、引用近零；单标注者；~54/379 补充为 peripheral bleed；growth 统计仅用 seed corpus。

### Auto Research 与批判文献（§6 要点）

- **闭环信号：** A-Evolve-Training 自报 30B **无人在环** post-training 四轮（dev metric 与外部 leaderboard 脱钩时自改搜索策略）。
- ** skeptical 文献同级成熟：** Agentic AI scientists 不适配自主发现；Dead Science Walking（出版偏置 × 机器速度）；human oversight 结构决定可靠性（经济学预注册研究）。

### 开放问题（§8 压缩）

- 跨 episode **方法连续性**（scientific amnesia 的反面）— taxonomy 按 substrate 索引，难记录「同一系统跨 cell 的轨迹」。
- **Governance-grade measurement** of self-improvement — 字段 **最 underpopulated** 的 niche。

## 对 wiki 的映射

- **升格实体页：** [`wiki/entities/paper-rsi-survey-2607-07663.md`](../../wiki/entities/paper-rsi-survey-2607-07663.md)
- **仓库归档：** [`sources/repos/recursive-self-improvement-deepgrounding.md`](../repos/recursive-self-improvement-deepgrounding.md)
- **概念互链：**
  - [`wiki/concepts/recursive-self-improvement.md`](../../wiki/concepts/recursive-self-improvement.md) — 宏观 RSI / Anthropic 连续体
  - [`wiki/queries/rsi-four-tier-five-pushes.md`](../../wiki/queries/rsi-four-tier-five-pushes.md) — 四层标准 × 五次推进（Datawhale 叙事）
  - [`wiki/concepts/ai-auto-research.md`](../../wiki/concepts/ai-auto-research.md) — 学术生命周期综述（2605.18661）
  - [`wiki/entities/awesome-rsi.md`](../../wiki/entities/awesome-rsi.md) — agent artifact × mode 策展

## 参考来源（原始）

- arXiv 摘要与 PDF：<https://arxiv.org/abs/2607.07663> · <https://arxiv.org/pdf/2607.07663>（2026-09-19 全文要点）
- GitHub：<https://github.com/deepgrounding/recursive-self-improvement>
