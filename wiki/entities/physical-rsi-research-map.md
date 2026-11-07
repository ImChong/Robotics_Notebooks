---
type: entity
project_id: physical-rsi-research-map
papers: ["2610.18842"]
tags: [physical-ai, recursive-self-improvement, embodied-ai, research-map, literature-review, evaluation, safety, harness]
status: complete
updated: 2026-11-07
related:
  - ./physical-rsi.md
  - ./robodojo.md
  - ./paper-compounding-physical-rsi.md
  - ./paper-harnesspai.md
  - ./paper-harness-vla.md
  - ../queries/rsi-four-tier-five-pushes.md
sources:
  - ../../sources/sites/sail-physical-rsi-research-map.md
  - ../../sources/papers/compounding-physical-rsi_arxiv_2611_06654.md
original:
  - ../../sources/sites/sail-physical-rsi-research-map.md
summary: "SAIL Research Lab 的 Physical RSI 是一份具身系统递归自我改进研究地图与文献目录，提供 L0–L5 分类及 harness、训练、安全、验证和评测等主题索引；不是单一机器人策略论文或 Physical RSI 1.0 系统。"
---

# Physical RSI Research Map（SAIL Research Lab）

**Physical RSI — Physical Recursive Self-Improvement** 是 SAIL-Research-Lab 维护的具身递归改进研究地图。项目将自己定位为遵循配套论文证据边界的阅读索引，而非排行榜；README 的目录徽章标示 325 项 works/resources（2026-11-07 复核；10-14 核查时为 301）。其核心问题是：物理交互能否不仅改变机器人的当前行为，还能因果性地改变其产生、评估或筛选后续改进的机制。

| 项目 | 内容 |
|------|------|
| 主入口 | <https://github.com/SAIL-Research-Lab/Physical-RSI> |
| 维护方 | SAIL-Research-Lab |
| 文献索引 | [paper-reference-index.md](https://github.com/SAIL-Research-Lab/Physical-RSI/blob/main/docs/paper-reference-index.md) |
| 专题材料 | [model_training.md](https://github.com/SAIL-Research-Lab/Physical-RSI/blob/main/docs/model_training.md) |
| 许可 | MIT（仓库）；引用论文与第三方材料各循其许可 |
| 核查日期 | 2026-10-14；2026-11-07 重新通读 README、`model_training.md` 与三张框架图 |

## 一句话定义

**一份把具身系统递归自我改进（Physical RSI）相关文献按「改了什么、增益迁移到哪、如何验证」组织起来的、证据边界明确的阅读地图**，不是机器人策略，也不是排行榜。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RSI | Recursive Self-Improvement | 递归自我改进：改进会因果性地改变后续改进机制 |
| SAIL | SAIL Research Lab | 本研究地图的维护方 |
| MIT | MIT License | 仓库采用的开源许可证 |

## 为什么重要

- 「self-improving / autonomous / lifelong」这类标签在具身论文里被广泛使用，该地图给出可检查的判据：保留的变化是否进入了未来改进的生成、评估或选择机制，而不只是任务分数上升。
- 它把 harness 适应、模型训练、安全、验证与基准放进同一框架，便于判断 [HarnessPAI](./paper-harnesspai.md)、[Compounding Physical RSI](./paper-compounding-physical-rsi.md)、[Physical RSI 1.0](./physical-rsi.md) 这类工作处于 L0–L5 的哪一档。
- README 的 Open Questions 列出六个尚未解决的问题（见下文），可作为选题清单。

## 核心机制与边界

项目强调，不应把所有部署后适应都称作递归自我改进：收集更多数据、修复一次失败或微调策略，若没有保留的变化进入未来改进机制，仍只是适应。其框架把执行系统 \(M=(\Theta,H)\)（模型与 harness）与改进过程 \(\Phi=(Adapt,Train,Ver)\) 分开，并要求区分“任务表现变好”与“产生未来更新的过程变好”。

## L0–L5 能力分类

README 按更新对象与收益迁移范围组织能力层级：

| 层级 | 含义（项目 README 概述） |
|------|-------------------------|
| L0 | 没有自主且被保留的改进闭环 |
| L1 / L3 | 继承的执行系统更新分别改善同一任务 / 迁移到其他任务 |
| L2 / L4 | 继承的改进程序更新分别支持同任务 / 跨任务改进 |
| L5 | 连续的程序修订在匹配预算和受保护的跨任务评测下，使后续改进能力持续提升 |

L5 的表述强调验证和控制变量；仅仅换了程序或得到一次更高任务分数，并不足以证明元层级改进。

## 研究地图覆盖范围

仓库将文献和资源归到以下主题：

- Agent harness 与执行：记忆、知识管理、技能库、工具构建、规划工作流、失败诊断与恢复。
- 模型训练与适应：数据准备、模仿学习、强化学习、自监督表征、持续适应、知识迁移与训练程序优化。
- 安全与保障：安全探索、安全迁移、运行时控制、监督完整性与风险分类。
- 验证与评测：候选更新验证、评估器可靠性、改进结果评估。
- 基准、数据集与平台：共享机器人数据、任务泛化、真实世界鲁棒性、反馈可靠性与机器人研发基准。

### 框架图要点（README 三张图，2026-11-07 逐图阅读）

- **执行系统与改进过程：** 左图把任务执行系统（harness H + 参数化模型 Θ）与改进过程 Φ（Adapt、Train、Verification）分开，外加 Evaluator E、经验反馈 D 以及包住交互的 Safety and Control。
- **验证器四步：** 候选构建（在系统副本上应用提议的更新）→ 实现检查（语法、格式、接口、执行）→ 有效性检查（更新是否解决目标问题）→ 保持性检查（Grasp、Push、Stack 等原有能力前后对比），再加元验证；最终由独立的 Evaluator 用物理结果测试。
- **任务同一性：** 新场景不等于新任务——同任务集内 κ 相同、ξ 不同；同一批技能换顺序则属不同任务。

### README 的六个开放问题（引稿件第 7 节）

不确定物理经验中的学习信号；harness 各组件的因果归因；改进回路的非平稳性；奖励投机与评测泄漏；递归更新中的安全再验证；对「已验证的递归增益」做基准。README 注明这是稿件范围内的空白，不是「别处没有相关工作」。

## 工程实践

- **选读路径：** 先读 README 的 Boundary 与 Capability Map，再按主题（harness / 训练 / 安全 / 验证 / 基准）进入目录；`docs/model_training.md` 是训练一节的展开，`docs/paper-reference-index.md` 是稿件 v34 参考文献的审计镜像（R001–R292，含每条在稿件中的引用小节）。
- **用来给具体工作定位：** 对一篇声称「自我改进」的论文，逐项问：更新对象是 H、Θ 还是 Φ？收益只在同任务还是跨任务？验证器是否独立于改进过程？评测是否在匹配预算与受保护的跨任务集上？
- **跟进更新：** 目录随仓库持续增补（页首「Latest Papers」表列 2026-09-29 至 10-13 的七篇），引用数字前请回到仓库核对当前版本。

## 局限与读法

- **局限：** 这是一份持续维护的二级索引；目录所列研究不必然都符合 Physical RSI 的严格定义，也不构成统一协议下的性能比较。
- README 称其与一篇 accompanying paper 对齐；该论文为 arXiv:[2610.18842](https://arxiv.org/abs/2610.18842) *Who Improves the Robot? Toward Physical Recursive Self-Improvement*（摘要页列出的仓库即本项目），[Compounding Physical RSI](./paper-compounding-physical-rsi.md) 也引用它作为 Physical RSI 的定义来源。库内暂未为该论文单独建节点。
- 需要判断单项方法是否真正实现递归改进时，应回到原论文检查物理反馈、更新保留与继承、验证器独立性、跨任务评估和资源预算等证据。

## 关联页面

- [Physical RSI 1.0（HKU MMLab × Kinetix AI）](./physical-rsi.md) — 同名但不同项目的具体系统。
- [RoboDojo](./robodojo.md) — 项目地图中的机器人操作评测基准。
- [Compounding Physical RSI](./paper-compounding-physical-rsi.md) — 引用本框架配套论文的具体方法（arXiv:2611.06654）：阶段级失败诊断 → 目标条件离线 RL 修补 → 权重插值候选经 RoboDojo-Sim 全量评测晋升；只更新策略权重、改进算子固定，按本页 L0–L5 读法属「继承执行系统更新」一类，不是程序级元改进（该页推测约在 L1–L3）。
- [HarnessPAI](./paper-harnesspai.md) 与 [Harness VLA](./paper-harness-vla.md) — 相关 harness 研究。
- [RSI 四 tier 五 pushes](../queries/rsi-four-tier-five-pushes.md) — RSI 分类与选型框架。
- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 研究地图中的验证与评测、基准与平台主题可按该链的四层评测选型归位
- [google-grounding-ai-agents-wet-labs](google-grounding-ai-agents-wet-labs.md) — Google Research 博客：湿实验自主化的三条研究线（接地到机器人 / 适配体闭环 / 蛋白结合剂）（湿实验侧的自主化，非机器人本体）

## 推荐继续阅读

- 配套论文 [arXiv:2610.18842](https://arxiv.org/abs/2610.18842)（库内暂无节点）
- 仓库 [docs/model_training.md](https://github.com/SAIL-Research-Lab/Physical-RSI/blob/main/docs/model_training.md)
- [Physical RSI 1.0](./physical-rsi.md) 与 [Compounding Physical RSI](./paper-compounding-physical-rsi.md)：两个可按 L0–L5 读法对照的具体系统

## 同名项目消歧

本页对应 **SAIL Research Lab 的研究地图仓库**。香港大学 MMLab × Kinetix AI 的 [Physical RSI 1.0](./physical-rsi.md) 是另一具体系统与项目页，名称相似但主体、材料和主张均不同；不可将两者的论文、榜单成绩或开源状态混为一谈。

## 参考来源

- [SAIL Physical RSI 项目归档](../../sources/sites/sail-physical-rsi-research-map.md)
- [Compounding Physical RSI 论文归档](../../sources/papers/compounding-physical-rsi_arxiv_2611_06654.md) — 其参考文献 [16] 指向本项目配套论文
- [项目 README](https://github.com/SAIL-Research-Lab/Physical-RSI#readme)
- [参考文献索引](https://github.com/SAIL-Research-Lab/Physical-RSI/blob/main/docs/paper-reference-index.md)
