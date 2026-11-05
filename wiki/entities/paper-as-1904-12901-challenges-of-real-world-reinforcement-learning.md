---
type: entity
tags: [paper, curated-index, awesome-sim2real, longchao-sim2real]
status: complete
updated: 2026-10-06
arxiv: "1904.12901"
summary: "LongchaoDa AwesomeSim2Real 收录；分组 Action / Action Delay。 本页是清单索引，细节以原文为准。"
related:
  - ../entities/awesome-sim2real.md
  - ../overview/lc-awesome-sim2real-technology-map.md
  - ../concepts/sim2real.md
  - ../methods/reinforcement-learning.md
  - ../tasks/locomotion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/lc_awesome_sim2real_1904_12901_challenges-of-real-world-reinforcement-l.md
  - ../../sources/papers/lc_awesome_sim2real_catalog.md
  - ../../sources/repos/awesome-sim2real.md
---

# Challenges of real-world reinforcement learning

**Challenges of real-world reinforcement learning** 收录于 [AwesomeSim2Real](https://github.com/LongchaoDa/AwesomeSim2Real) **第 003/139** 篇，分组 **Action / Action Delay**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

LongchaoDa AwesomeSim2Real 收录；分组 Action / Action Delay。 本页是清单索引，细节以原文为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移到真机 |
| MDP | Markov Decision Process | 状态–动作–转移–奖励形式化 |
| DR | Domain Randomization | 域随机化 |
| FM | Foundation Model | 大模型/基础模型增强迁移 |

## 为什么重要

- LongchaoDa AwesomeSim2Real 收录；分组 Action / Action Delay。 本页是清单索引，细节以原文为准。
- 想横向对照同一分组的其他工作，可以从 [AwesomeSim2Real 技术地图](../overview/lc-awesome-sim2real-technology-map.md) 逐条展开。
- 顺着列表实体 [AwesomeSim2Real](../entities/awesome-sim2real.md) 与站内 Sim2Real 方法/任务页，可以接回对应的学习主线。

## 流程总览

下图是论文明确陈述的论证结构（立场论文）：先论证真实系统与研究设定之间的鸿沟，再列出九项挑战，最后给出示例环境与评价指标。

```mermaid
flowchart TD
    GAP["§1 论点: 研究型 RL 设定与现实系统之间鸿沟过大<br/>常无好的模拟器 / 系统随机且非平稳 / 有强安全约束 / 运行昂贵或缓慢"] --> CH
    CH["§2 九项实际挑战, 每项按: 动机 → 形式化 → 文献中的做法 → 评价指标"] --> NINE["1 从外部行为策略固定日志离线训练<br/>2 在真实系统上用有限样本学习<br/>3 高维连续状态与动作空间<br/>4 不应或极少违反的安全约束<br/>5 部分可观测 / 非平稳 / 随机<br/>6 奖励未指定 / 多目标 / 风险敏感<br/>7 操作员需要可解释的策略与动作<br/>8 推理须在系统控制频率下实时完成<br/>9 执行器 / 传感器 / 奖励存在大或未知的延迟"]
    NINE --> ENV["§3 示例环境: DeepMind control suite 的 humanoid 任务<br/>经修改后同时呈现九项挑战"]
    ENV --> MET["评价指标 Table 2: Average Return / Warm-Start / Time-to-Rmin / Safety Violations / Worst-Case Perf. / Robust Performance / CVaR Return / Multi-Objective / Explainability 用户评估"]
    MET --> REL["§4 相关工作 → §5 结论: 解决全部九项的方法可适用于大量现实问题"]
```

图依据：立场论文论证结构图，论文摘要与 §1 的九项挑战列表、§2.1-2.9、§3 示例环境与 Table 2、§4 相关工作、§5 结论。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 003/139 |
| 分组 | Action / Action Delay |
| 出处 | 见清单 / 原文 |
| 论文 | <https://arxiv.org/abs/1904.12901> |

## 核心机制（归纳）

### 策展导读要点

LongchaoDa AwesomeSim2Real 收录；分组 Action / Action Delay。 本页是清单索引，细节以原文为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/1904.12901) 为准。
- 横向对照请回到 [技术地图](../overview/lc-awesome-sim2real-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/lc-awesome-sim2real-technology-map.md) 的 **Action / Action Delay** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome 列表可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Challenges of real-world reinforcement learning」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Action / Action Delay**；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的分组标签当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[AwesomeSim2Real](../entities/awesome-sim2real.md)
- 技术地图：[AwesomeSim2Real 技术地图](../overview/lc-awesome-sim2real-technology-map.md)
- 方法/任务：[sim2real.md](../concepts/sim2real.md)、[locomotion.md](../tasks/locomotion.md)

## 参考来源

- [`sources/papers/lc_awesome_sim2real_1904_12901_challenges-of-real-world-reinforcement-l.md`](../../sources/papers/lc_awesome_sim2real_1904_12901_challenges-of-real-world-reinforcement-l.md) — 本条目策展摘录
- [`sources/papers/lc_awesome_sim2real_catalog.md`](../../sources/papers/lc_awesome_sim2real_catalog.md) — 列表总表
- [`sources/repos/awesome-sim2real.md`](../../sources/repos/awesome-sim2real.md)
- 论文：<https://arxiv.org/abs/1904.12901>
- [arXiv PDF](https://arxiv.org/pdf/1904.12901)：已通读摘要、§1 引言、§3 示例环境（含 Table 2）、§4 相关工作、§5 结论；§2 各小节读各挑战的定义段与评价指标，文献举例略读；附录（另外三个任务）未读

## 推荐继续阅读

- [AwesomeSim2Real 仓库](https://github.com/LongchaoDa/AwesomeSim2Real)
- [原文](https://arxiv.org/abs/1904.12901)
