---
type: entity
project_id: flashrfcl
tags:
  - paper
  - rl
  - curriculum
  - ucsd
openreview: https://openreview.net/forum?id=DBLXUxxslN
status: draft
updated: 2026-11-12
related:
  - ../concepts/curriculum-learning.md
  - ../concepts/reward-design.md
  - ../methods/imitation-learning.md
  - ../methods/ppo.md
  - ../methods/sac.md
  - ./paper-notebook-maniskill3-gpu-parallelized-robotics-simulation.md
sources:
  - ../../sources/papers/flashrfcl_corl2026.md
original:
  - ../../sources/papers/flashrfcl_corl2026.md
summary: "FlashRFCL（CoRL 2026）：反向 + 正向课程结合高吞吐 GPU 仿真，每任务仅 1 条演示求解稀疏奖励高维机器人任务。本页仅依据摘要，全文未读到，草稿。"
---

# FlashRFCL（CoRL 2026）

> **状态说明：** 本页是 draft。论文全文未能获取（arXiv 无、OpenReview 被验证挡住、无项目页与代码），以下只写摘要里有的内容，不含方法细节与数值。

## 一句话定义

**FlashRFCL**（*FlashRFCL: Demonstration-Efficient Sparse Reward RL via Reverse-Forward Curriculum Learning in High-Throughput GPU Simulation*，Choi / Tao / Christensen）把基于演示的反向 + 正向自动课程与可扩展 RL、高吞吐 GPU 仿真结合，目标是每任务仅一条演示就能更快求解稀疏奖励的高维机器人任务（摘要自述）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RFCL | Reverse-Forward Curriculum Learning | 先从演示末端向前扩展起点（反向），再扩大到完整初始分布（正向）的课程，前作见 arXiv:2405.03379 |
| RL | Reinforcement Learning | 强化学习 |
| GPU sim | GPU-parallel simulation | 在 GPU 上并行运行大量环境的高吞吐仿真 |
| CoRL | Conference on Robot Learning | 机器人学习会议 |

## 为什么重要

- 摘要自述动机：稀疏奖励 + 演示可免去奖励塑形，但已有方法常需大量课程工程、长训练或大量演示。
- 若「每任务一条演示、训练更快」成立，可降低奖励设计与演示采集成本。**该成立与否需全文数据验证，本页未验证。**

## 核心原理与流程总览

以下仅是对摘要与前作 RFCL 名称含义的**推测性**概括，非论文细节：

```mermaid
flowchart LR
  D[每任务 1 条演示] --> R[反向课程 从演示后段起步]
  R --> F[正向课程 逐步扩大起始分布]
  F --> P[稀疏奖励下训练策略]
  G[高吞吐 GPU 仿真] --> P
```

具体 RL 算法、课程调度规则、GPU 仿真实现均未读到。

## 工程实践

- 未读到代码或实现细节；未找到公开仓库，故不写 `code:`。
- 前作 RFCL 有项目页 https://reverseforward-cl.github.io/（来自 arXiv 条目注释），可作背景阅读，非本文内容。

## 与清单说法的核对

清单（Glen Berseth CoRL 2026 阅读清单，LLM 据摘要起草）：「Sparse-reward RL from a single demonstration using a reverse-forward curriculum and high-throughput GPU simulation.」

| 清单要点 | 核对结果 |
|----------|----------|
| 稀疏奖励 RL | 与摘要一致 |
| 单条演示 | 与摘要一致（「one demonstration per task」） |
| 反向 + 正向课程 | 与标题 / 摘要一致 |
| 高吞吐 GPU 仿真 | 与摘要一致 |
| 仿真还是真机、任务数、速度提升倍数 | **无法核对**：摘要未提，全文未读到 |

清单摘要本身就来自摘要，故只能证明两者互相一致，不能证明论文正文的结果。

## 局限

- 论文自述局限：未读到。
- 本页局限：只基于摘要；无数值、无任务列表、无对比基线、无仿真到真机信息；「更快」的比较对象与幅度不明。

## 关联页面

- [课程学习](../concepts/curriculum-learning.md)
- [奖励设计](../concepts/reward-design.md)
- [模仿学习](../methods/imitation-learning.md)
- [PPO](../methods/ppo.md)
- [SAC](../methods/sac.md)
- [ManiSkill3：GPU 并行机器人仿真](./paper-notebook-maniskill3-gpu-parallelized-robotics-simulation.md)

## 参考来源

- [来源归档：flashrfcl_corl2026](../../sources/papers/flashrfcl_corl2026.md)
- 论文页（摘要）：<https://hichristensen.com/publication/2026-corl-flashrfcl/>
- 前作 RFCL：<https://arxiv.org/abs/2405.03379>

## 继续阅读

- 待论文 PDF 公开后补读全文，再把状态改为 complete。
