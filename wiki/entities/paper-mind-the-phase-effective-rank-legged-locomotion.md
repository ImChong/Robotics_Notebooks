---
type: entity
tags: [paper, quadruped, spot, representation]
status: complete
updated: 2026-09-15
arxiv: "2609.06958"
related:
  - ../tasks/locomotion.md
  - ../methods/ppo.md
  - ./paper-ebert-nonlinear-normal-modes.md
sources:
  - ../../sources/papers/mind_the_phase_effective_rank_arxiv_2609_06958.md
summary: "Mind the Phase（arXiv:2609.06958）：per-phase Jacobian effective rank; global average hides phase differences; network change reduces Spot joint jitter ~3x；截至入库日未见官方代码。"
---

# Mind the Phase（arXiv:2609.06958）

**Mind the Phase**（*Mind the Phase: Effective Rank and Representation Health in Legged Locomotion*，[arXiv:2609.06958](https://arxiv.org/abs/2609.06958)）由 **圣保罗大学（University of São Paulo）** 提出（公众号周更 ingest 见 [策展索引](../../sources/blogs/wechat_shenlan_weekly_humanoid_quadruped_2026-09-14.md)）。

## 一句话定义

关注步态相位：腿式运动策略的有效秩与表征健康度 — per-phase Jacobian effective rank。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ER | Effective Rank | 有效秩 |
| RL | Reinforcement Learning | 强化学习 |
| Jacobian | Policy Jacobian | 策略对观测的雅可比 |

## 为什么重要

腿式控制在不同相位需要不同表征容量；全局指标误导网络设计。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 圣保罗大学（University of São Paulo） |
| **开源** | **未见/待发布**（步骤 2.5 核查：截至 2026-09-14 无可运行官方仓库） |

## 核心原理

按步态相位计算策略 Jacobian 的有效秩；发现全局平均掩盖低秩相位；针对性网络修改提升 Spot 平滑度。

### 流程总览

```mermaid
flowchart LR
  policy[腿式策略] --> phase[步态相位划分]
  phase --> jac[Jacobian ER]
  jac --> diag[表征健康诊断]
  diag --> arch[网络结构调整]
  arch --> spot[Spot 部署]
```

## 源码运行时序图

**不适用** — 截至 **2026-09-14** arXiv 与常见项目页 **未见** 官方可运行代码仓库。

## 工程实践

| 项 | 说明 |
|----|------|
| 开源状态 | 未见官方仓库；以 arXiv 为准 |
| 复现入口 | 论文方法与超参；代码发布后再补 `sources/repos/` |
| 部署注意 | 在线估计相位需可靠接触检测；ER 计算开销可离线分析。 |

## 实验与评测

Spot 关节抖动、跟踪误差；分相位 ER 曲线。

## 结论

分相位有效秩揭示腿式策略表征瓶颈，指导结构修改显著降抖动。

1. 全局 ER 平均会误判健康度。
2. 支撑相与摆动相需求不同。
3. 网络加宽/分支可改善低秩相位。
4. Spot 抖动降约 3x。
5. 表征诊断应纳入腿式调试流程。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| 只看回报曲线 | 不见表征坍缩 |
| 均匀网络宽度 | 未按相位分配容量 |

## 局限与风险

仅 Spot 与特定步态；因果链需更多任务验证。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [ppo](../methods/ppo.md)
- [./paper-ebert-nonlinear-normal-modes.md](./paper-ebert-nonlinear-normal-modes.md)

## 参考来源

- [mind_the_phase_effective_rank_arxiv_2609_06958.md](../../sources/papers/mind_the_phase_effective_rank_arxiv_2609_06958.md)
- [公众号周更策展](../../sources/blogs/wechat_shenlan_weekly_humanoid_quadruped_2026-09-14.md)

## 推荐继续阅读

- [https://arxiv.org/abs/2609.06958](https://arxiv.org/abs/2609.06958)
