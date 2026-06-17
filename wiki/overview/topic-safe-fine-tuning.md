---
type: overview
tags: [topic, topic-safe-fine-tuning, safe-rl, cbf, lora, deployment]
status: complete
updated: 2026-06-17
summary: "真机安全微调专题汇总：Sim2Real 部署后的在线 RL 适配，低秩残差（SLowRL）、CBF/CLF 安全壳与生成式兜底，避免训练期硬件损伤。"
---

# 真机安全微调（专题汇总）

> **图谱专题视图**：本页是知识图谱「🛡️ 安全微调」专题的统一入口；在 [图谱专题视图](../../docs/graph.html?topic=safe-fine-tuning) 筛选时，本节点为汇总锚点。

## 一句话定义

**真机安全 RL 微调** 在已有 sim2real 策略能跑的基础上，于 **真实机器人上继续在线优化**，并用 **安全集约束、低秩更新或 Recovery 策略** 限制探索导致的摔倒与硬件风险。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Safe RL | Safe Reinforcement Learning | 带安全约束的强化学习 |
| CBF | Control Barrier Function | 安全集不变性屏障 |
| CLF | Control Lyapunov Function | Lyapunov 稳定性约束 |
| LoRA | Low-Rank Adaptation | 低秩参数高效微调 |
| CMDP | Constrained MDP | 带约束的马尔可夫决策过程 |

## 为什么重要

- **最后 10% 性能常在真机上抠**：但标准 RL 探索代价是摔机。
- **V23 专题主线**：SLowRL、Heracles 等给出可复现工程谱系。
- **与 WBC/CBF 交叉**：安全壳可在策略外或策略内实现。

## 本专题覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| 概念 | 真机安全微调总览 | [Safe Real-World RL Fine-Tuning](../concepts/safe-real-world-rl-fine-tuning.md) |
| 对比 | 残差 vs Real2Sim vs 真机 RL | [Sim2Real vs Real2Sim Fine-Tuning](../comparisons/sim2real-vs-real2sim-fine-tuning.md) |
| 安全 | CBF / CLF / Safety Filter | [Control Barrier Function](../concepts/control-barrier-function.md)、[CLF vs CBF](../comparisons/clf-vs-cbf.md) |
| 形式化 | 安全 LoRA 投影 | [Safe LoRA Update Projection](../formalizations/safe-lora-update-projection.md) |
| 实例 | SLowRL | [SLowRL Paper Entity](../entities/paper-slowrl-safe-lora-locomotion-sim2real.md) |

## 与其他专题的关系

- **[Sim2Real](./topic-sim2real.md)**：安全微调是部署链延伸。
- **[WBC](./topic-wbc.md)**：CBF/CLF 常作为执行层安全壳。
- **[IL/RL](./topic-learning.md)**：在线 RL 与离线 IL 的边界。

## 关联页面

- [Safety Filter](../concepts/safety-filter.md)
- [Balance Recovery](../tasks/balance-recovery.md)
- [Query: CLF/CBF in WBC](../queries/clf-cbf-in-wbc.md)

## 参考来源

- 本库归纳自 [Safe Real-World RL Fine-Tuning](../concepts/safe-real-world-rl-fine-tuning.md) 及 CBF/SLowRL 系列页
- 图谱专题定义：[docs/topic-filters.js](../../docs/topic-filters.js)（`safe-fine-tuning` 命中规则）
