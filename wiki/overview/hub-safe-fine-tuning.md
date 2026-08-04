---
type: overview
tags: [hub, hub-safe-fine-tuning, safe-rl, cbf, lora, deployment]
status: complete
updated: 2026-08-04
summary: "真机安全微调知识链汇总：Sim2Real 部署后的在线 RL 适配，低秩残差（SLowRL）、CBF/CLF 安全壳与生成式兜底，避免训练期硬件损伤。"
---

# 真机安全微调（知识链汇总）

> **知识链汇总**：本页是相关概念/方法的统一入口；对应策展纵深见图谱 [路线视图](../../docs/graph.html?depth=safe-control) 与 [路线页](../../roadmap/depth-safe-control.md)。

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
- **V23 知识链主线**：SLowRL、Heracles 等给出可复现工程谱系。
- **与 WBC/CBF 交叉**：安全壳可在策略外或策略内实现。

## 本知识链覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| 概念 | 真机安全微调总览 | [Safe Real-World RL Fine-Tuning](../concepts/safe-real-world-rl-fine-tuning.md) |
| 对比 | 残差 vs Real2Sim vs 真机 RL | [Sim2Real vs Real2Sim Fine-Tuning](../comparisons/sim2real-vs-real2sim-fine-tuning.md) |
| 安全 | CBF / CLF / Safety Filter | [Control Barrier Function](../concepts/control-barrier-function.md)、[CLF vs CBF](../comparisons/clf-vs-cbf.md) |
| 形式化 | 安全 LoRA 投影 | [Safe LoRA Update Projection](../formalizations/safe-lora-update-projection.md) |
| 实例 | SLowRL | [SLowRL Paper Entity](../entities/paper-slowrl-safe-lora-locomotion-sim2real.md) |

## 与其他知识链的关系

- **[Sim2Real](./hub-sim2real.md)**：安全微调是部署链延伸。
- **[WBC](./hub-wbc.md)**：CBF/CLF 常作为执行层安全壳。
- **[IL/RL](./hub-learning.md)**：在线 RL 与离线 IL 的边界。

## 关联页面

- [Safety Filter](../concepts/safety-filter.md)
- [Balance Recovery](../tasks/balance-recovery.md)
- [Query: CLF/CBF in WBC](../queries/clf-cbf-in-wbc.md)
- [CLIFT](../entities/paper-clift-closed-loop-iterative-finetuning.md) — 闭权重模型只给托管 SFT API 时的真机闭环微调路线

## 参考来源

- 本库归纳自 [Safe Real-World RL Fine-Tuning](../concepts/safe-real-world-rl-fine-tuning.md) 及 CBF/SLowRL 系列页
- 知识链定义：[docs/depth-filters.js](../../docs/depth-filters.js)（`safe-fine-tuning` 命中规则）
- 上游原始资料（本链概念页共同的 ingest 来源）：[SLowRL：运动控制的安全低秩自适应 RL（arXiv:2603.17092）](../../sources/papers/slowrl_arxiv_2603_17092.md)、[PAC-MAN：感知感知 CBF-RL 全身安全（arXiv:2607.28623）](../../sources/papers/pac_man_perceptive_cbf_rl_arxiv_2607_28623.md)、[最优控制理论](../../sources/papers/optimal_control_theory.md)
