---
type: overview
tags: [topic, topic-wbc, whole-body-control, humanoid, balance, tsid]
status: complete
updated: 2026-06-17
summary: "全身控制（WBC）专题汇总：质心/接触约束下的层级 QP、TSID/HQP 与 CBF 安全过滤，衔接 MPC 与 RL 策略的执行层。"
---

# 全身控制 WBC（专题汇总）

> **图谱专题视图**：本页是知识图谱「🦾 全身控制 (WBC)」专题的统一入口；在 [图谱专题视图](../../docs/graph.html?topic=wbc) 筛选时，本节点为汇总锚点。

## 一句话定义

**Whole-Body Control（WBC）** 在浮基人形/移动操作机器人上，同时满足 **平衡、接触、关节限位与任务优先级**，把高层目标转成可执行的全身关节/力矩指令。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 全身约束下的任务/力分配 |
| TSID | Task-Space Inverse Dynamics | 任务空间逆动力学 WBC 框架 |
| HQP | Hierarchical Quadratic Programming | 层级 QP 优先级控制 |
| CBF | Control Barrier Function | 安全集不变性约束 |
| CLF | Control Lyapunov Function | 稳定性/收敛性 Lyapunov 约束 |

## 为什么重要

- **人形是欠驱动浮基系统**：单任务 IK 不够，必须协调脚、手、躯干。
- **RL 策略的「执行壳」**：很多学习策略输出参考，由 WBC 保证可行与安全。
- **loco-manip 交汇点**：行走与操作共享同一动力学与接触约束。

## 本专题覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| 概念 | WBC 解决什么 | [Whole-Body Control](../concepts/whole-body-control.md) |
| 方法 | TSID / HQP | [TSID](../concepts/tsid.md)、[HQP](../concepts/hqp.md) |
| 安全 | CBF / CLF / 安全过滤 | [Control Barrier Function](../concepts/control-barrier-function.md)、[Safety Filter](../concepts/safety-filter.md) |
| 集成 | MPC 与 WBC | [MPC-WBC Integration](../concepts/mpc-wbc-integration.md) |
| 对比 | WBC vs 端到端 RL | [WBC vs RL](../comparisons/wbc-vs-rl.md) |

## 与其他专题的关系

- **[Locomotion](./topic-locomotion.md)**：步态与平衡是 WBC 典型应用。
- **[WBT](./topic-wbt.md)**：跟踪策略常与 WBC/PD 层叠。
- **[安全微调](./topic-safe-fine-tuning.md)**：CBF/CLF 常作为真机 RL 安全壳。

## 关联页面

- [Whole-Body Coordination](../concepts/whole-body-coordination.md)
- [Floating-Base Dynamics](../concepts/floating-base-dynamics.md)
- [Centroidal Dynamics](../concepts/centroidal-dynamics.md)

## 参考来源

- 本库归纳自 [Whole-Body Control](../concepts/whole-body-control.md)、[TSID](../concepts/tsid.md)、[HQP](../concepts/hqp.md)
- 图谱专题定义：[docs/topic-filters.js](../../docs/topic-filters.js)（`wbc` 命中规则）
