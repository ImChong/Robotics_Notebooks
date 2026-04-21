---
type: query
tags: [mpc, control, optimization, tuning, locomotion]
status: complete
updated: 2026-04-21
related:
  - ../methods/model-predictive-control.md
  - ../methods/trajectory-optimization.md
  - ../concepts/optimal-control.md
sources:
  - ../../sources/papers/mpc.md
summary: "MPC 调参指南：系统梳理了预测时域、代价函数权重 Q/R 矩阵、以及状态偏差惩罚对机器人稳定性的影响，并提供了实用的工程调试顺序。"
---

# Model Predictive Control (MPC) 调参指南

> **Query 产物**：本页由以下问题触发：「我的线性 MPC 跟踪很不稳，Base 晃动严重，应该先调哪个参数？Q 和 R 怎么给值？」
> 综合来源：[MPC 核心页](../methods/model-predictive-control.md)、[Optimal Control](../concepts/optimal-control.md)

---

调节 MPC 控制器本质上是在**跟踪性能**、**控制平滑度**与**计算开销**之间寻找平衡。

## 1. 权重矩阵的设计 (Q & R)

MPC 的目标是最小化 $J = \sum (x^T Q x + u^T R u)$。

- **状态权重 Q (State Penalty)**：
  - **Base Orientation (姿态)**：权重应设为最高。足式机器人姿态一旦偏离，重力补偿会失效。
  - **Velocity (速度)**：中等权重。
  - **Position (位置)**：通常设为较低权重，因为姿态和速度更决定瞬时稳定性。
- **动作权重 R (Control Penalty)**：
  - 代表你对电机“出力”的厌恶程度。
  - R 过小：动作剧烈，容易引起高频震荡和硬件损耗。
  - R 过大：动作迟缓，像是在“泥潭”里行走。

## 2. 预测时域 (Horizon) 与步长 (dt)

- **时域 (T)**：预测未来多久的运动。
  - 足式机器人通常预测 **1-2 个步态周期** (约 0.4s - 0.8s)。
  - 时域越长，全局观越好（能预见前方障碍），但计算开销呈三次方增长。
- **步长 (dt)**：通常建议与控制周期一致（10ms - 30ms）。

## 3. 推荐调试顺序

1. **先调静态平衡**：将期望速度设为 0，调高姿态权重 Q，直到机器人能稳稳对抗推力。
2. **增加控制平滑度**：逐渐增大 R，直到足端接地力不再发生剧烈跳变。
3. **开启动态跟踪**：引入速度指令，调优速度跟踪项的权重。

## 关联页面
- [Model Predictive Control (MPC)](../methods/model-predictive-control.md)
- [Trajectory Optimization (轨迹优化)](../methods/trajectory-optimization.md)

## 参考来源
- [sources/papers/mpc.md](../../sources/papers/mpc.md)
- Di Carlo, J., et al. (2018). *Dynamic Locomotion on the MIT Cheetah 3*.
