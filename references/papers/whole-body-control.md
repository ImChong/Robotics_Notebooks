# Whole-Body Control

聚焦任务空间控制、TSID、QP-WBC、人形全身运动控制相关论文。

## 关注问题

- 如何在多任务（平衡、跟踪、避障）之间设置优先级？
- 如何在保持动力学一致性的前提下实时求解关节力矩？
- 如何处理多接触状态下的约束（非穿地、摩擦锥）？
- 如何在高性能硬件上实现 kHz 级别的实时闭环？

## 代表性论文

### 核心方法论

- **Task Space Inverse Dynamics (TSID)** (Del Prete et al.) — 提出在接触约束下的 prioritized motion-force 控制框架，统一求解加速度与接触力。
- **Hierarchical Quadratic Programming (HQP)** (Escande et al., 2014) — 系统化了 HQP 的实时求解框架，奠定了人形机器人运动生成的工程基础。
- **Sentis & Khatib (2005)** — *Synthesis of Whole-Body Behaviors Through Hierarchical Control*. 奠定了全身行为层级控制范式。

### 工程框架与工具

- **Mastalli et al. (2020)** — *Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control*. 为 WBC 与轨迹优化提供高效求解器。

### 稳定性与平衡分析

- **Koolen et al. (2012)** — *Capturability-based Analysis and Control of Legged Locomotion*. 提出了 N-step Capturable 概念，为平衡恢复提供可行性判定。

## 关联页面

- [Whole-Body Control (WBC) (Concept)](../../wiki/concepts/whole-body-control.md)
- [TSID (Concept)](../../wiki/concepts/tsid.md)
- [HQP (Concept)](../../wiki/concepts/hqp.md)
- [Capture Point / DCM (Concept)](../../wiki/concepts/capture-point-dcm.md)
- [Crocoddyl (Entity)](../../wiki/entities/crocoddyl.md)
- [MPC 与 WBC 集成 (Concept)](../../wiki/concepts/mpc-wbc-integration.md)
