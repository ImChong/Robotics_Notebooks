# Whole-Body Control

聚焦任务空间控制、TSID、QP-WBC、人形全身运动控制相关论文。

## 关注问题

- 如何在多任务（平衡、跟踪、避障）之间设置优先级？
- 如何在保持动力学一致性的前提下实时求解关节力矩？
- 如何处理多接触状态下的约束（非穿地、摩擦锥）？
- 如何在高性能硬件上实现 kHz 级别的实时闭环？

## 代表性论文

### 经典理论线（Sentis–Khatib）

- **Khatib (1987)** — *Operational Space Formulation* — 操作空间运动/力统一；见 [OSF 实体](../../wiki/entities/paper-operational-space-formulation.md)。
- **Khatib et al. (2004)** — *Whole body dynamic behavior and control of human-like robots* (IJHR) — WBC 系统化起点；见 [IJHR 2004 实体](../../wiki/entities/paper-khatib-sentis-ijhr-2004-whole-body-dynamic-behavior.md)。
- **Sentis & Khatib (2006)** — *A Whole-Body Control Framework for Humanoids Operating in Human Environments* (ICRA) — 代表性人形环境框架；见 [ICRA 2006 实体](../../wiki/entities/paper-sentis-khatib-icra-2006-whole-body-control-framework.md)。
- **Fok et al. (2015)** — *ControlIt!* (arXiv:1506.01075) — WBOSC 开源软件；见 [ControlIt!](../../wiki/entities/controlit.md)。

### 核心方法论

- **Task Space Inverse Dynamics (TSID)** (Del Prete et al.) — 提出在接触约束下的 prioritized motion-force 控制框架，统一求解加速度与接触力。
- **Hierarchical Quadratic Programming (HQP)** (Escande et al., 2014) — 系统化了 HQP 的实时求解框架，奠定了人形机器人运动生成的工程基础。
- **Sentis & Khatib (2005)** — *Synthesis of Whole-Body Behaviors Through Hierarchical Control*. 奠定了全身行为层级控制范式。

### 工程框架与工具

- **Mastalli et al. (2020)** — *Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control*. 为 WBC 与轨迹优化提供高效求解器。
- **legbot-MPC-WBC** ([Robot-Nav](https://github.com/Robot-Nav/legbot-MPC-WBC)) — 四足 Convex MPC + WBC 分支 sim2sim/sim2real 参考；见 [legbot-mpc-wbc 实体](../../wiki/entities/legbot-mpc-wbc.md)。
- **legbot_lab** ([Robot-Nav](https://github.com/Robot-Nav/legbot_lab)) — 四足 Isaac Lab PPO / MoE-CTS RL + ONNX 部署；见 [Legbot Lab 实体](../../wiki/entities/legbot-lab.md) 与 [CTS 论文页](../../wiki/entities/paper-cts-concurrent-teacher-student-locomotion.md)。

### 稳定性与平衡分析

- **Koolen et al. (2012)** — *Capturability-based Analysis and Control of Legged Locomotion*. 提出了 N-step Capturable 概念，为平衡恢复提供可行性判定。

## 关联页面

- [Whole-Body Control (WBC) (Concept)](../../wiki/concepts/whole-body-control.md)
- [TSID (Concept)](../../wiki/concepts/tsid.md)
- [HQP (Concept)](../../wiki/concepts/hqp.md)
- [Capture Point / DCM (Concept)](../../wiki/concepts/capture-point-dcm.md)
- [Crocoddyl (Entity)](../../wiki/entities/crocoddyl.md)
- [MPC 与 WBC 集成 (Concept)](../../wiki/concepts/mpc-wbc-integration.md)
