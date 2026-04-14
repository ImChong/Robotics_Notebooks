# mpc

> 来源归档（ingest）

- **标题：** Model Predictive Control（MPC）— 理论与机器人应用
- **类型：** paper / survey
- **来源：** IEEE TAC / IJRR / conference
- **入库日期：** 2026-04-14
- **最后更新：** 2026-04-14
- **一句话说明：** 覆盖 MPC 经典理论、足式机器人应用和人形实现，支撑 MPC、MPC-WBC 集成等 wiki 页面。

## 核心论文摘录（MVP）

### 1) Constrained Model Predictive Control: Stability and Optimality (Mayne et al., 2000)
- **链接：** <https://doi.org/10.1016/S0005-1098(99)00214-9>
- **核心贡献：** MPC 稳定性理论的里程碑综述，证明了终端约束条件下 MPC 的渐近稳定性，奠定了 MPC 学术体系。
- **关键结论：** 终端集 + 终端代价 → 递归可行 + 渐近稳定
- **对 wiki 的映射：**
  - [Model Predictive Control (MPC)](../../wiki/methods/model-predictive-control.md)
  - [Optimal Control (OCP)](../../wiki/concepts/optimal-control.md)

### 2) Dynamic and Robust Legged Locomotion Using a Simplified Model (Di Carlo et al., 2018)
- **链接：** <https://ieeexplore.ieee.org/document/8594448>
- **核心贡献：** MIT Mini Cheetah / Cheetah 3 的 Convex MPC 实现，用简化质心模型（SRBD）在 250Hz 下实时求解落脚力，开启足式机器人 MPC 工程化。
- **对 wiki 的映射：**
  - [Model Predictive Control (MPC)](../../wiki/methods/model-predictive-control.md)
  - [MPC 与 WBC 集成](../../wiki/concepts/mpc-wbc-integration.md)
  - [Centroidal Dynamics](../../wiki/concepts/centroidal-dynamics.md)

### 3) A Unified MPC Framework for Whole-Body Dynamic Locomotion and Manipulation (Sleiman et al., 2021)
- **链接：** <https://arxiv.org/abs/2103.00946>
- **核心贡献：** 将 MPC 扩展到全身控制 + 操作任务，给出统一的多接触 MPC 框架（ANYmal + 机械臂）。
- **对 wiki 的映射：**
  - [MPC 与 WBC 集成](../../wiki/concepts/mpc-wbc-integration.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)

### 4) Nonlinear MPC for Humanoid Gait Generation (Wieber, 2006)
- **链接：** <https://www.inrialpes.fr/bipop/people/wieber/publis/Wieber06.pdf>
- **核心贡献：** 提出预览控制框架解决 humanoid 行走的 ZMP 约束，建立了 MPC 在双足机器人上的经典实现路线。
- **对 wiki 的映射：**
  - [Model Predictive Control (MPC)](../../wiki/methods/model-predictive-control.md)
  - [LIP / ZMP](../../wiki/concepts/lip-zmp.md)
  - [Capture Point / DCM](../../wiki/concepts/capture-point-dcm.md)

### 5) MPPI: Model Predictive Path Integral Control (Williams et al., 2017)
- **链接：** <https://ieeexplore.ieee.org/document/7989202>
- **核心贡献：** 基于信息论的采样 MPC，无需梯度计算，适合非凸代价函数和高维动作空间，在越野驾驶和 locomotion 中有应用。
- **对 wiki 的映射：**
  - [Model Predictive Control (MPC)](../../wiki/methods/model-predictive-control.md)
  - [Model-Based RL](../../wiki/methods/model-based-rl.md)

## 当前提炼状态

- [x] Mayne 稳定性理论 / Di Carlo Convex MPC / Sleiman 全身 MPC / Wieber humanoid / MPPI 五条摘要
- [ ] 后续补：MPC 在人形机器人上的实时性分析（求解器选型 OSQP vs qpOASES）
