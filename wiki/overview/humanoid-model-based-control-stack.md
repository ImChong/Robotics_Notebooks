---
type: overview
tags: [humanoid, model-based, mpc, wbc, control-stack]
status: complete
updated: 2026-07-14
related:
  - ./humanoid-motion-control-know-how-technology-map.md
  - ../../roadmap/depth-classical-control.md
  - ../concepts/optimal-control.md
  - ../methods/slip-vmc.md
  - ../concepts/srbd-convex-mpc-wbc.md
  - ../methods/centroidal-nmpc-wbc-stack.md
sources:
  - ../../sources/papers/humanoid_motion_control_know_how.md
summary: "飞书「传统运动控制方法（Model Base）」父节点：OCP→LIP/ZMP→SLIP/VMC→WBC/TSID→SRBD+凸MPC+WBC→CD+NMPC+WBC→状态估计七段主链索引。"
---

# 传统运动控制方法栈（Model-based）

飞书 Know-How **「传统运动控制方法（Model Base）」** 的图谱父节点：按保真度与实时性递进组织 **七段主链**，每段在 wiki 中有独立方法/概念页，并统一遵循 **原理 → 基本代码 → 局限性** 学习模板。

## 一句话定义

从最优控制理论出发，经简化模型与 MPC，到 WBC 执行与状态估计闭环的人形 Model-based 主干。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OCP | Optimal Control Problem | 全链理论起点 |
| LIP | Linear Inverted Pendulum | 步行简化模型 |
| ZMP | Zero Moment Point | 平衡判据 |
| SLIP | Spring-Loaded Inverted Pendulum | 弹簧腿支撑模型 |
| VMC | Virtual Model Control | 虚拟阻抗控制 |
| WBC | Whole-Body Control | 全刚体 QP 执行 |
| TSID | Task-Space Inverse Dynamics | WBC 常用实现 |
| MPC | Model Predictive Control | 滚动规划 |
| SRBD | Single Rigid Body Dynamics | 质心单刚体近似 |
| NMPC | Nonlinear MPC | 高保真质心规划 |

## 七段主链与 wiki 节点

| 序 | 飞书主题 | Wiki |
|----|----------|------|
| 1 | OCP | [Optimal Control](../concepts/optimal-control.md) |
| 2 | LIP+ZMP | [LIP/ZMP](../concepts/lip-zmp.md) |
| 3 | SLIP+VMC | [SLIP+VMC](../methods/slip-vmc.md) |
| 4 | WBD+WBC/TSID | [WBC](../concepts/whole-body-control.md)、[TSID](../concepts/tsid.md) |
| 5 | SRBD+凸MPC+WBC | [SRBD+凸MPC+WBC](../concepts/srbd-convex-mpc-wbc.md) |
| 6 | CD+NMPC+WBC | [Centroidal NMPC+WBC](../methods/centroidal-nmpc-wbc-stack.md) |
| 7 | 状态估计 | [State Estimation](../concepts/state-estimation.md) |

## 学习路线

- 纵深展开：[depth-classical-control](../../roadmap/depth-classical-control.md)
- 工具链： [Pinocchio](../entities/pinocchio.md)、MuJoCo、[建模与求解](../concepts/modeling-and-solving-for-control.md)

## 局限与风险

- **不是唯一正解**：平坦慢走可停在 LIP/SRBD；高动态需 CD-NMPC。
- **开源少于 RL**：飞书指出传统栈理论门槛高，需啃教材与四足参考实现。

## 关联页面

- [Know-How 技术地图](./humanoid-motion-control-know-how-technology-map.md)
- [MPC 与 WBC 集成](../concepts/mpc-wbc-integration.md)
- [WBC vs RL](../comparisons/wbc-vs-rl.md)

## 参考来源

- [humanoid_motion_control_know_how.md](../../sources/papers/humanoid_motion_control_know_how.md)

## 推荐继续阅读

- [depth-classical-control](../../roadmap/depth-classical-control.md)
