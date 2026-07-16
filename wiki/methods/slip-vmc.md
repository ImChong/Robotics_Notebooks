---
type: method
tags: [locomotion, control, model-based, slip, vmc, legged]
status: complete
updated: 2026-07-16
summary: "SLIP（弹簧负载倒立摆）用质心–足端弹簧近似腿足支撑相；VMC（虚拟模型控制）在任务空间施加虚拟弹簧/阻尼，是 LIP 之后、全刚体 WBC 之前的经典简化控制栈。"
related:
  - ../concepts/lip-zmp.md
  - ../concepts/whole-body-control.md
  - ../overview/humanoid-model-based-control-stack.md
  - ../overview/humanoid-motion-control-know-how-technology-map.md
  - ../../roadmap/depth-classical-control.md
sources:
  - ../../sources/raw/feishu_humanoid_motion_control_know_how_full_2026-07-14.md
  - ../../sources/papers/humanoid_motion_control_know_how.md
---

# SLIP + VMC（弹簧负载倒立摆与虚拟模型控制）

**SLIP（Spring-Loaded Inverted Pendulum）** 把支撑腿近似为质心与足端之间的弹簧；**VMC（Virtual Model Control）** 在任务空间定义虚拟弹簧、阻尼或力，再映射到关节力矩。飞书 Know-How 将其列为 LIP/ZMP 之后、全身动力学 WBC 之前的**中间简化层**。

## 一句话定义

用「质心挂在弹簧上」理解腿足支撑相，再用「虚拟弹簧」在笛卡尔空间塑造行为，而不一开始就解全刚体逆动力学。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SLIP | Spring-Loaded Inverted Pendulum | 支撑相质心–足端弹簧倒立摆模型 |
| VMC | Virtual Model Control | 任务空间虚拟阻抗/力元素映射到关节力矩 |
| LIP | Linear Inverted Pendulum | 常高度线性倒立摆，SLIP 的进一步简化 |
| CoM | Center of Mass | 质心，SLIP 状态核心 |
| WBC | Whole-Body Control | 全刚体多任务 QP，SLIP/VMC 之后的精确层 |
| MPC | Model Predictive Control | 可基于 SLIP 生成落脚点或质心参考 |

## 为什么重要

- **比 LIP 多一步物理**：弹簧腿可刻画支撑相长度变化与部分能量交换，常用于四足/双足**跑步**直觉。
- **比全刚体 WBC 轻**：VMC 适合快速原型与教学，工程上常作 WBC 的任务空间目标生成器。
- **Know-How 课程坐标**：RoboParty 文档把 SLIP+VMC 放在 OCP→LIP→**SLIP**→WBC 链上，强调「原理 → 最小代码 → 局限性」。

## 核心原理

**SLIP 状态（示意）：** 质心位置、速度，足端位置；腿长成弹簧 $l$，刚度 $k$，支撑相约束 $f_n \ge 0$。

**VMC：** 在任务空间定义期望力
\[
\mathbf{f}_{\mathrm{vm}} = K_p (\mathbf{x}^* - \mathbf{x}) + K_d (\dot{\mathbf{x}}^* - \dot{\mathbf{x}})
\]
经雅可比转置 $\tau = J^\top \mathbf{f}_{\mathrm{vm}}$ 下发关节（需奇异性与力矩限幅）。

## 主要技术路线

| 路线 | 代表链接 | 说明 |
|------|----------|------|
| 简化模型 | [LIP/ZMP](../concepts/lip-zmp.md) | 常高度倒立摆 |
| 任务空间 | [Whole-Body Control](../concepts/whole-body-control.md) | VMC 力映射到关节 |
| 纵深 | [传统控制纵深](../../roadmap/depth-classical-control.md) | Stage 1 简化模型 |

## 工程实践（最小实现）

1. 仿真中单腿 SLIP：给定初始水平速度，调节弹簧刚度使周期步态稳定。
2. 在腿式机器人上为质心高度/姿态加 VMC，观察与真实电机 PD 的差异。
3. 与 [LIP/ZMP](../concepts/lip-zmp.md) 对比：何时常高度假设够用、何时必须 SLIP。

## 局限与风险

- **历史局限（LIP 时代遗留对比）：** SLIP+VMC 相对 LIP 已能动态稳定，但仍为启发式合外力/落脚点，非全刚体 OCP。
- **不适用复杂接触**：多足同时接触、大幅躯干转动时假设弱。
- **VMC 非约束感知**：摩擦锥、力矩限需外层 WBC/QP；接触检测可用力传感或关节力矩虚功估计。
- **点足时代产物**：首次实现人形**动态**控制，是点足系统稳定算法来源（Raibert Leg Lab → 波士顿动力谱系）。

## 关联页面

- [LIP / ZMP](../concepts/lip-zmp.md) — 更简化的平衡模型
- [Whole-Body Control](../concepts/whole-body-control.md) — 全刚体精确层
- [Model-based 控制栈](../overview/humanoid-model-based-control-stack.md)
- [Know-How 技术地图](../overview/humanoid-motion-control-know-how-technology-map.md)

## 参考来源

- [humanoid_motion_control_know_how.md](../../sources/papers/humanoid_motion_control_know_how.md) — 飞书 Know-How 方法链

## 推荐继续阅读

- Raibert, *Legged Robots That Balance*（经典腿足平衡）
- [ETH Robot Dynamics Lecture Notes](https://rsl.ethz.ch/education-students/lecture-notes.html) — 传统路线推荐教材
