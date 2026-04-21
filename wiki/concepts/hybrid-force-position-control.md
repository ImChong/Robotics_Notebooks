---
type: concept
tags: [control, manipulation, force-control, position-control, contact-rich, assembly]
status: complete
updated: 2026-04-20
summary: "力位混合控制把不同方向上的控制目标拆分为位置跟踪与力跟踪两类，是接触丰富操作中最常见的执行层结构之一。"
sources:
  - ../../sources/papers/contact_dynamics.md
  - ../../sources/papers/contact_planning.md
related:
  - ./impedance-control.md
  - ./contact-rich-manipulation.md
  - ./whole-body-control.md
  - ../tasks/manipulation.md
  - ../queries/contact-rich-manipulation-guide.md
---

# Hybrid Force-Position Control（力位混合控制）

**力位混合控制**：把任务空间拆成“该控位置的方向”和“该控力的方向”，让机器人在一个子空间内严格跟踪几何目标，在另一个子空间内稳定施加期望接触力。

## 一句话定义

它不是在同一方向同时强行控位置又控力，而是先决定**哪些自由度听位置，哪些自由度听力**。

## 为什么重要

在复杂的接触任务中，如果仅使用位置控制，即使极小的几何误差（如环境位置估计偏差 1mm）也可能产生巨大的接触力，导致工件损坏或机器人保护性停机。如果仅使用力控制，机器人则会失去对末端位姿的精确约束。

力位混合控制（Hybrid Control）通过 **任务空间的正交分解**，在不同的物理维度上分别满足位置约束和力约束。它是工业装配（插拔、打磨）、协作机器人交互以及人形机器人精细操作的基础。

## 核心原理：任务框架（Task Frame）

力位混合控制的核心在于 **Mason 提出的任务框架（Task Frame）** 概念。在接触发生时，任务空间被划分为两个正交的子空间：

1. **位置子空间（Position Subspace）**：机器人可以自由移动或需要精确跟踪轨迹的方向（通常是切线方向）。
2. **力子空间（Force Subspace）**：机器人受到环境约束，无法自由移动，但需要维持特定压力的方向（通常是法线方向）。

### 数学表达：选择矩阵

通过定义一个对角阵 **选择矩阵 $S$**，我们可以显式地将任务目标分开。$S$ 的对角元素为 1 或 0：
- $S_{ii} = 1$ 表示第 $i$ 个自由度受位置控制。
- $S_{ii} = 0$ 表示第 $i$ 个自由度受力控制。

定义互补矩阵 $\bar{S} = I - S$，则 $\bar{S}$ 选出了所有受力控制的方向。

控制律通常具有以下形式：
$$ \tau = J^T ( S f_{pos} + \bar{S} f_{force} ) $$
其中：
- $f_{pos}$ 是位置控制器计算的虚功力（如 PID 产生的力）。
- $f_{force}$ 是力控制器计算的期望反馈力。

## 典型应用场景

### 1. 工业打磨与擦拭
- **法向方向**：使用力控维持恒定的下压力，确保打磨质量均一，不因表面起伏而忽轻忽重。
- **切向方向**：使用位置控制跟踪预设的打磨路径，确保覆盖整个作业面。

### 2. 精密插拔（Peg-in-Hole）
- **轴向方向**：使用位置控制执行插入动作。
- **径向/转动方向**：使用力控制或柔顺控制，允许末端在遇到阻碍时产生位移，实现自主寻孔和对齐。

### 3. 旋拧螺丝
- **进给方向**：维持恒定的预紧力，防止起子打滑。
- **旋转方向**：控制转速或转矩，直到达到预设的拧紧力矩。

## 与阻抗控制（Impedance Control）的区别

虽然两者都处理接触力，但哲学不同：

| 特性 | 力位混合控制 | 阻抗控制 |
|------|------------|---------|
| **核心思路** | 任务空间正交分解 | 模拟弹簧-阻尼系统 |
| **控制目标** | 显式跟踪力值 $F_d$ | 控制动态关系 $M, B, K$ |
| **环境要求** | 需要明确知道接触面法向 | 对环境几何不敏感，更通用 |
| **稳定性** | 在刚性接触中可能存在切换震荡 | 物理本质上更稳定（Passive） |
| **适用场景** | 高精度力跟踪（如称重、恒压打磨） | 碰撞处理、人机协作、不确定地形步行 |

## 实施中的挑战

1. **选择矩阵的实时更新**：当接触面几何形状复杂或发生变化时，任务框架的方向必须实时重定位（例如擦拭一个球体表面），这对感知系统提出了高要求。
2. **接触切换的瞬态性能**：从自由运动切换到接触运动时，由于系统刚度的突变，容易产生冲击力或受力方向的逻辑冲突。
3. **传感器噪声与延迟**：末端力传感器（六维力/力矩传感器）的噪声和控制环路的延迟会限制力控的带宽，可能导致高频震荡。

## 何时使用

- **选择混合控制**：当你明确知道任务在哪些维度需要力，哪些维度需要位（如平面作业、已知几何体的装配）。
- **选择阻抗控制**：当环境非常不确定，或者任务目标是保持整体柔顺性（如在人群中行走、抓取位置未知的物体）。

## 参考来源

- Raibert, M. H., & Craig, J. J. (1981). *Hybrid position/force control of manipulators*. Journal of Dynamic Systems, Measurement, and Control.
- Khatib, O. (1987). *A unified approach for motion and force control of robot manipulators: The operational space formulation*. IEEE Journal on Robotics and Automation.
- [sources/papers/contact_dynamics.md](../../sources/papers/contact_dynamics.md) — 接触力基础

## 关联页面

- [Impedance Control](./impedance-control.md)
- [Contact-Rich Manipulation](./contact-rich-manipulation.md)
- [Whole-Body Control](./whole-body-control.md)
- [Manipulation](../tasks/manipulation.md)
- [Query：接触丰富操作实践指南](../queries/contact-rich-manipulation-guide.md)
