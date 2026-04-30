---
type: entity
tags: [textbook, kinematics, dynamics, control, lie-group, screw-theory, foundational]
status: complete
updated: 2026-04-30
related:
  - ../formalizations/se3-representation.md
  - ../concepts/floating-base-dynamics.md
  - ../concepts/whole-body-control.md
  - ../methods/trajectory-optimization.md
  - ./pinocchio.md
sources:
  - ../../sources/papers/modern_robotics_textbook.md
summary: "Lynch & Park 的现代机器人学经典教材，独特之处是全程使用李群 / 螺旋理论作为统一数学语言，覆盖配置空间到全身控制、抓取、移动机器人的完整体系，是本知识库传统机器人学部分的主要参考底座。"
---

# Modern Robotics (Lynch-Park 教材)

**Modern Robotics: Mechanics, Planning, and Control** 是 Kevin M. Lynch（Northwestern）与 Frank C. Park（SNU）2017 年由 Cambridge University Press 出版的本科级机器人学教材。它在传统教材（Craig、Spong、Siciliano）之外提供了**用李群 / 螺旋理论统一描述刚体运动、运动学与动力学**的视角，配套 Coursera 6 门专项课程与开源库，是本知识库传统机器人学部分的主要参考底座。

## 为什么重要？

1. **语言上的统一**：传统教材用 D-H 参数 + 旋转矩阵 + 欧拉角，每章独立；Lynch-Park 全程用 SE(3)、twist、PoE，链条干净。
2. **本科可读但仍是研究语言**：用的就是 Pinocchio / Crocoddyl / TSID 等现代库的底层数学，学完直接接得上工业代码。
3. **覆盖广度对齐"传统机器人栈"**：13 章覆盖了从最底层（C-space）到最上层（grasping、wheeled mobile）的完整传统机器人学，是 RL/LLM 时代之前「人形/四足/机械臂控制」的最大公约数。

## 章节地图（与本知识库的对应）

| 章节 | 教材主题 | 对应已有页面 |
|------|---------|------------|
| Ch 2 | Configuration Space | （未直接覆盖，可补） |
| Ch 3 | Rigid-Body Motions | [SE(3) Representation](../formalizations/se3-representation.md) |
| Ch 4 | Forward Kinematics (PoE) | （部分隐含在 [pinocchio](./pinocchio.md)） |
| Ch 5 | Velocity Kinematics & Statics | （隐含在 [whole-body-control](../concepts/whole-body-control.md) 的 Jacobian 部分） |
| Ch 6 | Inverse Kinematics | （未直接覆盖） |
| Ch 7 | Closed Chains | （未直接覆盖） |
| Ch 8 | Dynamics of Open Chains | [Floating Base Dynamics](../concepts/floating-base-dynamics.md) |
| Ch 9 | Trajectory Generation | [Trajectory Optimization](../methods/trajectory-optimization.md) |
| Ch 10 | Motion Planning | （RRT/PRM 未直接覆盖） |
| Ch 11 | Robot Control | [WBC](../concepts/whole-body-control.md), [TSID](../concepts/tsid.md), [Impedance Control](../concepts/impedance-control.md) |
| Ch 12 | Grasping & Manipulation | [Friction Cone](../formalizations/friction-cone.md), [Contact Wrench Cone](../formalizations/contact-wrench-cone.md) |
| Ch 13 | Wheeled Mobile Robots | （未直接覆盖） |

## 核心数学语言：李群 / 螺旋理论

Lynch-Park 将下面这些贯穿全书：

- **SO(3) / SE(3)**：刚体姿态与位姿的群结构
- **so(3) / se(3)**：对应李代数（角速度、空间速度向量空间）
- **Twist $\mathcal{V} \in \mathbb{R}^6$**：空间速度（角速度 + 线速度）
- **Wrench $\mathcal{F} \in \mathbb{R}^6$**：空间力（力矩 + 力）
- **PoE 公式**：正运动学写成 $T(\theta) = e^{[\mathcal{S}_1]\theta_1} e^{[\mathcal{S}_2]\theta_2} \cdots e^{[\mathcal{S}_n]\theta_n} M$
- **空间 vs 物体雅可比**：两种坐标系下的 Jacobian 表示

这套语言是 Pinocchio、Crocoddyl、TSID、Drake 这些现代机器人库的内部实现语言，不是新东西，但教材级清晰阐述较少。

## 局限

- **不覆盖 RL / IL**：本书是 2017 年传统机器人学视角，不涉及深度强化学习、模仿学习、VLA
- **不覆盖 sim2real / 真机部署工程**：纯理论 + 仿真层
- **接触动力学浅尝辄止**：Ch 12 抓取讲了静态接触，但完整的接触动力学（complementarity、impulse-based）需要 Featherstone 等更专的资料

## 推荐使用方式

| 你想做什么 | 建议读哪些章节 |
|-----------|--------------|
| 入门人形/四足控制的数学语言 | Ch 3 → Ch 5 → Ch 8 → Ch 11 |
| 实现 IK 求解器 | Ch 6（数值法部分）+ 配套 Python 库 |
| 理解 Pinocchio / TSID 的内部实现 | Ch 3, 4, 5, 8（PoE + Spatial Vectors） |
| 写传统采样规划（RRT/PRM） | Ch 10 |
| 学抓取力学 | Ch 12 + [friction-cone.md](../formalizations/friction-cone.md) |

## 关联页面

- [SE(3) Representation](../formalizations/se3-representation.md) — 教材 Ch 3 对应的形式化
- [Floating Base Dynamics](../concepts/floating-base-dynamics.md) — 教材 Ch 8 在浮基系统上的延伸
- [Whole-Body Control](../concepts/whole-body-control.md) — 教材 Ch 11 控制章节的现代延伸
- [Pinocchio](./pinocchio.md) — 直接使用本教材数学语言的现代机器人库
- [Trajectory Optimization](../methods/trajectory-optimization.md) — 教材 Ch 9 的现代化版本

## 参考来源

- [sources/papers/modern_robotics_textbook.md](../../sources/papers/modern_robotics_textbook.md)
- Lynch, K. M., & Park, F. C. (2017). *Modern Robotics: Mechanics, Planning, and Control*. Cambridge University Press.
- [Northwestern Mechatronics Wiki — Modern Robotics](https://hades.mech.northwestern.edu/index.php/Modern_Robotics)
- [PDF (官方免费)](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf)
