---

type: entity
tags: [textbook, kinematics, dynamics, control, lie-group, screw-theory, foundational, northwestern]
status: complete
updated: 2026-11-07
related:
  - ../overview/modern-robotics-wechat-principles-series.md
  - ./python-robotics.md
  - ./learn-robotics-qqfly-guide.md
  - ../formalizations/lie-group-rigid-body-motions.md
  - ../formalizations/se3-representation.md
  - ../formalizations/forward-kinematics.md
  - ../formalizations/inverse-kinematics.md
  - ../methods/newtons-method.md
  - ../formalizations/robot-jacobian.md
  - ../concepts/floating-base-dynamics.md
  - ../concepts/whole-body-control.md
  - ../methods/trajectory-optimization.md
  - ./pinocchio.md
  - ./linear-algebra-curriculum.md
sources:
  - ../../sources/papers/modern_robotics_textbook.md
  - ../../sources/raw/wechat_modern_robotics_album_4521219024549937157.md
original:
  - ../../sources/papers/modern_robotics_textbook.md
summary: "Lynch & Park 的现代机器人学经典教材，独特之处是全程使用李群 / 螺旋理论作为统一数学语言，覆盖配置空间到全身控制、抓取、移动机器人的完整体系，是本知识库传统机器人学部分的主要参考底座。"
---

# Modern Robotics (Lynch-Park 教材)

## 一句话定义

**Modern Robotics: Mechanics, Planning, and Control** 是 Kevin M. Lynch（Northwestern）与 Frank C. Park（SNU）2017 年由 Cambridge University Press 出版的本科级机器人学教材。它在传统教材（Craig、Spong、Siciliano）之外提供了**用李群 / 螺旋理论统一描述刚体运动、运动学与动力学**的视角，配套 Coursera 6 门专项课程与开源库，是本知识库传统机器人学部分的主要参考底座。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TSID | Task-Space Inverse Dynamics | 任务空间逆动力学求解关节力矩的 WBC 实现 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| LLM | Large Language Model | 大语言模型，常作高层任务/语言接口 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Manipulation | Robot Manipulation | 抓取、移动、操作物体的任务总称 |
| IL | Imitation Learning | 从专家演示学习策略，奖励难定义时的主路线 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| IK | Inverse Kinematics | 满足末端/姿态约束求解关节角的运动学逆解 |
| URDF | Unified Robot Description Format | 统一机器人描述格式 |
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富的刚体物理仿真引擎 |

## 为什么重要？

1. **语言上的统一**：传统教材用 D-H 参数 + 旋转矩阵 + 欧拉角，每章独立；Lynch-Park 全程用 SE(3)、twist、PoE，链条干净。
2. **本科可读但仍是研究语言**：用的就是 Pinocchio / Crocoddyl / TSID 等现代库的底层数学，学完直接接得上工业代码。
3. **覆盖广度对齐"传统机器人栈"**：13 章覆盖了从最底层（C-space）到最上层（grasping、wheeled mobile）的完整传统机器人学，是 RL/LLM 时代之前「人形/四足/机械臂控制」的最大公约数。

## 章节地图（与本知识库的对应）

| 章节 | 教材主题 | 对应已有页面 |
|------|---------|------------|
| Ch 2 | Configuration Space | （未直接覆盖，可补） |
| Ch 3 | Rigid-Body Motions | [李群与刚体运动](../formalizations/lie-group-rigid-body-motions.md)、[SE(3) Representation](../formalizations/se3-representation.md) |
| Ch 4 | Forward Kinematics (PoE) + URDF 一节 + 附录 C（DH） | [正向运动学](../formalizations/forward-kinematics.md)；部分隐含在 [pinocchio](./pinocchio.md) |
| Ch 5 | Velocity Kinematics & Statics（空间 / 体雅可比、奇异、可操作度） | [机器人雅可比](../formalizations/robot-jacobian.md)；隐含在 [whole-body-control](../concepts/whole-body-control.md) 的 Jacobian 部分 |
| Ch 6 | Inverse Kinematics | [Newton–Raphson / 数值 IK](../methods/newtons-method.md)、[逆运动学形式化](../formalizations/inverse-kinematics.md) |
| Ch 7 | Closed Chains | （未直接覆盖） |
| Ch 8 | Dynamics of Open Chains（**固定基座**开链） | [Floating Base Dynamics](../concepts/floating-base-dynamics.md)（浮动基延伸，教材本身不含） |
| Ch 9 | Trajectory Generation | [Trajectory Optimization](../methods/trajectory-optimization.md) |
| Ch 10 | Motion Planning | （RRT/PRM 未直接覆盖） |
| Ch 11 | Robot Control | [WBC](../concepts/whole-body-control.md), [TSID](../concepts/tsid.md), [Impedance Control](../concepts/impedance-control.md) |
| Ch 12 | Grasping & Manipulation | [Friction Cone](../formalizations/friction-cone.md), [Contact Wrench Cone](../formalizations/contact-wrench-cone.md) |
| Ch 13 | Wheeled Mobile Robots | （未直接覆盖） |

## 核心原理：李群 / 螺旋理论（核心数学语言）

Lynch-Park 将下面这些贯穿全书：

- **SO(3) / SE(3)**：刚体姿态与位姿的群结构
- **so(3) / se(3)**：对应李代数（角速度、空间速度向量空间）
- **Twist $\mathcal{V}=(\omega,v) \in \mathbb{R}^6$**：空间速度（角速度在前、线速度在后）
- **Wrench $\mathcal{F}=(m,f) \in \mathbb{R}^6$**：空间力（力矩在前、力在后）
- **PoE 公式**：正运动学写成 $T(\theta) = e^{[\mathcal{S}_1]\theta_1} e^{[\mathcal{S}_2]\theta_2} \cdots e^{[\mathcal{S}_n]\theta_n} M$
- **空间 vs 物体雅可比**：两种坐标系下的 Jacobian 表示，$J_s=[\mathrm{Ad}_{T_{sb}}]J_b$
- **动力学（Ch 8）**：$\tau=M(\theta)\ddot\theta+h(\theta,\dot\theta)$；单刚体 $\mathcal F_b=\mathcal G_b\dot{\mathcal V}_b-[\mathrm{ad}_{\mathcal V_b}]^\top\mathcal G_b\mathcal V_b$，$\mathcal G_b=\mathrm{diag}(\mathcal I_b,mI)$；牛顿–欧拉前向 / 后向递推求逆动力学；$M(\theta)=\sum_iJ_{ib}^\top\mathcal G_iJ_{ib}$

这套语言是 Pinocchio、Crocoddyl、TSID、Drake 这些现代机器人库的内部实现语言，不是新东西，但教材级清晰阐述较少（此句为知识库作者的判断，教材本身未提这些库；这些库的空间向量实现更接近 Featherstone 的记号，与教材并不逐项相同——推测，未逐库核对）。

> 原文核对注记（对照官方 PDF）：附录 C 的 DH 约定为 $\mathrm{Rot}(\hat x,\alpha_{i-1})\mathrm{Trans}(\hat x,a_{i-1})\mathrm{Trans}(\hat z,d_i)\mathrm{Rot}(\hat z,\phi_i)$；教材中**没有**阻尼最小二乘（DLS）；单位四元数在附录 B.3。

## 局限

- **不覆盖 RL / IL**：本书是 2017 年传统机器人学视角，不涉及深度强化学习、模仿学习、VLA
- **不覆盖 sim2real / 真机部署工程**：纯理论 + 仿真层
- **接触动力学浅尝辄止**：Ch 12 讲的是接触运动学、库仑摩擦、形封闭 / 力封闭与操作；教材 §12.5 明说基于互补问题（LCP / NCP）的摩擦接触仿真「本章未涉及」，完整接触动力学需要更专的资料（如 [MuJoCo 计算文档](../../sources/sites/mujoco_computation_docs.md) 对比 LCP 与凸模型）
- **核对范围说明**：本页对 Ch 2–6、8（部分）、12（部分）与附录 B/C 的表述已对照官方 PDF；Ch 7、9、10、11、13 只依据目录核对，「配套 Coursera 6 门课」未在 PDF 内核实（以官方站点为准）。
- **动力学只含固定基座开链**：Ch 8 的 $M(\theta)$、牛顿–欧拉算法针对基座固定的开链，浮动基系统见 [Floating Base Dynamics](../concepts/floating-base-dynamics.md)

## 工程实践（推荐使用方式）

| 你想做什么 | 建议读哪些章节 |
|-----------|--------------|
| 入门人形/四足控制的数学语言 | Ch 3 → Ch 5 → Ch 8 → Ch 11 |
| 实现 IK 求解器 | Ch 6（数值法部分）+ 配套 Python 库 |
| 理解 Pinocchio / TSID 的内部实现 | Ch 3, 4, 5, 8（PoE + Spatial Vectors） |
| 写传统采样规划（RRT/PRM） | Ch 10 + [PythonRobotics](./python-robotics.md) 可运行示例 |
| 移动机器人导航算法代码直觉 | [PythonRobotics](./python-robotics.md) 定位/规划/跟踪模块 |
| 学抓取力学 | Ch 12 + [friction-cone.md](../formalizations/friction-cone.md) |

## 关联页面

- [正向运动学](../formalizations/forward-kinematics.md) — Ch 4 DH/连乘对照；部署侧再接 PoE
- [逆运动学](../formalizations/inverse-kinematics.md) — Ch 6 解析 / 数值 / 冗余
- [雅可比矩阵](../formalizations/robot-jacobian.md) — Ch 5 空间/物体雅可比与力对偶
- [李群、李代数与刚体旋转](../formalizations/lie-group-rigid-body-motions.md) — Ch 3 李群 / 指数映射的工程分工（含公众号策展导读）
- [SE(3) Representation](../formalizations/se3-representation.md) — 教材 Ch 3 对应的 DL 表示对比
- [Floating Base Dynamics](../concepts/floating-base-dynamics.md) — 教材 Ch 8 在浮基系统上的延伸
- [Whole-Body Control](../concepts/whole-body-control.md) — 教材 Ch 11 控制章节的现代延伸
- [Pinocchio](./pinocchio.md) — 直接使用本教材数学语言的现代机器人库
- [线性代数学习策展](./linear-algebra-curriculum.md) — L0 通用矩阵语言，再接本教材 Ch 2–3
- [Trajectory Optimization](../methods/trajectory-optimization.md) — 教材 Ch 9 的现代化版本
- [PythonRobotics](./python-robotics.md) — Ch 10/13 移动机器人算法的 Python 实现与动画演示
- [开源机器人学学习指南（qqfly）](./learn-robotics-qqfly-guide.md) — 中文自学路线：Craig 工业臂打底后再接本教材 PoE/李群章

## 参考来源

- [sources/papers/modern_robotics_textbook.md](../../sources/papers/modern_robotics_textbook.md)
- Lynch, K. M., & Park, F. C. (2017). *Modern Robotics: Mechanics, Planning, and Control*. Cambridge University Press.
- [Northwestern Mechatronics Wiki — Modern Robotics](https://hades.mech.northwestern.edu/index.php/Modern_Robotics)
- [PDF (官方免费)](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf)
- [sources/papers/robot_link_rotor_inertia_primary_refs.md](../../sources/papers/robot_link_rotor_inertia_primary_refs.md) — 连杆/转子惯量一手资料合集（Ch.8 开放链刚体动力学 + URDF + Gautier–Khalil 1990 + MuJoCo armature）

## 推荐继续阅读

- [Modern Robotics 官方 PDF](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf) 与 [配套资源页](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) — 含 Coursera 专项课程与软件库
- [modernrobotics.org](http://modernrobotics.org) — 作者维护的配套站点（PDF 扉页给出的地址）
