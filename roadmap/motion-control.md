# 主路线：运动控制算法工程师成长路线

**摘要**：面向人形与双足运动控制的算法工程师成长路线：按 L0–L6 分阶段打通传统控制主干（LIP/ZMP→质心动力学→MPC→TSID/WBC）。

## 三句话先懂这条路线（极简版）

1. **先把传统控制主干打通**：LIP/ZMP → Centroidal → MPC → TSID/WBC。  
2. **再把学习方法接上去**：RL/IL 用来补能力，不是替代控制结构。  
3. **每一层都要有可运行输出**：代码、实验记录、失败复盘，缺一不可。  

<a id="roadmap-nav-start"></a>

## 先看哪里（导航）

- 想看 **最短可执行路径**：跳到 [最小可执行学习路径（90 天版本）](#最小可执行学习路径90-天版本)。  
- 想看 **完整路线**：按 L0 → L6 依次阅读。  
- 想走纵深（原文见下方 [可选纵深](#depth-optional-index)）：  
  - [如果目标是 RL 运动控制](#depth-rl-locomotion)  
  - [如果目标是模仿学习与技能迁移](#depth-imitation-learning)  
  - [如果目标是安全控制](#depth-safe-control)
  - [如果目标是接触丰富的操作任务](#depth-contact-manipulation)

**这条路线怎么用：**
- 每个阶段都有：前置知识 → 核心问题 → 推荐做什么 → 推荐读什么 → 学完输出什么
- 不只是“学了哪些东西”，而是“学完能做什么”
- 如果某个阶段的前置知识你已经有了，可以跳过，直接从不会的地方开始

**结合 Know-How 的推荐读法：**
- **先看路线层**：先区分自己当前是在补传统控制主干，还是在补强化学习 / 模仿学习分支
- **再看问题层**：始终带着四个问题去学——建模 + 求解、Sim2Real、运动学可行 vs 动力学可行、人形机器人与其他机器人的区别
- **最后看方法层**：每学一个方法，都至少回答三件事——原理是什么、最小代码怎么写、它会在哪些场景失效

**两条主线不要混着学：**
- **传统控制主线：** OCP → LIP/ZMP → Centroidal Dynamics → MPC → TSID/WBC → State Estimation
- **Learning-based 主线：** RL 基础 → locomotion RL → imitation learning / motion prior → sim2real / teacher-student
- 建议优先把传统主线学通，再把 RL / IL 当作扩展层接上去；否则容易只会调超参数，不理解控制结构为什么这样设计

---

## 最小可执行学习路径（90 天版本）

如果你希望“少而精、尽快跑起来”，可以先只做这 5 件事：

1. 跑通一个 [Locomotion](../wiki/tasks/locomotion.md) 仿真环境（站立 + 前进）。  
2. 实现一个倒立摆 [LQR](../wiki/formalizations/lqr.md) 或简单 [MPC](../wiki/methods/model-predictive-control.md)。  
3. 跑通一个最小 [Whole-Body Control](../wiki/concepts/whole-body-control.md) / [TSID](../wiki/concepts/tsid.md) 示例。  
4. 用 PPO 训练一个基础策略，并阅读 [WBC vs RL](../wiki/comparisons/wbc-vs-rl.md) 做方法取舍。  
5. 完成一次最小 [Sim2Real](../wiki/concepts/sim2real.md) checklist（哪怕只在仿真内做 domain randomization 对比）。  

> 完成这 5 件事后，再回到 L0-L6 补理论，会更快理解“为什么要学这些”。

---

## L0 数学与编程基础

**这条不需要深入，但不能跳过。**

**本阶段入口：** [Modern Robotics](../wiki/entities/modern-robotics-book.md)、[SE(3) 表示](../wiki/formalizations/se3-representation.md)、[Pinocchio](../wiki/entities/pinocchio.md)、[Crocoddyl](../wiki/entities/crocoddyl.md)。

### 前置知识
- 高中数学 + 一点微积分直觉
- 会写 Python（能读、能改、能跑通）

### 核心问题
- 线性代数在机器人里到底怎么用（矩阵、向量、变换）
- 优化问题的直觉是什么

### 推荐做什么
- 把 Python / NumPy / Pinocchio 环境的代码跑通一套
- 不用刷题，但要有手感和直觉
- 用 Modern Robotics 配套 Python 库跑通 `MatrixExp3`、`MatrixExp6`、`FKinSpace` 这类最小函数，确认自己能把矩阵指数和刚体位姿变换连起来

### 推荐读什么
- 《Linear Algebra Done Right》（不用全看，只看核心直觉）
- 3Blue1Brown 的线性代数视频（强烈推荐）
- [Modern Robotics](../wiki/entities/modern-robotics-book.md) Ch 2-3：Configuration Space、Rigid-Body Motions
- [SE(3) 表示](../wiki/formalizations/se3-representation.md)（本仓库）
- [Pinocchio](../wiki/entities/pinocchio.md)（本仓库）

### 学完输出什么
- 能用 NumPy 写简单矩阵运算
- 能跑通一个机械臂正运动学 Demo

---

## L1 机器人学骨架

**这条是所有后续内容的基座，跳过后面一定会补。**

**本阶段入口：** [Modern Robotics](../wiki/entities/modern-robotics-book.md)、[Humanoid Robot](../wiki/entities/humanoid-robot.md)、[Pinocchio](../wiki/entities/pinocchio.md)、[Floating Base Dynamics](../wiki/concepts/floating-base-dynamics.md)。

### 前置知识
- L0 内容
- 刚体在三维空间里怎么旋转、怎么描述朝向

### 核心问题
- 机器人每个关节的角度和末端执行器位置是什么关系
- 怎么用数学描述这件事
- 正逆运动学是什么
- 为什么 twist、screw axis、PoE 比只记 D-H 参数更适合接后面的 Pinocchio / TSID / WBC

### 推荐做什么
- 用 Pinocchio 或 Robotics Toolbox 建模一个简单机械臂
- 写出正运动学和逆运动学代码
- 理解雅可比矩阵是什么
- 用 Modern Robotics 的 PoE 公式手写一个 2-3 自由度机械臂的 `FKinSpace` / `JacobianSpace`，再和 Pinocchio 输出对齐

### 推荐读什么
- [Modern Robotics](../wiki/entities/modern-robotics-book.md) Ch 4-6：Forward Kinematics、Velocity Kinematics、Inverse Kinematics
- [斯坦福《机器人学导论》(B站)](https://www.bilibili.com/video/BV17T421k78T/)
- 跑通 Pinocchio 官方 Tutorial
- [Humanoid Robot](../wiki/entities/humanoid-robot.md)（本仓库）
- [Floating Base Dynamics](../wiki/concepts/floating-base-dynamics.md)（本仓库）

### 学完输出什么
- 能自己建模一个简单机器人并计算正逆运动学
- 能解释雅可比矩阵在机器人里是什么、有什么用
- 能区分 space Jacobian 与 body Jacobian，并知道它们在任务空间控制里如何进入速度/力映射

---

## L2 动力学与刚体建模

**从运动学到动力学，是控制机器人最重要的跳跃。**

**本阶段入口：** [Modern Robotics](../wiki/entities/modern-robotics-book.md)、[Floating Base Dynamics](../wiki/concepts/floating-base-dynamics.md)、[Centroidal Dynamics](../wiki/concepts/centroidal-dynamics.md)、[Contact Dynamics](../wiki/concepts/contact-dynamics.md)、[Contact Wrench Cone](../wiki/formalizations/contact-wrench-cone.md)。

### 前置知识
- L1 内容（运动学）
- 一点微积分和常微分方程直觉

### 核心问题
- 关节力矩怎么驱动机器人运动
- 质量矩阵、重力项、科里奥利项是什么
- 浮动基系统（人形机器人的躯干）为什么不能用固定基方法
- wrench、Jacobian transpose、虚功原理如何把任务空间力映射到关节力矩

### 推荐做什么
- 用 Pinocchio 写一个单刚体动力学正逆动力学 Demo
- 理解 centroidal dynamics 的基本形式
- 理解浮动基系统的状态表示问题
- 用 Modern Robotics Ch 8 的开链动力学接口跑一遍 `InverseDynamics` / `MassMatrix` / `ForwardDynamics`，再对照 Pinocchio 的 RNEA / CRBA / ABA

### 推荐读什么
- [Modern Robotics](../wiki/entities/modern-robotics-book.md) Ch 5、Ch 8：Statics、Dynamics of Open Chains
- Featherstone 《Robot Dynamics》相关章节
- Pinocchio 文档的 Centroidal 部分
- [Floating Base Dynamics](../wiki/concepts/floating-base-dynamics.md)（本仓库）
- [Centroidal Dynamics](../wiki/concepts/centroidal-dynamics.md)（本仓库）
- [Contact Dynamics](../wiki/concepts/contact-dynamics.md)（本仓库）

### 学完输出什么
- 能解释正逆动力学在机器人控制里的作用
- 能理解 centroidal dynamics 为什么重要
- 对“这个力矩能让机器人产生什么运动”有直觉
- 能把“任务空间力 / 接触 wrench → 关节力矩”的关系写成 Jacobian transpose 形式

---

## L3 控制基础与最优化

**没有控制理论，后面的 MPC / WBC / RL 全都接不上。**

**本阶段入口：** [Modern Robotics](../wiki/entities/modern-robotics-book.md)、[Optimal Control](../wiki/concepts/optimal-control.md)、[LQR](../wiki/formalizations/lqr.md)、[Model Predictive Control](../wiki/methods/model-predictive-control.md)、[HQP](../wiki/concepts/hqp.md)、[Trajectory Optimization](../wiki/methods/trajectory-optimization.md)。

### 前置知识
- L2 内容（动力学）
- 一点数值优化直觉

### 核心问题
- PID / LQR / MPC 分别在解决什么问题
- QP（二次规划）是什么，为什么在机器人控制里到处都是
- 最优控制的核心思想是什么
- 轨迹生成、反馈控制和约束优化分别处在控制栈的哪一层

### 推荐做什么
- 用 Python 写一个倒立摆的 LQR 控制器
- 用 qpOASES 或 OSQP 跑一个简单 QP
- 理解 MPC 的滚动时域思想
- 复现 Modern Robotics Ch 9 的三次/五次时间缩放轨迹，并给一个机械臂末端轨迹加 PD / computed torque tracking

### 推荐读什么
- [Modern Robotics](../wiki/entities/modern-robotics-book.md) Ch 9、Ch 11：Trajectory Generation、Robot Control
- [Underactuated Robotics](https://arxiv.org/abs/1709.10219)（TEDRAKE）
- 《Robotics: Modelling, Planning and Control》- Siciliano 相关章节
- [LQR](../wiki/formalizations/lqr.md)（本仓库）
- [Optimal Control](../wiki/concepts/optimal-control.md)（本仓库）
- [Model Predictive Control (MPC)](../wiki/methods/model-predictive-control.md)（本仓库）
- [Whole-Body Control](../wiki/concepts/whole-body-control.md)（本仓库）

### 学完输出什么
- 能解释 LQR 和 MPC 的区别
- 能理解 QP 在 WBC 里是解决什么问题的
- 能自己搭一个简单模型的 MPC
- 能说明 computed torque、PD、阻抗控制与后续 WBC 任务控制之间的关系

---

## L4 人形运动控制主干

**这是本路线的核心，也是当前项目的技术栈主干。**

**本阶段入口：** [Modern Robotics](../wiki/entities/modern-robotics-book.md)、[LIP / ZMP](../wiki/concepts/lip-zmp.md)、[Capture Point / DCM](../wiki/concepts/capture-point-dcm.md)、[Centroidal Dynamics](../wiki/concepts/centroidal-dynamics.md)、[Trajectory Optimization](../wiki/methods/trajectory-optimization.md)、[MPC](../wiki/methods/model-predictive-control.md)、[TSID](../wiki/concepts/tsid.md)、[Whole-Body Control](../wiki/concepts/whole-body-control.md)。

这一阶段要建立的不是“我看过多少算法名词”，而是一条稳定的方法链：
- **先学简化模型**：LIP / ZMP 帮你建立步行和平衡直觉
- **再学更真实的中层模型**：Centroidal Dynamics 把接触力、线动量、角动量带进来
- **再学上层规划**：Trajectory Optimization / MPC 负责“未来几步怎么走”
- **最后学下层执行**：TSID / WBC 负责“每个关节怎么出力”

同时，始终用三件事检查自己是否真的学懂：
1. **原理**：这个方法的状态、约束、目标函数分别是什么
2. **最小代码**：我能不能用一个小例子把核心 loop 跑通
3. **局限性**：这个方法在什么情况下会失效，为什么还需要下一层方法接上来

这一整条链路是：

```
LIP / ZMP
  ↓
Centroidal Dynamics
  ↓
Trajectory Optimization / MPC
  ↓
TSID / WBC
```

**Modern Robotics 在 L4 的位置：**
- Ch 3-5 提供任务空间位姿、twist、Jacobian、wrench 的统一语言
- Ch 8 解释开链动力学，帮助理解 Pinocchio / TSID 中的逆动力学项
- Ch 9 解释轨迹生成，是 MPC / trajectory optimization 的低维入口
- Ch 11 解释 computed torque、motion control、force control，是理解 WBC 任务层的前置材料

注意：Modern Robotics 本身不是人形 locomotion 教材，不会直接教 LIP/ZMP、centroidal MPC 或浮动基接触切换；它更像是这条主路线的“语法书”。学 L4 时遇到坐标变换、Jacobian、wrench、逆动力学不清楚，就回到对应章节补。

### L4.1 LIP / ZMP

**前置知识：** [L2 动力学与刚体建模](#l2-动力学与刚体建模) + [L3 控制基础与最优化](#l3-控制基础与最优化)

**核心问题：** 双足机器人怎么在地上走而不倒

**推荐做什么：**
- 实现一个最简单的 ZMP 步态生成
- 用 LIP 模型生成质心轨迹

**推荐读什么：**
- Kajita et al., "Biped walking pattern generation by using preview control of zero-moment point"
- [LIP / ZMP](../wiki/concepts/lip-zmp.md)（本仓库）
- [Capture Point / DCM](../wiki/concepts/capture-point-dcm.md)（本仓库）
- [ZMP / LIP 形式化](../wiki/formalizations/zmp-lip.md)（本仓库）

**学完输出什么：**
- 能解释 ZMP 和支撑多边形的关系
- 能用 LIP 模型生成简单步行轨迹

---

### L4.2 Centroidal Dynamics

**前置知识：** [L4.1 LIP / ZMP](#l41-lip--zmp)

**核心问题：** LIP 简化太狠了，真实人形平衡和接触力怎么描述

**推荐做什么：**
- 用 centroidal dynamics 建模人形机器人
- 理解 centroidal momentum matrix 是什么

**推荐读什么：**
- Orin et al., "Centroidal dynamics of a humanoid robot"
- [Centroidal Dynamics](../wiki/concepts/centroidal-dynamics.md)（本仓库）
- [Contact Dynamics](../wiki/concepts/contact-dynamics.md)（本仓库）

**学完输出什么：**
- 能解释 centroidal dynamics 和 LIP 的区别
- 理解线动量、角动量在平衡控制里的作用

---

### L4.3 Trajectory Optimization / MPC

**前置知识：** [L4.2 Centroidal Dynamics](#l42-centroidal-dynamics) + [L3 控制基础与最优化](#l3-控制基础与最优化)

**核心问题：** 整段质心轨迹和接触力怎么规划，MPC 在线怎么做

**推荐做什么：**
- 用 CasADi 或 Crocoddyl 实现一个 centroidal MPC
- 在仿真里跑通一个双足行走 MPC

**推荐读什么：**
- "Convex MPC for Bipedal Locomotion" (Bellicoso et al.)
- [Trajectory Optimization](../wiki/methods/trajectory-optimization.md)（本仓库）
- [Model Predictive Control (MPC)](../wiki/methods/model-predictive-control.md)（本仓库）
- [MPC 调参指南](../wiki/queries/mpc-tuning-guide.md)（本仓库）
- [MPC 求解器选型](../wiki/queries/mpc-solver-selection.md)（本仓库）

**学完输出什么：**
- 能实现一个简化版的 centroidal MPC
- 能解释预测时域、代价函数设计、约束处理的思路

---

### L4.4 TSID / Whole-Body Control

**前置知识：** [L4.3 Trajectory Optimization / MPC](#l43-trajectory-optimization--mpc)

**核心问题：** 上层规划出来的参考轨迹，怎么变成每个关节该出的力

**推荐做什么：**
- 用 TSID 库实现一个全身任务控制器
- 同时处理躯干稳住、足端跟踪、接触约束

**推荐读什么：**
- Del Prete et al., "Prioritized motion-force control of constrained fully-actuated robots"
- [TSID](../wiki/concepts/tsid.md)（本仓库）
- [TSID Formulation](../wiki/formalizations/tsid-formulation.md)（本仓库）
- [Whole-Body Control](../wiki/concepts/whole-body-control.md)（本仓库）
- [WBC 实现指南](../wiki/queries/wbc-implementation-guide.md)（本仓库）
- [WBC 调参指南](../wiki/queries/wbc-tuning-guide.md)（本仓库）

**学完输出什么：**
- 能用 TSID 框架实现一个多层优先级 WBC
- 能解释任务空间目标怎么映射到关节力矩

---

## L5 强化学习与模仿学习

**学完 L4 后，你应该已经对 model-based control 有了完整理解。L5 是另一条路：learning-based。**

**本阶段入口：** [Reinforcement Learning](../wiki/methods/reinforcement-learning.md)、[Policy Optimization](../wiki/methods/policy-optimization.md)、[PPO vs SAC](../wiki/comparisons/ppo-vs-sac.md)、[Imitation Learning](../wiki/methods/imitation-learning.md)、[Behavior Cloning](../wiki/methods/behavior-cloning.md)、[DAgger](../wiki/methods/dagger.md)、[Motion Retargeting](../wiki/concepts/motion-retargeting.md)。

这一阶段最容易踩的坑，是把 RL / IL 当成“跳过建模”的捷径。更稳的学习方式是：
- 把 RL / IL 看成**能力扩展层**，不是替代所有控制结构的万能钥匙
- 始终追问：这个策略学到的是高层决策、低层 tracking，还是把两者混在一起了
- 遇到 sim2real、接触切换、可解释性问题时，回到 L4 的模型与约束视角重新审题

### L5.1 强化学习基础

**前置知识：** L2 + L3 内容（优化直觉）

**核心问题：** RL 怎么让人形机器人自己学会走路

**推荐做什么：**
- 用 PPO 在简单环境（gymnasium）里训一个策略
- 理解 reward shaping、policy gradient、value function 的意义

**推荐读什么：**
- Spinning Up (OpenAI)
- [Reinforcement Learning](../wiki/methods/reinforcement-learning.md)（本仓库）
- [Policy Optimization](../wiki/methods/policy-optimization.md)（本仓库）
- [PPO vs SAC](../wiki/comparisons/ppo-vs-sac.md)（本仓库）

**学完输出什么：**
- 能解释 PPO 的核心思路
- 能设计一个简单的 RL reward 并训练

---

### L5.2 RL 在人形运动控制里的应用

**前置知识：** L5.1 + L4.3/4.4

**核心问题：** RL 怎么和 MPC / WBC 结合，sim2real 怎么做到

**推荐做什么：**
- 用 IsaacGym / IsaacLab 训练一个人形行走策略
- 尝试 RL + WBC 的组合框架

**推荐读什么：**
- "DeepMimic" (Peng et al.)
- "AMP: Adversarial Motion Priors"
- legged_gym / IsaacGymEnvs
- [legged_gym](../wiki/entities/legged-gym.md)（本仓库）
- [Isaac Gym / Isaac Lab](../wiki/entities/isaac-gym-isaac-lab.md)（本仓库）
- [WBC vs RL](../wiki/comparisons/wbc-vs-rl.md)（本仓库）
- [MPC vs RL](../wiki/comparisons/mpc-vs-rl.md)（本仓库）
- [Query：开源运动控制项目导航](../wiki/queries/open-source-motion-control-projects.md)（本仓库）

**学完输出什么：**
- 能在仿真里训练一个人形行走 RL 策略
- 能解释 RL 和 WBC 各自的优势和局限

---

### L5.3 模仿学习

**前置知识：** L5.1

**核心问题：** 用人类动作数据教机器人做动作

**推荐做什么：**
- 用 MoCap 数据做 motion retargeting
- 尝试 Behavior Cloning + DAgger

**推荐读什么：**
- "ASE: Adversarial Skill Embeddings"
- "DeepMimic"
- [Imitation Learning](../wiki/methods/imitation-learning.md)（本仓库）
- [Behavior Cloning](../wiki/methods/behavior-cloning.md)（本仓库）
- [DAgger](../wiki/methods/dagger.md)（本仓库）
- [Motion Retargeting](../wiki/concepts/motion-retargeting.md)（本仓库）

**学完输出什么：**
- 能把一段 MoCap 数据迁移到人形机器人上
- 能解释 DAgger 为什么比纯 BC 更好

---

## L6 综合实战

**到这里，你应该已经对运动控制和学习两条路都有理解了。最后一步是让它们真正串起来。**

**本阶段入口：** [Sim2Real](../wiki/concepts/sim2real.md)、[System Identification](../wiki/concepts/system-identification.md)、[Domain Randomization](../wiki/concepts/domain-randomization.md)、[Sim2Real Checklist](../wiki/queries/sim2real-checklist.md)、[部署检查清单](../wiki/queries/sim2real-deployment-checklist.md)、[机器人策略调试手册](../wiki/queries/robot-policy-debug-playbook.md)。

### 前置知识
- L4 全流程
- L5 RL 和 IL 的基本操作

### 核心问题
- 怎么从训练到部署形成闭环
- 怎么把仿真训练结果迁移到真实机器人
- 怎么设计一个完整的 RL + WBC pipeline

### 推荐做什么
- 设计并训练一个完整的人形 RL + WBC pipeline
- 做一次 sim2real 迁移
- 调 domain randomization 参数观察效果

### 推荐读什么
- [Sim2Real](../wiki/concepts/sim2real.md)（本仓库）
- [System Identification](../wiki/concepts/system-identification.md)（本仓库）
- [Domain Randomization](../wiki/concepts/domain-randomization.md)（本仓库）
- [Sim2Real Checklist](../wiki/queries/sim2real-checklist.md)（本仓库）
- [Sim2Real 部署检查清单](../wiki/queries/sim2real-deployment-checklist.md)（本仓库）
- [机器人策略调试手册](../wiki/queries/robot-policy-debug-playbook.md)（本仓库）

### 学完输出什么
- 一个能跑的人形 RL 策略（仿真内）
- 一次 sim2real 迁移实验记录
- 对整个 pipeline 的理解文档

---

---
<a id="depth-optional-index"></a>

## 可选纵深（原四条「如果目标是…」学习路径全文并入）

以下对应原先仓库中的 `roadmap/learning-paths/if-goal-*.md`；与上文 [先看哪里（导航）](#roadmap-nav-start) 中的「想走纵深」互链。

<a id="depth-rl-locomotion"></a>

## 学习路径：如果目标是人形 RL 运动控制

> 如果你已经明确想用强化学习路线来做人形机器人运动控制，从这里切入。

**这条路径怎么用：**
- 目标读者是有编程基础、想快速把 RL 和人形 locomotion 串起来的人
- 不需要从头学完所有控制理论，但 RL 基础和 locomotion 概念必须有
- 每个阶段都有前置知识、核心问题、推荐做什么、推荐读什么、学完输出什么

**和主路线的关系：**
- 本路径是主路线的“快速分支版本”
- 如果你在某个阶段遇到理论卡点，回到 [主路线：运动控制成长路线](motion-control.md) 查对应章节

---

### Stage 0 RL 基础准备

**如果已经有 RL 基础，可以跳过这个阶段。**

### 前置知识
- Python 熟练
- 深度学习基础（知道 MLP、loss、梯度反向传播是什么）
- 一点概率统计直觉

### 核心问题
- RL 在解决什么问题
- Policy gradient 和 Q-learning 的核心区别是什么
- PPO 为什么是当前最主流的机器人 RL 算法

### 推荐做什么
- 用 Stable-Baselines3 或 Spinning Up 跑一个倒立摆或 HalfCheetah 环境
- 对比 on-policy（PPO）和 off-policy（SAC）的训练曲线

### 推荐读什么
- Spinning Up (OpenAI) — Part 1: Key Concepts
- [Reinforcement Learning](../wiki/methods/reinforcement-learning.md)（本仓库）

### 学完输出什么
- 能解释什么是 MDP、policy、value function、return
- 能在简单环境里用 PPO 训练一个可用的策略

---

### Stage 1 人形 locomotion 概念基础

**不做 model-based control 的人形 RL，也需要懂 locomotion 在解决什么问题。**

### 前置知识
- Stage 0 内容
- 刚体动力学基本直觉（力、力矩、加速度）

### 核心问题
- 人形机器人在 locomotion 时面临的核心挑战是什么
- 为什么平衡、接触切换、高维动作空间是难题
- RL 在 locomotion 里能解决什么问题、不能解决什么

### 推荐做什么
- 读 2-3 篇 recent humanoid RL 论文的 related work / introduction（不用全懂，大概知道大家在解决什么问题）
- 在 IsaacGym 或 Mujoco 里跑通一个人形环境

### 推荐读什么
- [Locomotion](../wiki/tasks/locomotion.md)（本仓库）
- [WBC vs RL](../wiki/comparisons/wbc-vs-rl.md)（本仓库）

### 学完输出什么
- 能解释为什么 RL 适合做 locomotion
- 能在仿真里让一个人形模型站起来走几步

---

### Stage 2 RL + Locomotion 核心方法

### 前置知识
- Stage 0 + Stage 1 内容
- 理解 reward shaping、termination condition、observation space 设计

### 核心问题
- 人形 RL 的 reward 怎么设计（平衡 reward + 前进 reward + 平滑 reward）
- 观测空间怎么构建（关节角、角速度、IMU、接触状态）
- 动作空间是 position、velocity 还是 torque
- 训练不稳定的原因是什么（sparse reward、early termination、exploration）

### 推荐做什么
- 自己设计一个简单人形 RL reward，在 IsaacGym 或 Mujoco Humanoid 里训练
- 对比不同 action space（position / velocity / torque）的训练效果
- 调一调 reward weight 看看对步态的影响

### 推荐读什么
- "Emergence of Locomotion Behaviours in Rich Environments" (Heess et al., 2017)
- legged_gym README 和代码
- [Reinforcement Learning](../wiki/methods/reinforcement-learning.md)（本仓库）

### 学完输出什么
- 一个能在平地上稳定行走的人形 RL 策略（仿真内）
- 对 reward 设计有第一手直觉

---

### Stage 3 MPC / WBC + RL 组合框架

**纯 RL 很难直接用到高精度控制场景，需要和传统控制框架结合。**

### 前置知识
- Stage 2 内容
- 理解 MPC 和 WBC 是什么（不需要能写，但需要懂基本思想）

### 核心问题
- RL 策略和 MPC / WBC 怎么组合
- 常见框架：RL 训练低层策略 + MPC 做高层规划；RL 提供 prior 给优化器
- 什么时候该用 RL，什么时候该用 MPC

### 推荐做什么
- 读 1-2 篇 RL + WBC / MPC 结合的论文（推荐：AMP、DeepMimic、MimicKit）
- 理解"RL 提供动作 prior，QP/WBC 负责实时跟踪"这个模式

### 推荐读什么
- "AMP: Adversarial Motion Priors" (Peng et al., 2021)
- [Whole-Body Control](../wiki/concepts/whole-body-control.md)（本仓库）
- [Model Predictive Control (MPC)](../wiki/methods/model-predictive-control.md)（本仓库）

### 学完输出什么
- 能解释 RL 和 MPC / WBC 各自的适用场景
- 能判断一个新问题是适合 RL 解还是 MPC 解

---

### Stage 4 Sim2Real 迁移

**仿真里能跑是真机部署的前提，但不是全部。**

### 前置知识
- Stage 3 内容
- 理解 Domain Randomization 的原理

### 核心问题
- 为什么 sim2real gap 存在（物理参数不准、执行延迟、传感噪声）
- Domain Randomization 的正确打开方式（随机化哪些参数、范围怎么定）
- 动作延迟怎么处理
- 在线微调什么时候有用、什么时候没用

### 推荐做什么
- 给 Stage 2 训练好的策略加 DR，再零样本迁移到更难的仿真地形
- 调 DR 范围，观察策略鲁棒性变化

### 推荐读什么
- [Sim2Real](../wiki/concepts/sim2real.md)（本仓库）
- [Domain Randomization](../wiki/concepts/domain-randomization.md)（本仓库）
- [System Identification](../wiki/concepts/system-identification.md)（本仓库）

### 学完输出什么
- 一次 sim2real 实验记录（哪怕只是仿真内不同地形的迁移）
- 对 DR 参数设置的直觉

---

### Stage 5 进阶方向

### 前置知识
- Stage 4 内容

根据研究方向选一个深入：

**方向 A：更复杂的 locomotion**
- 跑、跳、楼梯、崎岖地形
- 关键词：CPI、NMPC、WBC、RL + sim2real

**方向 B：模仿学习初始化**
- 用 MoCap 数据初始化 RL 策略
- 关键词：ASE、CALM、DeepMimic

**方向 C：视觉 + locomotion**
- 端到端视觉策略
- 关键词：perception、terrain mapping、learning-based navigation

**方向 D：单位置适配**
- 一个策略迁移到不同机器人形态
- 关键词：domain randomization、meta-learning、multi-task RL

### 推荐读什么
- 参考 [Humanoid Control Roadmap](../wiki/roadmaps/humanoid-control-roadmap.md) 的进阶专题部分

---

### 快速入口汇总

| 阶段 | 核心问题 | 本仓库入口 |
|------|---------|-----------|
| Stage 0 | RL 基础 | [Reinforcement Learning](../wiki/methods/reinforcement-learning.md) |
| Stage 1 | locomotion 概念 | [Locomotion](../wiki/tasks/locomotion.md) |
| Stage 2 | RL + locomotion 方法 | [Reinforcement Learning](../wiki/methods/reinforcement-learning.md) |
| Stage 3 | RL + 控制组合 | [Whole-Body Control](../wiki/concepts/whole-body-control.md) |
| Stage 4 | Sim2Real | [Sim2Real](../wiki/concepts/sim2real.md) |

### 和其他页面的关系

- 完整成长路线参考：[主路线：运动控制算法工程师成长路线](motion-control.md)
- 人形控制全景图：[Humanoid Control Roadmap](../wiki/roadmaps/humanoid-control-roadmap.md)
- 技术栈地图：[tech-map/dependency-graph.md](../tech-map/dependency-graph.md)

---

<a id="depth-imitation-learning"></a>

## 学习路径：如果目标是模仿学习与技能迁移

> 如果你更感兴趣的是怎么把人类的动作迁移到机器人上，从这里切入。

**这条路径怎么用：**
- 目标读者是有深度学习基础、想理解如何让机器人从人类演示中学习技能的人
- 重点不是理论证明，而是：从数据到策略的完整 pipeline
- 每个阶段都有前置知识、核心问题、推荐做什么、推荐读什么、学完输出什么

**和主路线的关系：**
- 模仿学习（IL）和强化学习（RL）在很多实际项目里是组合使用，不是非此即彼
- 本路径和 RL 路径在 Stage 3 之后有很多交叉
- 如果你不知道该走哪条，先走 IL 路径，因为它更容易出可感知的结果

---

### Stage 0 深度学习与时序建模基础

**如果已经有 PyTorch 熟练度和序列模型基础，可以跳过。**

### 前置知识
- Python 熟练
- 理解 MLP、loss、梯度反向传播
- 知道什么叫监督学习

### 核心问题
- 序列数据（关节角、MoCap、动作）怎么用神经网络建模
- RNN / LSTM / Transformer 在时序建模上的核心区别是什么
- diffusion model 在生成式建模里是什么角色

### 推荐做什么
- 用 PyTorch 跑一个 LSTM 预测简单时序数据的 Demo
- 对比 MLP 和 LSTM 在时序任务上的表现差异

### 推荐读什么
- "Illustrated Guide to LSTM" (Google Blog)
- 跑通一个 Motion Transformer 官方 Demo（如果能访问）

### 学完输出什么
- 能解释为什么时序数据需要特殊建模方法
- 能用 LSTM 对简单序列做预测

---

### Stage 1 模仿学习核心概念

### 前置知识
- Stage 0 内容
- 理解什么是监督学习

### 核心问题
- Behavior Cloning 的核心思想是什么
- 为什么 BC 会有 compounding error（协奏曲错误）
- DAgger 为什么能缓解 compounding error
- 模仿学习和强化学习的根本区别是什么

### 推荐做什么
- 用 BC 训练一个简单机械臂跟随演示轨迹
- 对比 BC 和 DAgger 在长时程任务上的效果差异

### 推荐读什么
- "A Reduction of Imitation Learning and Stochastic Gradient Descent to Online Learning" (Ross & Bagnell, 2010)
- [Imitation Learning](../wiki/methods/imitation-learning.md)（本仓库）

### 学完输出什么
- 能解释 compounding error 是什么、为什么出现
- 能在简单任务里用 BC 训练一个可用的策略

---

### Stage 2 Motion Retargeting（动作迁移）

**这是人形机器人技能学习的核心技术：从人类动作到机器人动作。**

### 前置知识
- Stage 1 内容
- 理解 kinematics 和 inverse kinematics 基础

### 核心问题
- 为什么不能直接把人类关节角度映射到机器人（骨骼结构不同）
- 怎么用 IK 或 learning-based 方法做 retargeting
- 人体动作数据的不同来源（MoCap、VRI、视频）各有什么优缺
- retargeting 后的数据还需要哪些后处理（时间对齐、重采样、姿态约束）

### 推荐做什么
- 用一套 MoCap 数据，通过 IK 或 retargeting 方法迁移到人形机器人模型上
- 观察迁移后动作的可行性（关节限位、自碰撞、地面穿透）

### 推荐读什么
- "SFV: Surveillance from Videos" / "DensePose" 相关 retargeting 工作
- "ASE: Adversarial Skill Embeddings" (Peng et al., 2022) — 有 retargeting pipeline 描述

### 学完输出什么
- 一段成功 retargeting 到人形机器人模型上的人类走路数据
- 对骨骼结构差异导致的问题有第一手直觉

---

### Stage 3 Diffusion Policy 与生成式动作

**Diffusion Policy 是 2023-2024 年在机器人模仿学习里最活跃的方向。**

### 前置知识
- Stage 2 内容
- 理解 diffusion model 的基本原理（不需要能写，但需要懂去噪过程）

### 核心问题
- Diffusion Policy 和传统 BC 的核心区别是什么
- 为什么 diffusion model 在高维动作空间表现更好
- 怎么把视觉输入结合进 diffusion policy
- diffusion 采样时间过长怎么解决

### 推荐做什么
- 用一个开源 Diffusion Policy 实现（如 RoboDiff、Diffusion Policy 官方）跑一个简单任务
- 对比 diffusion policy 和 LSTM BC 在同样任务上的效果

### 推荐读什么
- "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (Chi et al., 2023)
- [Imitation Learning](../wiki/methods/imitation-learning.md)（本仓库）

### 学完输出什么
- 一个用 Diffusion Policy 训练的动作策略
- 能解释 diffusion process 在机器人动作生成里的优势

---

### Stage 4 技能嵌入与对抗式学习

**单个技能会了，怎么让机器人同时掌握多个技能、并在新场景里组合？**

### 前置知识
- Stage 3 内容

### 核心问题
- 什么是 skill embedding，为什么需要把技能压缩到隐空间
- 对抗式模仿学习（ASE）和普通 BC 的区别是什么
- 怎么在一个隐空间里做技能插值和组合
- 为什么 latent variable 能帮助解决 compounding error

### 推荐做什么
- 读懂 ASE 的方法 pipeline
- 在能找到的开源代码上跑一个 two-skill interpolation 实验

### 推荐读什么
- "ASE: Adversarial Skill Embeddings for Hierarchical Reinforcement Learning" (Peng et al., 2022)
- "Learning Latent Plans from Play" (Lynch et al., 2020)

### 学完输出什么
- 能解释 skill embedding 的意义
- 对对抗式学习方法在机器人技能学习里的作用有直观理解

---

### Stage 5 仿真到真实迁移

**模仿学习训练的策略，迁移到真实机器人上会遇到哪些问题？**

### 前置知识
- Stage 4 内容
- 理解 sim2real gap 的基本概念

### 核心问题
- IL 训练数据和真实机器人动作空间的差异怎么处理
- 观测空间不匹配（相机角度、传感器噪声）怎么处理
- 在线微调（online fine-tuning）对 IL 策略有没有用
- 怎么判断一个 IL 策略是“真的学会了”还是“在记忆演示”

### 推荐做什么
- 给 Stage 2/3 训练的策略加动作空间噪声和观测噪声，观察鲁棒性
- 设计一个简单的 domain randomization 实验

### 推荐读什么
- [Sim2Real](../wiki/concepts/sim2real.md)（本仓库）
- [Domain Randomization](../wiki/concepts/domain-randomization.md)（本仓库）

### 学完输出什么
- 对 IL 策略的 sim2real 差距有第一手认识
- 能设计针对性的 DR 实验来提升策略鲁棒性

---

### Stage 6 进阶方向

### 前置知识
- Stage 5 内容

**方向 A：Video-based IL**
- 用 RGB 视频而非 MoCap 做动作迁移
- 关键词：Pose estimation、Video imitation、DensePose

**方向 B：Multi-modal IL**
- 结合视觉、触觉、力传感器做多模态技能学习
- 关键词：multimodal、haptic、force feedback

**方向 C：Long-horizon 任务**
- 把多个技能串成一个长序列
- 关键词：task planning、skill chaining、HTN

**方向 D：Humanoid 特有技能**
- 走路、跑步、跳跃、平衡
- 关键词：locomotion IL、motion retargeting for humanoid

---

### 快速入口汇总

| 阶段 | 核心问题 | 本仓库入口 |
|------|---------|-----------|
| Stage 0 | 时序建模基础 | [Imitation Learning](../wiki/methods/imitation-learning.md) |
| Stage 1 | BC / DAgger | [Imitation Learning](../wiki/methods/imitation-learning.md) |
| Stage 2 | Motion Retargeting | [Imitation Learning](../wiki/methods/imitation-learning.md) |
| Stage 3 | Diffusion Policy | [Imitation Learning](../wiki/methods/imitation-learning.md) |
| Stage 4 | Skill Embedding | [Imitation Learning](../wiki/methods/imitation-learning.md) |
| Stage 5 | Sim2Real | [Sim2Real](../wiki/concepts/sim2real.md) |

### 和其他页面的关系

- 完整成长路线参考：[主路线：运动控制算法工程师成长路线](motion-control.md)
- RL 路径对比参考：[如果目标是 RL 运动控制](#depth-rl-locomotion)
- 人形控制全景图：[Humanoid Control Roadmap](../wiki/roadmaps/humanoid-control-roadmap.md)
- 技术栈地图：[tech-map/dependency-graph.md](../tech-map/dependency-graph.md)

---

<a id="depth-safe-control"></a>

## 学习路径：如果目标是安全控制（CLF / CBF / Safe RL）

> 如果你想让机器人在满足安全约束的前提下运动，从这里切入。

**这条路径怎么用：**
- 目标读者是有控制理论基础、想加入安全保证的工程师或研究者
- 需要有基础线性代数和微分方程直觉；RL 基础有助于后期阶段
- 每个阶段有前置知识、核心问题、推荐做什么、学完输出什么

---

### Stage 0 数学基础

### 前置知识
- 线性代数（矩阵、特征值）
- 微分方程基础（稳定性直觉）
- 一点凸优化基础（QP 是什么）

### 核心问题
- Lyapunov 稳定性是什么意思
- 为什么 Lyapunov 函数能证明系统收敛

### 推荐读什么
- [Lyapunov 稳定性形式化](../wiki/formalizations/lyapunov.md)
- Khalil, *Nonlinear Systems* — Chapter 4（稳定性定义）

### 学完输出什么
- 能手工验证一个简单系统的 Lyapunov 稳定性
- 理解正定函数和负半定导数的含义

---

### Stage 1 CLF / CBF 基础

### 核心问题
- CLF 和 CBF 分别解决什么问题（收敛 vs. 安全边界）
- 两者如何联合放入 QP 优化

### 推荐读什么
- [Control Lyapunov Function](../wiki/formalizations/control-lyapunov-function.md)
- [Control Barrier Function](../wiki/concepts/control-barrier-function.md)
- [CLF vs CBF 对比](../wiki/comparisons/clf-vs-cbf.md)
- Ames et al., *Control Barrier Function based Quadratic Programs* (2017)

### 推荐做什么
- 用 Python + CVXPY 实现一个 CBF-QP，保证 2D 小车不越过边界

### 学完输出什么
- 能解释 CLF 和 CBF 的数学定义和功能区别
- 能写出 CBF-QP 的标准形式

---

### Stage 2 CLF+CBF 在 WBC/MPC 中的应用

### 核心问题
- 如何把 CLF 和 CBF 约束嵌入 WBC 的 QP 层
- Safety filter 和 MPC safety constraint 有什么区别

### 推荐读什么
- [Whole-Body Control](../wiki/concepts/whole-body-control.md)
- [Query：CLF+CBF 在 WBC/MPC 中联合使用](../wiki/queries/clf-cbf-in-wbc.md)
- Zeng et al., *Safety-Critical Model Predictive Control* (2021)

### 推荐做什么
- 在一个简单 locomotion 环境中加入 CBF safety filter，观察对步态的影响

### 学完输出什么
- 能描述 safety filter 的工作方式
- 能识别 WBC 中哪些约束层可以加 CLF/CBF 项

---

### Stage 3 Safe RL

### 核心问题
- Constrained MDP（CMDP）和标准 MDP 的区别
- 如何用 Lagrangian 方法或 barrier 方法训练安全策略

### 推荐读什么
- Garcia & Fernandez, *A Comprehensive Survey on Safe RL* (2015)
- [Reinforcement Learning](../wiki/methods/reinforcement-learning.md)

### 推荐做什么
- 用 Safety Gym（OpenAI）或 safe-control-gym 跑一个 constrained RL 实验

### 学完输出什么
- 能解释 CMDP 的形式化定义
- 能对比 model-based 安全保证 vs. model-free 安全奖励塑形的优劣

---

### 关联页面

- [Lyapunov 稳定性](../wiki/formalizations/lyapunov.md)
- [Control Lyapunov Function](../wiki/formalizations/control-lyapunov-function.md)
- [Control Barrier Function](../wiki/concepts/control-barrier-function.md)
- [CLF vs CBF 对比](../wiki/comparisons/clf-vs-cbf.md)
- [Whole-Body Control](../wiki/concepts/whole-body-control.md)
- [Query：CLF+CBF 在 WBC/MPC 中联合使用](../wiki/queries/clf-cbf-in-wbc.md)

---

<a id="depth-contact-manipulation"></a>

## 学习路径：如果目标是接触丰富的操作任务

> 如果你想让机器人完成装配、拧螺丝、插拔等需要精细接触的任务，从这里切入。

**这条路径怎么用：**
- 目标读者是有 RL/IL 基础、想深入操作任务的工程师
- 需要了解基础动力学和控制，Python 编程熟练
- 每个阶段有前置知识、核心问题、推荐做什么、学完输出什么

---

### Stage 0 操作基础

### 前置知识
- 机器人运动学基础（正逆运动学）
- 基础控制理论（PID、阻抗控制概念）
- Python + ROS 或类似框架

### 核心问题
- 什么叫"接触丰富"，和 free-space manipulation 有什么区别
- 刚性抓取和顺应性控制分别适合什么场景

### 推荐读什么
- [Manipulation](../wiki/tasks/manipulation.md)
- [Contact-Rich Manipulation](../wiki/concepts/contact-rich-manipulation.md)
- Mason, *Mechanics of Robotic Manipulation* — Chapter 1-2

### 学完输出什么
- 能区分 prehensile / non-prehensile / contact-rich 操作
- 理解阻抗控制（impedance control）的基本原理

---

### Stage 1 接触力建模与控制

### 核心问题
- 怎么建模接触力和摩擦
- 阻抗控制 vs. 导纳控制 vs. 力控有什么区别

### 推荐读什么
- [Contact Dynamics](../wiki/concepts/contact-dynamics.md)
- [Whole-Body Control](../wiki/concepts/whole-body-control.md)
- Hogan, *Impedance Control* (1985)

### 推荐做什么
- 在 MuJoCo 或 Isaac Sim 中实现一个 impedance controller，完成推箱子任务
- 观察接触刚度参数对稳定性的影响

### 学完输出什么
- 能写出 impedance control 的数学形式
- 能解释 soft contact 模型（MuJoCo）vs. hard contact 的区别

---

### Stage 2 模仿学习用于操作

### 核心问题
- 为什么 contact-rich 任务适合 IL 而不是 RL
- ACT 和 Diffusion Policy 各适合什么场景

### 推荐读什么
- [Imitation Learning](../wiki/methods/imitation-learning.md)
- [Behavior Cloning](../wiki/methods/behavior-cloning.md)
- [Bimanual Manipulation](../wiki/tasks/bimanual-manipulation.md)
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT, 2023)

### 推荐做什么
- 用 ACT 框架收集遥操作数据并训练一个抓取策略
- 对比 chunk size 对 contact-rich 任务成功率的影响

### 学完输出什么
- 能解释 ACT 的 action chunking 机制和 Diffusion Policy 的多模态优势
- 能设计一个 contact-rich 任务的数据采集方案

---

### Stage 3 Contact-Rich 策略进阶

### 核心问题
- 如何在 sim2real 中处理 contact-rich 任务的接触不一致问题
- 有哪些专门针对 contact-rich 的学习方法

### 推荐读什么
- [Query：接触丰富操作实践指南](../wiki/queries/contact-rich-manipulation-guide.md)
- [Demo Data Collection Guide](../wiki/queries/demo-data-collection-guide.md)
- Luo et al., *DEFT: Dexterous Fine-Grained Manipulation Transformer* (2024)

### 推荐做什么
- 尝试在真机上部署 ACT 策略，使用 FT sensor 记录接触力
- 分析失败案例，找出接触力误差模式

### 学完输出什么
- 能识别 contact-rich sim2real 迁移的主要瓶颈
- 能设计 FT sensor feedback 的简单修正机制

---

### 关联页面

- [Manipulation](../wiki/tasks/manipulation.md)
- [Contact-Rich Manipulation](../wiki/concepts/contact-rich-manipulation.md)
- [Bimanual Manipulation](../wiki/tasks/bimanual-manipulation.md)
- [Imitation Learning](../wiki/methods/imitation-learning.md)
- [Behavior Cloning](../wiki/methods/behavior-cloning.md)
- [Query：接触丰富操作实践指南](../wiki/queries/contact-rich-manipulation-guide.md)

---

## 常见卡点

### 1. 学了一堆理论，不知道怎么串起来
解决思路：按上面 L0 → L6 的顺序走，每个阶段都有输出物，不要只看不练。

### 2. RL 训练不稳定，不知道怎么调
解决思路：从 IL 初始化 RL 起步（先给一个好的 policy prior），比纯 RL 从零训稳得多。

### 3. 不知道自己的模型对不对
解决思路：先用 WBC / MPC 这类 model-based 方法做 baseline，RL 结果要有对比才知道好不好。

### 4. Sim2Real 差距太大
解决思路：先做好 System Identification，再用 Domain Randomization 扩大扰动范围，最后考虑在线自适应。

### 5. Modern Robotics 看完了，但不知道和人形控制怎么接
解决思路：把它当作数学语言和固定基机器人基础。Ch 3-6 接 L0/L1 的 SE(3)、PoE、Jacobian；Ch 8 接 L2 的动力学和 Pinocchio；Ch 9/11 接 L3 的轨迹生成和控制；真正进入人形后，还要补 [Floating Base Dynamics](../wiki/concepts/floating-base-dynamics.md)、[Centroidal Dynamics](../wiki/concepts/centroidal-dynamics.md)、[Contact Dynamics](../wiki/concepts/contact-dynamics.md) 和 [Whole-Body Control](../wiki/concepts/whole-body-control.md)。

---

## 和其他页面的关系

- 本路线是 `Robotics_Notebooks` 当前最核心的执行入口
- 传统机器人学主教材：[Modern Robotics](../wiki/entities/modern-robotics-book.md)
- 更详细的阶段参考：[Humanoid Control Roadmap](../wiki/roadmaps/humanoid-control-roadmap.md)
- 实战经验补充：[Query：人形机器人运动控制 Know-How](../wiki/queries/humanoid-motion-control-know-how.md)
- 可选纵深快速入口（锚点）：
  - [如果目标是 RL 运动控制](#depth-rl-locomotion)
  - [如果目标是模仿学习与技能迁移](#depth-imitation-learning)
  - [如果目标是安全控制](#depth-safe-control)
  - [如果目标是接触丰富的操作任务](#depth-contact-manipulation)
- 技术栈地图参考：[tech-map/dependency-graph.md](../tech-map/dependency-graph.md)
