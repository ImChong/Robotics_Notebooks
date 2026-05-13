# 主路线：运动控制算法工程师成长路线

**摘要**：面向人形与双足运动控制的算法工程师成长路线：按 L0–L6 分阶段打通传统控制主干（LIP/ZMP→质心动力学→MPC→TSID/WBC）。

## 三句话先懂这条路线（极简版）

1. **先把传统控制主干打通**：LIP/ZMP → Centroidal → MPC → TSID/WBC。  
2. **再把学习方法接上去**：RL/IL 用来补能力，不是替代控制结构。  
3. **每一层都要有可运行输出**：代码、实验记录、失败复盘，缺一不可。  

## 先看哪里（导航）

- 想看 **最短可执行路径**：跳到 [最小可执行学习路径（90 天版本）](#最小可执行学习路径90-天版本)。  
- 想看 **完整路线**：按 L0 → L6 依次阅读。  
- 想走分支：  
  - [如果目标是 RL 运动控制](learning-paths/if-goal-locomotion-rl.md)  
  - [如果目标是模仿学习与技能迁移](learning-paths/if-goal-imitation-learning.md)  
  - [如果目标是安全控制](learning-paths/if-goal-safe-control.md)
  - [如果目标是接触丰富的操作任务](learning-paths/if-goal-contact-manipulation.md)

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
- 学习路径快速入口：
  - [如果目标是 RL 运动控制](learning-paths/if-goal-locomotion-rl.md)
  - [如果目标是模仿学习与技能迁移](learning-paths/if-goal-imitation-learning.md)
  - [如果目标是安全控制](learning-paths/if-goal-safe-control.md)
  - [如果目标是接触丰富的操作任务](learning-paths/if-goal-contact-manipulation.md)
- 技术栈地图参考：[tech-map/dependency-graph.md](../tech-map/dependency-graph.md)
