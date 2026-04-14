# Robotics Notebooks Index

本项目以运动控制为切入口，通向机器人全栈工程能力。

这是知识入口总索引。**如果你第一次来，从 [README](README.md) 开始**，那里有完整的使用说明。

---

## 快速入口

| 你的目标 | 从这里进入 |
|---------|-----------|
| 想有一条学习路线照着走 | [路线A：运动控制成长路线](roadmap/route-a-motion-control.md) |
| 想用强化学习做 locomotion | [RL Locomotion 学习路径](roadmap/learning-paths/if-goal-locomotion-rl.md) |
| 想学模仿学习与技能迁移 | [IL 学习路径](roadmap/learning-paths/if-goal-imitation-learning.md) |
| 想看知识概念和方法 | 直接翻下面的 wiki 目录 |
| 想看模块关系和依赖 | [tech-map 总览](tech-map/overview.md) |

---

## 四个模块的分工

如果你想知道每个目录负责什么：

### wiki/ — 结构化知识页
回答"某个概念或方法是什么"。

重点页面：
- [Optimal Control (OCP)](wiki/concepts/optimal-control.md)
- [LIP / ZMP](wiki/concepts/lip-zmp.md)
- [Centroidal Dynamics](wiki/concepts/centroidal-dynamics.md)
- [TSID](wiki/concepts/tsid.md)
- [Whole-Body Control](wiki/concepts/whole-body-control.md)
- [State Estimation](wiki/concepts/state-estimation.md)
- [System Identification](wiki/concepts/system-identification.md)
- [Floating Base Dynamics](wiki/concepts/floating-base-dynamics.md)
- [Contact Dynamics](wiki/concepts/contact-dynamics.md)
- [Capture Point / DCM](wiki/concepts/capture-point-dcm.md)
- [Sim2Real](wiki/concepts/sim2real.md)
- [Reinforcement Learning](wiki/methods/reinforcement-learning.md)
- [Imitation Learning](wiki/methods/imitation-learning.md)
- [Model Predictive Control (MPC)](wiki/methods/model-predictive-control.md)
- [Trajectory Optimization](wiki/methods/trajectory-optimization.md)
- [Locomotion](wiki/tasks/locomotion.md)
- [WBC vs RL](wiki/comparisons/wbc-vs-rl.md)
- [Isaac Gym / Isaac Lab](wiki/entities/isaac-gym-isaac-lab.md)
- [MuJoCo](wiki/entities/mujoco.md)
- [legged_gym](wiki/entities/legged-gym.md)
- [Pinocchio](wiki/entities/pinocchio.md)
- [Crocoddyl](wiki/entities/crocoddyl.md)
- [Unitree](wiki/entities/unitree.md)

### roadmap/ — 成长路线
回答"应该先学什么、再学什么、学完输出什么"。

核心路线：
- [路线A：运动控制成长路线](roadmap/route-a-motion-control.md)
- [RL Locomotion 路径](roadmap/learning-paths/if-goal-locomotion-rl.md)
- [IL 技能迁移路径](roadmap/learning-paths/if-goal-imitation-learning.md)

### tech-map/ — 技术栈地图
回答"模块之间是什么关系"。

- [技术栈模块总览](tech-map/overview.md)
- [模块依赖关系图](tech-map/dependency-graph.md)

### references/ — 论文导航
按主题整理的论文列表，补充 wiki 的深度阅读需求。

---

## 当前推荐阅读顺序

如果你从人形机器人运动控制切入，建议顺序：

1. [LIP / ZMP](wiki/concepts/lip-zmp.md)
2. [Centroidal Dynamics](wiki/concepts/centroidal-dynamics.md)
3. [Trajectory Optimization](wiki/methods/trajectory-optimization.md)
4. [Model Predictive Control (MPC)](wiki/methods/model-predictive-control.md)
5. [TSID](wiki/concepts/tsid.md)
6. [Whole-Body Control](wiki/concepts/whole-body-control.md)
7. [State Estimation](wiki/concepts/state-estimation.md)
8. [System Identification](wiki/concepts/system-identification.md)
9. [Sim2Real](wiki/concepts/sim2real.md)
10. [Locomotion](wiki/tasks/locomotion.md)

---

## 当前主线知识链

```
LIP / ZMP
  ↓
Centroidal Dynamics
  ↓
Trajectory Optimization / MPC
  ↓
TSID / WBC
  ↓
State Estimation / System Identification / Sim2Real
```

---

## Obsidian Dataview 动态查询

> 以下查询块在 Obsidian + Dataview 插件中可直接使用。在 GitHub/普通 Markdown 渲染器中显示为代码块。

**查看所有 concept 页面及状态：**
```dataview
TABLE status, tags FROM "wiki/concepts"
SORT file.name ASC
```

**查看所有 draft 状态的页面（待完善）：**
```dataview
TABLE type, file.folder FROM "wiki"
WHERE status = "draft"
SORT type ASC
```

**查看带 rl 标签的所有页面：**
```dataview
LIST FROM "wiki"
WHERE contains(tags, "rl")
SORT type ASC
```

---

## 与其他项目的边界

- [`Humanoid_Robot_Learning_Paper_Notebooks`](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks)：单篇论文深读
- [`Robotics_Notebooks`](https://github.com/ImChong/Robotics_Notebooks)：跨模块知识组织
- [`ImChong.github.io`](https://github.com/ImChong/ImChong.github.io)：个人简历

> 论文项目负责点，技术栈项目负责线和面，个人主页负责展示。

---




### Entities（实体页）

- [Crocoddyl](entities/crocoddyl.md) — Crocoddyl** 是一个面向机器人最优控制与轨迹优化的开源工具箱，长期由 **LAAS-CNRS / INRIA / Gepetto / Stack-of-Tasks** 这条学术与开源路线 `📅unknown` `[entity_page]`
- [Isaac Gym / Isaac Lab](entities/isaac-gym-isaac-lab.md) — Isaac Gym** 和 **Isaac Lab** 是 NVIDIA 机器人强化学习生态里的两代核心框架。 `📅unknown` `[entity_page]`
- [legged_gym](entities/legged-gym.md) — legged_gym** 是 ETH Zurich RSL（Robotic Systems Lab）开源的足式机器人强化学习训练框架，建立在 **Isaac Gym** 之上。 `📅unknown` `[entity_page]`
- [MuJoCo](entities/mujoco.md) — MuJoCo（Multi-Joint dynamics with Contact）** 是机器人与控制领域最经典的物理引擎之一，现在由 **Google DeepMind** 维护并开源。 `📅unknown` `[entity_page]`
- [Pinocchio](entities/pinocchio.md) — Pinocchio** 是机器人领域最主流的刚体运动学与动力学计算库之一，长期由 **Stack-of-Tasks / INRIA / LAAS-CNRS** 这条学术与开源路线推动。 `📅unknown` `[entity_page]`
- [Unitree](entities/unitree.md) — Unitree Robotics（宇树科技）** 是当前腿式机器人和人形机器人领域最有影响力的公司之一。 `📅unknown` `[entity_page]`

### Wiki Concepts（概念页）

- [Capture Point / DCM](concepts/capture-point-dcm.md) — Capture Point（捕获点）** 和 **DCM（Divergent Component of Motion，发散运动分量）** 是腿式机器人动态平衡与步态控制里两个非常关键的概念，用来描 `📅unknown` `[wiki_page]`
- [Centroidal Dynamics](concepts/centroidal-dynamics.md) — Centroidal Dynamics（质心动力学）**：用机器人整体质心的线动量和角动量来描述全身动力学的一种中层建模方式。 `📅unknown` `[wiki_page]`
- [Contact Dynamics](concepts/contact-dynamics.md) — Contact Dynamics（接触动力学）**：研究机器人与地面、物体、墙面等环境发生接触时，接触力、约束和系统运动之间关系的动力学框架。 `📅unknown` `[wiki_page]`
- [Domain Randomization](concepts/domain-randomization.md) — 域随机化**：在仿真训练中主动随机化物理参数、视觉纹理、环境设置，让策略被迫学会适应各种变化的泛化能力，从而实现零样本从仿真迁移到现实。 `📅unknown` `[wiki_page]`
- [Floating Base Dynamics](concepts/floating-base-dynamics.md) — Floating Base Dynamics（浮动基动力学）**：描述机器人在基座不固定于世界坐标系时，其整体动力学如何建模与控制的框架。 `📅unknown` `[wiki_page]`
- [LIP / ZMP](concepts/lip-zmp.md) — LIP（Linear Inverted Pendulum, 线性倒立摆）** 和 **ZMP（Zero Moment Point, 零力矩点）** 是双足机器人行走控制里最经典的一对基础模型与稳定 `📅unknown` `[wiki_page]`
- [MPC 与 WBC 集成：人形机器人 locomotion 的典型控制架构](concepts/mpc-wbc-integration.md) — MPC 负责"大尺度规划"（质心往哪走、落脚点放哪），WBC 负责"全身执行"（怎么协调关节力矩来跟踪 MPC 发出的指令）**——两者分层配合，组成当前人形机器人 locomotion 最主流的 `📅unknown` `[wiki_page]`
- [Optimal Control (OCP)](concepts/optimal-control.md) — 最优控制**：给定一个动力学系统和一个代价函数，求解在有限或无限时域内使得代价最小的控制输入序列的理论框架。 `📅unknown` `[wiki_page]`
- [Reward Design](concepts/reward-design.md) — 奖励函数设计（Reward Design）**：强化学习中定义智能体优化目标的核心环节。奖励函数的好坏直接决定策略能不能学出来、学出来后的行为是否符合预期。 `📅unknown` `[wiki_page]`
- [Sim2Real](concepts/sim2real.md) — Sim2Real**（仿真到现实迁移）：在仿真环境训练控制策略，然后部署到真实机器人上。 `📅unknown` `[wiki_page]`
- [State Estimation](concepts/state-estimation.md) — State Estimation（状态估计）**：根据传感器观测、机器人模型和历史信息，估计机器人当前最可能真实状态的过程。 `📅unknown` `[wiki_page]`
- [System Identification](concepts/system-identification.md) — System Identification（系统辨识 / SysID）**：通过实验数据估计机器人动力学、执行器、摩擦、延迟等模型参数，使模型更接近真实系统的过程。 `📅unknown` `[wiki_page]`
- [TSID](concepts/tsid.md) — TSID（Task Space Inverse Dynamics，任务空间逆动力学）** 是一种典型的人形机器人全身控制方法，用来在满足动力学与接触约束的前提下，把任务空间目标转成可执行的关节加速 `📅unknown` `[wiki_page]`
- [Whole-Body Control (WBC)](concepts/whole-body-control.md) — 全身控制**：对人形机器人等复杂系统，同时协调多个肢体/关节完成全身任务的控制方法。 `📅unknown` `[wiki_page]`

### Wiki Methods（方法页）

- [Diffusion Policy](methods/diffusion-policy.md) — Diffusion Policy**：将扩散生成模型（Diffusion Model）用于机器人模仿学习，通过逆扩散过程从噪声中生成动作序列的策略学习方法。 `📅unknown` `[method_page]`
- [Imitation Learning (IL)](methods/imitation-learning.md) — 模仿学习**：通过专家演示数据，让机器人学会从状态到动作的映射，核心是“抄”。 `📅unknown` `[method_page]`
- [Model Predictive Control (MPC)](methods/model-predictive-control.md) — 模型预测控制**：一种基于滚动时域优化的控制方法，在每个时刻求解一个有限时域的最优控制问题，只执行第一步，然后重复。 `📅unknown` `[method_page]`
- [Policy Optimization](methods/policy-optimization.md) — 策略优化**：通过直接对策略参数做梯度上升或近似优化，使期望累积奖励最大化的一类强化学习方法。 `📅unknown` `[method_page]`
- [Reinforcement Learning (RL)](methods/reinforcement-learning.md) — 强化学习**：通过与环境交互，以最大化累积 reward 为目标学习决策策略的机器学习范式。 `📅unknown` `[method_page]`
- [Trajectory Optimization](methods/trajectory-optimization.md) — Trajectory Optimization（轨迹优化）**：把机器人“从哪里出发、怎么运动、最终到哪里去”写成一个带动力学和约束的优化问题，求一条满足目标且代价尽量小的轨迹。 `📅unknown` `[method_page]`

### Wiki Tasks（任务页）

- [Loco-Manipulation](tasks/loco-manipulation.md) — 移动操作（Loco-Manipulation）**：机器人在运动（行走/移动）的同时执行操作任务（抓取/推动/交互），要求同时具备行走能力和上肢操作能力。 `📅unknown` `[task_page]`
- [Locomotion](tasks/locomotion.md) — 运动/行走**：让机器人（尤其人形/足式）实现稳定、高效、多地形移动的能力。 `📅unknown` `[task_page]`
- [Manipulation](tasks/manipulation.md) — 操作**：让机器人的手/末端执行器抓取、移动、操作物体。 `📅unknown` `[task_page]`
- [ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation](tasks/ultra-survey.md) — 统一多模态控制：实现人形机器人自主全身移动操作 `📅unknown` `[task_page]`

### Wiki Formalizations（形式化基础）

- [Bellman 方程](formalizations/bellman-equation.md) — Bellman 方程**：值函数的递归关系，揭示了"未来奖励"与"当前决策"之间的数学联系，是几乎所有强化学习算法的理论基础。 `📅unknown` `[formalization_page]`
- [LQR / iLQR](formalizations/lqr.md) — LQR（Linear Quadratic Regulator，线性二次调节器）**：最优控制中最经典的解析解，针对线性系统 + 二次代价函数，给出最优状态反馈增益的闭式解。**iLQR（itera `📅unknown` `[formalization_page]`
- [Markov Decision Process (MDP)](formalizations/mdp.md) — 马尔可夫决策过程**：在离散时间步中，智能体根据当前状态选择动作，环境根据转移概率回应新状态和奖励的数学框架，是强化学习的理论基础。 `📅unknown` `[formalization_page]`

### Wiki Comparisons（对比页）

- [WBC vs RL: Whole-Body Control vs Reinforcement Learning](comparisons/wbc-vs-rl.md) — 人形机器人运动控制领域最常见的两种路线对比。 `📅unknown` `[comparison_page]`

### Wiki Overview（总览）

- [Robot Learning Overview](overview/robot-learning-overview.md) — 机器人学习**：让机器人通过数据学会完成复杂任务的方法集合，核心是把“如何做”从人工编程转向从经验中学习。 `📅unknown` `[overview_page]`

### Roadmaps（路线页）

- [成长路线总览](README.md) — 本目录用于承载 `Robotics_Notebooks` 的成长路线设计。 `📅unknown` `[wiki_page]`
- [学习路径：如果目标是全栈通用能力](learning-paths/if-goal-generalist.md) — 推荐顺序： `📅unknown` `[roadmap_page]`
- [学习路径：如果目标是模仿学习与技能迁移](learning-paths/if-goal-imitation-learning.md) — 这条路径怎么用： `📅unknown` `[roadmap_page]`
- [学习路径：如果目标是人形 RL 运动控制](learning-paths/if-goal-locomotion-rl.md) — 这条路径怎么用： `📅unknown` `[roadmap_page]`
- [学习路径：如果目标是全身控制与优化](learning-paths/if-goal-whole-body-control.md) — 推荐顺序： `📅unknown` `[roadmap_page]`
- [路线A：运动控制算法工程师成长路线](route-a-motion-control.md) — 这条路线怎么用： `📅unknown` `[wiki_page]`
- [路线B：机器人全栈工程师扩展路线](route-b-fullstack.md) — 1. 感知与视觉 `📅unknown` `[wiki_page]`

### Tech-map Nodes（技术栈节点）

- [技术栈地图总览](README.md) — 本目录用于承载 `Robotics_Notebooks` 的技术栈地图、模块依赖关系、标准化模块卡片，以及研究方向导航。 `📅unknown` `[wiki_page]`
- [模块依赖关系图](dependency-graph.md) — 本页的目标不是做花哨图，而是先把 `Robotics_Notebooks` 当前最重要的依赖关系讲清楚。 `📅unknown` `[wiki_page]`
- [Humanoid Locomotion](modules/control/humanoid-locomotion.md) — 人形双足步行、平衡与扰动恢复是当前主攻方向之一。 `📅unknown` `[wiki_page]`
- [MPC](modules/control/mpc.md) — 模型预测控制是连接模型、约束与优化求解的重要方法。 `📅unknown` `[wiki_page]`
- [Whole-Body Control](modules/control/whole-body-control.md) — 全身控制是人形机器人运动控制的重要枢纽。 `📅unknown` `[wiki_page]`
- [Behavior Cloning](modules/il/behavior-cloning.md) — 模仿学习最基础的切入口。 `📅unknown` `[wiki_page]`
- [Diffusion Policy](modules/il/diffusion-policy.md) — 当前模仿学习中的重要生成式方法之一。 `📅unknown` `[wiki_page]`
- [Motion Retarget](modules/il/motion-retarget.md) — 连接人体动作数据与人形机器人技能迁移的关键模块。 `📅unknown` `[wiki_page]`
- [线性代数](modules/math/linear-algebra.md) — 机器人场景重点：向量空间、矩阵变换、特征值分解、最小二乘、雅可比相关计算。 `📅unknown` `[wiki_page]`
- [模块模板](modules/module-template.md) — - `📅unknown` `[wiki_page]`
- [Humanoid RL](modules/rl/humanoid-rl.md) — 聚焦人形机器人 locomotion 与 skill learning 中的强化学习问题。 `📅unknown` `[wiki_page]`
- [PPO](modules/rl/ppo.md) — PPO 是机器人强化学习中最常见、最实用的基础算法之一。 `📅unknown` `[wiki_page]`
- [Sim2Real](modules/rl/sim2real.md) — 重点关注域随机化、系统辨识、鲁棒训练与部署闭环。 `📅unknown` `[wiki_page]`
- [动力学](modules/robotics/dynamics.md) — 重点包括牛顿欧拉法、拉格朗日法、浮动基动力学、接触建模。 `📅unknown` `[wiki_page]`
- [运动学](modules/robotics/kinematics.md) — 重点包括正逆运动学、雅可比、微分运动学。 `📅unknown` `[wiki_page]`
- [刚体运动](modules/robotics/rigid-body-motion.md) — 重点包括旋转表示、坐标变换、SE(3)、Twist/Wrench。 `📅unknown` `[wiki_page]`
- [Deployment](modules/system/deployment.md) — 部署阶段关注控制频率、硬件接口、安全性与调试闭环。 `📅unknown` `[wiki_page]`
- [ROS2](modules/system/ros2.md) — 机器人系统工程和集成阶段的重要基础设施。 `📅unknown` `[wiki_page]`
- [Simulation](modules/system/simulation.md) — 仿真是控制、学习与部署之间的中介层。 `📅unknown` `[wiki_page]`
- [全栈技术域总览](overview.md) — 本页不是单纯列方向，而是 `Robotics_Notebooks` 的技术栈模块入口页。 `📅unknown` `[wiki_page]`
- [研究方向导航](research-directions/README.md) — 后续将按“问题驱动”组织研究方向，而不是按学科简单罗列。 `📅unknown` `[wiki_page]`

### References（参考资料页）

- [参考导航 / References](references/README.md) — 这里不是原始资料堆，也不是知识页正文。 `📅unknown` `[reference_page]`
- [Benchmark 索引 / Benchmarks](references/benchmarks/README.md) — 这里用于整理 locomotion、humanoid、learning control 等方向最常见的 benchmark 与环境。 `📅unknown` `[reference_page]`
- [Humanoid Environments](references/benchmarks/humanoid-environments.md) — 用于整理人形机器人常用训练环境与评测场景。 `📅unknown` `[reference_page]`
- [Locomotion Benchmarks](references/benchmarks/locomotion-benchmarks.md) — 用于整理平地行走、越障、跑酷、复杂地形运动等 benchmark。 `📅unknown` `[reference_page]`
- [论文导航 / Papers](references/papers/README.md) — 这里不是论文全文仓库，也不是逐篇精读笔记区。 `📅unknown` `[reference_page]`
- [Humanoid Hardware](references/papers/humanoid-hardware.md) — 聚焦人形机器人硬件架构、执行器、传感器与系统设计相关论文。 `📅unknown` `[reference_page]`
- [Imitation Learning](references/papers/imitation-learning.md) — 聚焦行为克隆、DAgger、Diffusion Policy、Motion Prior、Motion Retarget 等方向。 `📅unknown` `[reference_page]`
- [Locomotion RL](references/papers/locomotion-rl.md) — 聚焦人形/腿足机器人 locomotion 中的强化学习论文。 `📅unknown` `[reference_page]`
- [Sim2Real](references/papers/sim2real.md) — 聚焦域随机化、系统辨识、鲁棒训练、部署经验与真实机器人迁移。 `📅unknown` `[reference_page]`
- [Survey Papers](references/papers/survey-papers.md) — 用于汇总机器人学习、运动控制、人形机器人、模仿学习等方向的综述论文。 `📅unknown` `[reference_page]`
- [Whole-Body Control](references/papers/whole-body-control.md) — 聚焦任务空间控制、TSID、QP-WBC、人形全身运动控制相关论文。 `📅unknown` `[reference_page]`
- [开源生态 / Repos](references/repos/README.md) — 这里不是代码仓库镜像，而是开源项目与工具链的导航层。 `📅unknown` `[reference_page]`
- [Humanoid Projects](references/repos/humanoid-projects.md) — 聚焦人形机器人运动控制、模仿学习、感知与部署相关开源项目。 `📅unknown` `[reference_page]`
- [Retarget Tools](references/repos/retarget-tools.md) — 聚焦人体动作到机器人动作的重定向工具与项目。 `📅unknown` `[reference_page]`
- [RL Frameworks](references/repos/rl-frameworks.md) — 人形/腿足机器人 RL 训练常用开源框架。 `📅unknown` `[reference_page]`
- [Simulation](references/repos/simulation.md) — 当前重点平台： `📅unknown` `[reference_page]`
- [Utilities](references/repos/utilities.md) — 收录 Pinocchio、RBDL、Drake、curobo 等通用工具链。 `📅unknown` `[reference_page]`
