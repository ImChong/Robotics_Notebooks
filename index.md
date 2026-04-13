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

## 与其他项目的边界

- [`Humanoid_Robot_Learning_Paper_Notebooks`](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks)：单篇论文深读
- [`Robotics_Notebooks`](https://github.com/ImChong/Robotics_Notebooks)：跨模块知识组织
- [`ImChong.github.io`](https://github.com/ImChong/ImChong.github.io)：个人简历

> 论文项目负责点，技术栈项目负责线和面，个人主页负责展示。

---

## Page Catalog（页面目录）

> 以下是本知识库全部页面的索引，按 type 分组。
> 每个页面含一句话摘要、主要 tags、首次创建日期（📅）和被引用数（⬆️），来自 `exports/site-data-v1.json` + git log。
> 本目录由脚本自动生成，不手动编辑。

### Entities（实体页）

- [Crocoddyl](wiki/entities/crocoddyl.md) — 如果说 Pinocchio 提供的是机器人运动学、动力学和导数的高质量计算底座，那 Crocoddyl 提供的就是。... `📅2026-04-11` `⬆️1` `[entity_page]`  _entities, entity, humanoid, locomotion_
- [Isaac Gym / Isaac Lab](wiki/entities/isaac-gym-isaac-lab.md) — Isaac Gym：NVIDIA 早期的 GPU 加速机器人 RL 仿真框架，主打大规模并行训练... `📅2026-04-11` `⬆️2` `[entity_page]`  _entities, entity, locomotion, rl_
- [legged_gym](wiki/entities/legged-gym.md) — 如果说 Isaac Gym 提供的是高并行 GPU 仿真底座，那 legged_gym 提供的就是。... `📅2026-04-11` `⬆️0` `[entity_page]`  _entities, entity, humanoid, locomotion_
- [MuJoCo](wiki/entities/mujoco.md) — MuJoCo 是一个强调刚体动力学、接触仿真、控制与优化友好性的机器人物理引擎... `📅2026-04-11` `⬆️0` `[entity_page]`  _entities, entity, humanoid, locomotion_
- [Pinocchio](wiki/entities/pinocchio.md) — 如果说 MuJoCo、Isaac Gym 这类工具是在做"仿真"，那 Pinocchio 做的是。... `📅2026-04-11` `⬆️1` `[entity_page]`  _entities, entity, humanoid, control_
- [Unitree](wiki/entities/unitree.md) — 如果说很多论文和算法都在讲"机器人应该怎么走、怎么跑、怎么控制"，那 Unitree 的重要性在于。... `📅2026-04-11` `⬆️0` `[entity_page]`  _entities, entity, humanoid, locomotion_

### References（参考资料页）

- [Humanoid Hardware](references/papers/humanoid-hardware.md) — 聚焦人形机器人硬件架构、执行器、传感器与系统设计相关论文... `📅2026-04-08` `⬆️0` `[reference_page]`  _papers, humanoid, hardware_
- [Imitation Learning](references/papers/imitation-learning.md) — 聚焦行为克隆、DAgger、Diffusion Policy、Motion Prior、Motion Retarget 等方向... `📅2026-04-08` `⬆️0` `[reference_page]`  _papers, il_
- [Locomotion RL](references/papers/locomotion-rl.md) — 聚焦人形/腿足机器人 locomotion 中的强化学习论文... `📅2026-04-08` `⬆️2` `[reference_page]`  _papers, locomotion, rl_
- [Sim2Real](references/papers/sim2real.md) — 聚焦域随机化、系统辨识、鲁棒训练、部署经验与真实机器人迁移... `📅2026-04-08` `⬆️2` `[reference_page]`  _papers, sim2real_
- [Survey Papers](references/papers/survey-papers.md) — 用于汇总机器人学习、运动控制、人形机器人、模仿学习等方向的综述论文... `📅2026-04-08` `⬆️4` `[reference_page]`  _papers_
- [Whole-Body Control](references/papers/whole-body-control.md) — 聚焦任务空间控制、TSID、QP-WBC、人形全身运动控制相关论文... `📅2026-04-08` `⬆️4` `[reference_page]`  _papers, control, optimization_
- [Humanoid Projects](references/repos/humanoid-projects.md) — 聚焦人形机器人运动控制、模仿学习、感知与部署相关开源项目... `📅2026-04-08` `⬆️3` `[reference_page]`  _repos, humanoid_
- [Retarget Tools](references/repos/retarget-tools.md) — 聚焦人体动作到机器人动作的重定向工具与项目... `📅2026-04-08` `⬆️0` `[reference_page]`  _repos, il_
- [RL Frameworks](references/repos/rl-frameworks.md) — 当前重点框架。... `📅2026-04-08` `⬆️4` `[reference_page]`  _repos, humanoid, locomotion_
- [Simulation](references/repos/simulation.md) — 当前重点平台。... `📅2026-04-08` `⬆️6` `[reference_page]`  _repos, tooling_
- [Utilities](references/repos/utilities.md) — 收录 Pinocchio、RBDL、Drake、curobo 等通用工具链... `📅2026-04-08` `⬆️4` `[reference_page]`  _repos, tooling_
- [Humanoid Environments](references/benchmarks/humanoid-environments.md) — 用于整理人形机器人常用训练环境与评测场景... `📅2026-04-08` `⬆️0` `[reference_page]`  _benchmarks, humanoid_
- [Locomotion Benchmarks](references/benchmarks/locomotion-benchmarks.md) — 用于整理平地行走、越障、跑酷、复杂地形运动等 benchmark... `📅2026-04-08` `⬆️2` `[reference_page]`  _benchmarks, locomotion_

### Roadmaps（路线页）

- [路线A：运动控制算法工程师成长路线](roadmap/route-a-motion-control.md) — 这条路线怎么用。... `📅2026-04-08` `⬆️2` `[roadmap_page]`  _roadmap, control, dynamics_
- [路线B：机器人全栈工程师扩展路线](roadmap/route-b-fullstack.md) — 目标：在路线A基础上，从"控制切入口"逐步扩展到感知、规划、软件系统、部署与整机集成... `📅2026-04-08` `⬆️0` `[roadmap_page]`  _roadmap_
- [学习路径：如果目标是全栈通用能力](roadmap/learning-paths/if-goal-generalist.md) — 推荐顺序。... `📅2026-04-08` `⬆️0` `[roadmap_page]`  _roadmap_
- [学习路径：如果目标是模仿学习与技能迁移](roadmap/learning-paths/if-goal-imitation-learning.md) — 这条路径怎么用。... `📅2026-04-08` `⬆️0` `[roadmap_page]`  _roadmap, rl, il_
- [学习路径：如果目标是人形 RL 运动控制](roadmap/learning-paths/if-goal-locomotion-rl.md) — 这条路径怎么用。... `📅2026-04-08` `⬆️1` `[roadmap_page]`  _roadmap, humanoid, locomotion_
- [学习路径：如果目标是全身控制与优化](roadmap/learning-paths/if-goal-whole-body-control.md) — 推荐顺序。... `📅2026-04-08` `⬆️0` `[roadmap_page]`  _roadmap, control, optimization_

### Tech-map Nodes（技术栈节点）

- [全栈技术域总览](tech-map/overview.md) — 本页不是单纯列方向，而是 `Robotics_Notebooks` 的技术栈模块入口页... `📅2026-04-08` `⬆️0` `[tech_map_node]`  _tech-map, humanoid, locomotion_
- [模块依赖关系图](tech-map/dependency-graph.md) — 本页的目标不是做花哨图，而是先把 `Robotics_Notebooks` 当前最重要的依赖关系讲清楚... `📅2026-04-08` `⬆️2` `[tech_map_node]`  _tech-map, control, dynamics_
- [Humanoid Locomotion](tech-map/modules/control/humanoid-locomotion.md) — 人形双足步行、平衡与扰动恢复是当前主攻方向之一... `📅2026-04-08` `⬆️0` `[tech_map_node]`  _tech-map, humanoid, locomotion_
- [MPC](tech-map/modules/control/mpc.md) — 模型预测控制是连接模型、约束与优化求解的重要方法... `📅2026-04-08` `⬆️0` `[tech_map_node]`  _tech-map, control_
- [Whole-Body Control](tech-map/modules/control/whole-body-control.md) — 全身控制是人形机器人运动控制的重要枢纽... `📅2026-04-08` `⬆️0` `[tech_map_node]`  _tech-map, control_
- [Behavior Cloning](tech-map/modules/il/behavior-cloning.md) — 模仿学习最基础的切入口... `📅2026-04-08` `⬆️0` `[tech_map_node]`  _tech-map_
- [Diffusion Policy](tech-map/modules/il/diffusion-policy.md) — 当前模仿学习中的重要生成式方法之一... `📅2026-04-08` `⬆️0` `[tech_map_node]`  _tech-map_
- [Motion Retarget](tech-map/modules/il/motion-retarget.md) — 连接人体动作数据与人形机器人技能迁移的关键模块... `📅2026-04-08` `⬆️0` `[tech_map_node]`  _tech-map, il_
- [线性代数](tech-map/modules/math/linear-algebra.md) — 机器人场景重点：向量空间、矩阵变换、特征值分解、最小二乘、雅可比相关计算... `📅2026-04-08` `⬆️0` `[tech_map_node]`  _tech-map_
- [Humanoid RL](tech-map/modules/rl/humanoid-rl.md) — 聚焦人形机器人 locomotion 与 skill learning 中的强化学习问题... `📅2026-04-08` `⬆️0` `[tech_map_node]`  _tech-map, humanoid, locomotion_
- [PPO](tech-map/modules/rl/ppo.md) — PPO 是机器人强化学习中最常见、最实用的基础算法之一... `📅2026-04-08` `⬆️0` `[tech_map_node]`  _tech-map, rl_
- [Sim2Real](tech-map/modules/rl/sim2real.md) — 重点关注域随机化、系统辨识、鲁棒训练与部署闭环... `📅2026-04-08` `⬆️0` `[tech_map_node]`  _tech-map, rl, sim2real_
- [动力学](tech-map/modules/robotics/dynamics.md) — 重点包括牛顿欧拉法、拉格朗日法、浮动基动力学、接触建模... `📅2026-04-08` `⬆️0` `[tech_map_node]`  _tech-map, dynamics_
- [运动学](tech-map/modules/robotics/kinematics.md) — 重点包括正逆运动学、雅可比、微分运动学... `📅2026-04-08` `⬆️0` `[tech_map_node]`  _tech-map_
- [刚体运动](tech-map/modules/robotics/rigid-body-motion.md) — 重点包括旋转表示、坐标变换、SE(3)、Twist/Wrench... `📅2026-04-08` `⬆️0` `[tech_map_node]`  _tech-map_
- [Deployment](tech-map/modules/system/deployment.md) — 部署阶段关注控制频率、硬件接口、安全性与调试闭环... `📅2026-04-08` `⬆️0` `[tech_map_node]`  _tech-map_
- [ROS2](tech-map/modules/system/ros2.md) — 机器人系统工程和集成阶段的重要基础设施... `📅2026-04-08` `⬆️0` `[tech_map_node]`  _tech-map_
- [Simulation](tech-map/modules/system/simulation.md) — 仿真是控制、学习与部署之间的中介层... `📅2026-04-08` `⬆️0` `[tech_map_node]`  _tech-map, tooling_

### Formalizations（形式化基础）

- [MDP](wiki/formalizations/mdp.md) — 马尔可夫决策过程：强化学习的数学理论根基，包含状态/动作/转移/奖励/折扣五元组与最优性定义
- [Bellman 方程](wiki/formalizations/bellman-equation.md) — 值函数的递归关系：几乎所有 RL 算法的理论基础，包含 Value Iteration / Policy Iteration / TD Learning

### Wiki Pages（知识页）

- [Capture Point / DCM](wiki/concepts/capture-point-dcm.md) — Capture Point：如果机器人已经在往前倒，那么要想在有限步内停住，脚应该踩到哪里... `📅2026-04-11` `⬆️0` `[wiki_page]`  _concept, control, dynamics_
- [Centroidal Dynamics](wiki/concepts/centroidal-dynamics.md) — Centroidal Dynamics（质心动力学）：用机器人整体质心的线动量和角动量来描述全身动力学的一种中... `📅2026-04-11` `⬆️10` `[wiki_page]`  _concept, locomotion, control_
- [Contact Dynamics](wiki/concepts/contact-dynamics.md) — Contact Dynamics（接触动力学）：研究机器人与地面、物体、墙面等环境发生接触时，接触力、约束和系... `📅2026-04-11` `⬆️1` `[wiki_page]`  _concept, locomotion, control_
- [Domain Randomization](wiki/concepts/domain-randomization.md) — 域随机化：在仿真训练中主动随机化物理参数、视觉纹理、环境设置，让策略被迫学会适应各种变化的泛化能力，从而实现零... `📅2026-04-07` `⬆️6` `[wiki_page]`  _concept, locomotion, control_
- [Floating Base Dynamics](wiki/concepts/floating-base-dynamics.md) — Floating Base Dynamics（浮动基动力学）：描述机器人在基座不固定于世界坐标系时，其整体动力... `📅2026-04-11` `⬆️2` `[wiki_page]`  _concept, locomotion, control_
- [LIP / ZMP](wiki/concepts/lip-zmp.md) — LIP：用"质心高度近似不变"的简化模型描述双足机器人平衡与步行... `📅2026-04-11` `⬆️6` `[wiki_page]`  _concept, locomotion, control_
- [Optimal Control (OCP)](wiki/concepts/optimal-control.md) — 最优控制：给定一个动力学系统和一个代价函数，求解在有限或无限时域内使得代价最小的控制输入序列的理论框架... `📅2026-04-07` `⬆️8` `[wiki_page]`  _concept, control, optimization_
- [Sim2Real](wiki/concepts/sim2real.md) — Sim2Real（仿真到现实迁移）：在仿真环境训练控制策略，然后部署到真实机器人上... `📅2026-04-07` `⬆️14` `[wiki_page]`  _concept, humanoid, locomotion_
- [State Estimation](wiki/concepts/state-estimation.md) — State Estimation（状态估计）：根据传感器观测、机器人模型和历史信息，估计机器人当前最可能真实状... `📅2026-04-11` `⬆️5` `[wiki_page]`  _concept, locomotion, control_
- [System Identification](wiki/concepts/system-identification.md) — System Identification（系统辨识 / SysID）：通过实验数据估计机器人动力学、执行器、... `📅2026-04-11` `⬆️3` `[wiki_page]`  _concept, locomotion, control_
- [TSID](wiki/concepts/tsid.md) — 如果说 MPC / Centroidal Dynamics 在回答"机器人整体接下来该怎么动"，那 TSID... `📅2026-04-11` `⬆️9` `[wiki_page]`  _concept, control, dynamics_
- [Whole-Body Control (WBC)](wiki/concepts/whole-body-control.md) — 全身控制：对人形机器人等复杂系统，同时协调多个肢体/关节完成全身任务的控制方法... `📅2026-04-07` `⬆️25` `[wiki_page]`  _concept, locomotion, control_
- [Imitation Learning (IL)](wiki/methods/imitation-learning.md) — 模仿学习：通过专家演示数据，让机器人学会从状态到动作的映射，核心是"抄"... `📅2026-04-07` `⬆️10` `[wiki_page]`  _method, locomotion, control_
- [Model Predictive Control (MPC)](wiki/methods/model-predictive-control.md) — 模型预测控制：一种基于滚动时域优化的控制方法，在每个时刻求解一个有限时域的最优控制问题，只执行第一步，然后重复... `📅2026-04-07` `⬆️14` `[wiki_page]`  _method, humanoid, locomotion_
- [Reinforcement Learning (RL)](wiki/methods/reinforcement-learning.md) — 强化学习：通过与环境交互，以最大化累积 reward 为目标学习决策策略的机器学习范式... `📅2026-04-07` `⬆️17` `[wiki_page]`  _method, locomotion, control_
- [Trajectory Optimization](wiki/methods/trajectory-optimization.md) — Trajectory Optimization（轨迹优化）：把机器人"从哪里出发、怎么运动、最终到哪里去"写成... `📅2026-04-11` `⬆️6` `[wiki_page]`  _method, locomotion, control_
- [Locomotion](wiki/tasks/locomotion.md) — 运动/行走：让机器人（尤其人形/足式）实现稳定、高效、多地形移动的能力... `📅2026-04-07` `⬆️17` `[wiki_page]`  _task, locomotion, control_
- [Manipulation](wiki/tasks/manipulation.md) — 操作：让机器人的手/末端执行器抓取、移动、操作物体... `📅2026-04-07` `⬆️1` `[wiki_page]`  _task, locomotion, control_
- [WBC vs RL: Whole-Body Control vs Reinforcement Learning](wiki/comparisons/wbc-vs-rl.md) — 人形机器人运动控制领域最常见的两种路线对比... `📅2026-04-07` `⬆️4` `[wiki_page]`  _comparison, control, optimization_
- [Robot Learning Overview](wiki/overview/robot-learning-overview.md) — 机器人学习：让机器人通过数据学会完成复杂任务的方法集合，核心是把"如何做"从人工编程转向从经验中学习... `📅2026-04-07` `⬆️1` `[wiki_page]`  _overview, locomotion, control_
