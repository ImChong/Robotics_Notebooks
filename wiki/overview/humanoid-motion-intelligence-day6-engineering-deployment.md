---
type: overview
tags: [overview, humanoid, sim2real, deployment]
status: complete
updated: 2026-10-07
related:
  - ./humanoid-motion-intelligence-day5-world-models-decision.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/blogs/humanoid_motion_intelligence_day6_engineering_deployment_2026_10_07.md
summary: "Day 6 导读：整理原文 29 项工作，逐项链接其独立详情节点。"
---

# 具身智能从入门到精通 Day 6：工程与实机部署

> **文章节点**：Yuanxq（具身智能研究室）原文的站内导读。论文和项目各自链接到独立详情。

## 一句话观点

工程部署把状态估计、仿真覆盖、推理时序、硬件差异和安全约束连成一个闭环；单项算法的提升必须在真机接口中验证。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| EKF | Extended Kalman Filter | 融合传感器与运动模型估计系统状态。 |
| WBC | Whole-Body Control | 协调机器人的全身自由度和任务约束。 |
| Sim2Real | Simulation-to-Real | 从仿真训练或验证迁移至实体机器人。 |
| RL | Reinforcement Learning | 使用奖励和交互数据优化策略行为。 |

## 方法关系

```mermaid
flowchart LR
    A["传感器、数据与仿真"] --> B["状态估计与系统标定"]
    B --> C["策略训练、评测与安全约束"]
    C --> D["机器人执行"]
    D --> E["新观测、日志和反馈"]
    E --> A
```

接触估计影响支撑判断，模型参数影响控制和迁移，推理延迟影响目标是否仍有效，安全过滤器受扰动和跟踪误差约束。

## 论文与项目独立详情

| 工作 | 独立详情 | 作用 |
|---|---|---|
| Contact-Aided Invariant Extended Kalman Filtering for Legged Robot State Estimation | [独立详情](../entities/paper-notebook-contact-aided-invariant-ekf-for-legged-robots.md) | contact-aided state estimation |
| Operational Space Formulation | [独立详情](../entities/paper-operational-space-formulation.md) | unified robot motion and force control |
| Whole-Body Behaviors through Hierarchical Control of Behavioral Primitives | [独立详情](../entities/paper-whole-body-behaviors-primitives.md) | hierarchical behavioral primitives |
| The Stack of Tasks | [独立详情](../entities/paper-hmi-stack-of-tasks.md) | prioritized inverse kinematics |
| Momentum Control with Hierarchical Inverse Dynamics | [独立详情](../entities/paper-momentum-control-hierarchical-id.md) | humanoid momentum and inverse dynamics |
| Optimization-based Atlas Locomotion Planning and Control | [独立详情](../entities/paper-atlas-locomotion-optimization-stack.md) | Atlas planning, estimation and control |
| Crocoddyl | [独立详情](../entities/crocoddyl.md) | multi-contact optimal control framework |
| Berkeley Humanoid | [独立详情](../entities/paper-notebook-berkeley-humanoid-a-research-platform-for-learni.md) | learning-based humanoid platform |
| ASAP | [独立详情](../entities/paper-notebook-asap-aligning-simulation-and-real-world-physics.md) | sim-to-real physics alignment |
| ToddlerBot | [独立详情](../entities/paper-loco-manip-161-141-toddlerbot.md) | open-source learning-compatible humanoid |
| FiatLux | [独立详情](../entities/paper-fiatlux.md) | long-horizon humanoid benchmark |
| Bundled Contact Gradients | [独立详情](../entities/paper-bundled-contact-gradients.md) | stable gradients at contact events |
| Online Sim-to-Real Adaptation via Closed-Loop System Modeling | [独立详情](../entities/paper-online-sim2real-closed-loop-modeling.md) | closed-loop online system modeling |
| CoPRE | [独立详情](../entities/paper-copre-proprioceptive-contact.md) | proprioceptive contact detection |
| TAPESIM | [独立详情](../entities/paper-tapesim.md) | efficient adhesive-tape simulation |
| X2Real 论文 | [独立详情](../entities/paper-x2real.md) | sim-to-real generalist-policy benchmark |
| X2Real 项目 | [独立详情](../entities/x2real-project.md) | benchmark architecture, assets and reproducibility status |
| MotionForge | [独立详情](../entities/paper-motionforge.md) | dynamic-object task and data generation |
| The Cartesian Hand | [独立详情](../entities/paper-cartesian-hand-linear-fingers.md) | in-hand manipulation with linear fingers |
| H2RBench | [独立详情](../entities/paper-h2rbench.md) | human-to-robot transfer benchmark |
| PRIMO | [独立详情](../entities/paper-primo-human-motion-odometry.md) | human-motion prior for odometry |
| SmoothRL | [独立详情](../entities/paper-smoothrl.md) | online RL during asynchronous execution |
| HumanoidVLN | [独立详情](../entities/paper-humanoidvln.md) | physics-grounded humanoid navigation |
| EmbodiedGen V2 | [独立详情](../entities/paper-embodiedgen-v2-sim-ready-world-engine.md) | simulation-ready 3D world generation |
| SIMPLE | [独立详情](../entities/paper-loco-manip-161-075-simple.md) | simulation-based humanoid loco-manipulation |
| UniLab | [独立详情](../entities/unilab.md) | heterogeneous asynchronous RL architecture |
| Safety-Critical Whole-Body Control | [独立详情](../entities/paper-motion-cerebellum-safewbc.md) | safe control barrier functions |
| PRIME | [独立详情](../entities/prime-system-id.md) | physically consistent inertial and motion estimation |
| Humanoid Everyday | [独立详情](../entities/humanoid-everyday-dataset.md) | open-world manipulation dataset |
| Systematic Sim-to-Real Transfer for Diverse Legged Robots | [独立详情](../entities/paper-pace-sim2real-legged-robots.md) | hardware-aware legged sim-to-real |

## 评读边界

记录机器人型号、传感器、控制周期、推理硬件、进程边界、安全机制和复位条件。吞吐、仿真—真机相关性与真机成功率是不同证据。

## 结论

把方法的输入输出和时间接口写清，再验证感知、估计、规划、控制和安全约束能否共同闭环。

## 关联页面

- [Day 5：世界模型与决策](./humanoid-motion-intelligence-day5-world-models-decision.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [Day 6 原文与索引归档](../../sources/blogs/humanoid_motion_intelligence_day6_engineering_deployment_2026_10_07.md)
- [humanoid-motion-intelligence 项目](https://github.com/RealXiaoze/humanoid-motion-intelligence)

## 推荐继续阅读

- [ASAP 项目页](https://agile.human2humanoid.com/)
