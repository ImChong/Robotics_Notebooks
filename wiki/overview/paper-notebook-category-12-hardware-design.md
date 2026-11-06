---
type: overview
tags: [humanoid-paper-notebooks, paper-index, overview]
status: complete
updated: 2026-09-15
related:
  - ./humanoid-paper-notebooks-index.md
summary: "Paper Notebooks 分类 12：硬件设计（39 篇深读笔记索引）。"
original:
  - ../../sources/repos/robot_learning_paper_notebooks_readme_reading_cat12_hw.md
---

# Paper Notebooks · Hardware Design

**Hardware Design**（`12_Hardware_Design`）是 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 主页面的第 **12** 类。

## 一句话定义

分类 12 汇总人形 / 灵巧手等**硬件设计**方向的论文笔记索引：执行器、手、腿踝、整机平台与身体-控制协同设计。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 学习型控制最终要落到真机：执行器带宽与透明度、关节结构、手的自由度直接决定策略能做什么、sim2real 差距多大。
- 上游 README 把硬件放在横跨整条路线的工程层（如 Berkeley Humanoid 作为 LCP 的部署平台、ToddlerBot 提供校准与数字孪生工具链）。

## 核心原理

本类条目大致可按主题分组（据上游站点分类 12 标题归纳，分组为推断）：

- **整机平台**：Berkeley Humanoid、ToddlerBot、Fauna Sprout、Duke Humanoid、iRonCub 3 等。
- **执行器**：Proprioceptive actuator（MIT Cheetah）、准直驱（QDD）、Human-Level Actuation、Bowden 腱驱。
- **灵巧手 / 腕**：连杆驱动、SMA-电机混合、ORCA、RAPID、RUKA、ByteWrist、DexWrist。
- **腿 / 踝与肌骨**：DecARt Leg、最优踝设计、肌骨人形。
- **协同设计**：Brain-Body Co-design、Embracing Evolution、Chin-Up 协同设计（RL 与形态联合优化）。

## 工程实践

- 读单篇笔记时先看其硬件指标（自由度、扭矩 / 速度、成本、是否开源），再看是否配套仿真模型与训练代码。
- 优先阅读带「深读笔记」链接的条目；标「待深读」的仅有 wiki 实体页，内容较浅。

## 局限

- 本页只是索引，不含各论文的定量对比；站点首页分类 12 为 21 条，本库为 39 条（含待深读与跨类条目），引用数字须注明口径。
- 多数条目尚为「待深读」。

## 本类论文索引

- [A 21-DOF Humanoid Dexterous Hand with Hybrid SMA-Motor Actuation](../entities/paper-notebook-a-21-dof-humanoid-dexterous-hand-with-hybrid-sma.md) — 待深读
- [A Framework for Optimal Ankle Design of Humanoid Robots](../concepts/humanoid-parallel-joint-kinematics.md) — 见 wiki 实体页
- [AGILOped](../entities/paper-notebook-agiloped-agile-open-source-humanoid-robot-for-re.md) — 待深读
- [Aero Hand Open](../entities/paper-aero-hand-open.md) — 腱驱手 MuJoCo sim2real（[索引页](../entities/paper-aero-hand-open.md)）
- [Antagonistic Bowden-Cable Actuation of a Lightweight Robotic Hand](../entities/paper-notebook-antagonistic-bowden-cable-actuation-of-a-lightwe.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/Antagonistic_Bowden-Cable_Actuation_of_a_Lightweight_Robotic_Hand/Antagonistic_Bowden-Cable_Actuation_of_a_Lightweight_Robotic_Hand.html)
- [Berkeley Humanoid Lite](../entities/berkeley-humanoid-lite.md) — 待深读
- [Berkeley Humanoid](../entities/paper-notebook-berkeley-humanoid-a-research-platform-for-learni.md) — 待深读
- [ByteWrist](../entities/paper-notebook-bytewrist-a-parallel-robotic-wrist-enabling-flex.md) — 待深读
- [CAD-Driven Co-Design for Flight-Ready Jet-Powered Humanoids](../entities/paper-notebook-cad-driven-co-design-for-flight-ready-jet-powere.md) — 待深读
- [Characteristics, Management, and Utilization of Muscles in Musculoskeletal Humanoids](../entities/paper-notebook-characteristics-management-and-utilization-of-mu.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/Characteristics_Management_and_Utilization_of_Muscles_in_Musculoskeletal_Humanoids/Characteristics_Management_and_Utilization_of_Muscles_in_Musculoskeletal_Humanoids.html)
- [DIJIT](../entities/paper-notebook-dijit-a-robotic-head-for-an-active-observer.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/DIJIT_A_Robotic_Head_for_an_Active_Observer/DIJIT_A_Robotic_Head_for_an_Active_Observer.html)
- [DecARt Leg](../entities/paper-notebook-decart-leg-design-and-evaluation-of-a-novel-huma.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/DecARt_Leg_Novel_Humanoid_Robot_Leg_with_Decoupled_Actuation/DecARt_Leg_Novel_Humanoid_Robot_Leg_with_Decoupled_Actuation.html)
- [Design and Control of a Bipedal Robotic Character](../entities/paper-notebook-design-and-control-of-a-bipedal-robotic-characte.md) — 待深读
- [DexWrist](../entities/paper-notebook-dexwrist-a-robotic-wrist-for-constrained-and-dyn.md) — 待深读
- [Dexterous Teleoperation of 20-DoF ByteDexter Hand via Human Motion Retargeting](../entities/paper-notebook-dexterous-teleoperation-of-20-dof-bytedexter-han.md) — 待深读
- [Embracing Evolution](../entities/paper-notebook-embracing-evolution-a-call-for-body-control-co-d.md) — 待深读
- [Evolutionary Continuous Adaptive RL-Powered Co-Design for Humanoid Chin-Up Performance](../entities/paper-notebook-evolutionary-continuous-adaptive-rl-powered-co-d.md) — 待深读
- [Exceeding the Maximum Speed Limit of the Joint Angle for the Redundant Tendon-driven Structures of Musculoskeletal Humanoids](../entities/paper-notebook-exceeding-the-maximum-speed-limit-of-the-joint-a.md) — 待深读
- [Explosive Output to Enhance Jumping Ability](../entities/paper-notebook-explosive-output-to-enhance-jumping-ability-a-va.md) — 待深读
- [Fauna Sprout](../entities/paper-notebook-fauna-sprout-a-lightweight-approachable-develope.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/Fauna_Sprout_A_lightweight_approachable_developer-ready_humanoid_robot/Fauna_Sprout_A_lightweight_approachable_developer-ready_humanoid_robot.html)
- [Human-Level Actuation for Humanoids](../entities/paper-notebook-human-level-actuation-for-humanoids.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/Human-Level_Actuation_for_Humanoids/Human-Level_Actuation_for_Humanoids.html)
- [Integrated linkage-driven dexterous anthropomorphic robotic hand](../entities/paper-notebook-integrated-linkage-driven-dexterous-anthropomorp.md) — 待深读
- [Micro-Wheeled_leg-Robot](../entities/paper-notebook-micro-wheeled-leg-robot.md) — 待深读
- [ORCA](../entities/paper-notebook-orca-open-source-reliable-cost-effective-anthrop.md) — 待深读
- [OSMO](../entities/paper-notebook-osmo-open-source-tactile-glove-for-human-to-robo.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/OSMO_Open-Source_Tactile_Glove_for_Human-to-Robot_Skill_Transfer/OSMO_Open-Source_Tactile_Glove_for_Human-to-Robot_Skill_Transfer.html)
- [Olaf](../methods/disney-olaf-character-robot.md) — 见 wiki 实体页
- [PIMBS](../entities/paper-notebook-pimbs-efficient-body-schema-learning-for-musculo.md) — 待深读
- [Proprioceptive actuator design in the MIT Cheetah](../entities/paper-notebook-proprioceptive-actuator-design-in-the-mit-cheeta.md) — 待深读
- [Quasi-Direct Drive for Low-Cost Compliant Robotic Manipulation](../entities/paper-notebook-quasi-direct-drive-for-low-cost-compliant-roboti.md) — 待深读
- [RAPID Hand](../entities/paper-notebook-rapid-hand-a-robust-affordable-perception-integr.md) — 待深读
- [RUKA](../entities/paper-notebook-ruka-rethinking-the-design-of-humanoid-hands-wit.md) — 待深读
- [The Duke Humanoid](../entities/paper-notebook-the-duke-humanoid-design-and-control-for-energy.md) — 待深读
- [The MIT Humanoid Robot](../entities/paper-notebook-the-mit-humanoid-robot-design-motion-planning-an.md) — 待深读
- [ToddlerBot](../entities/paper-loco-manip-161-141-toddlerbot.md) — 见 wiki 实体页
- [Toward Humanoid Brain-Body Co-design](../entities/paper-notebook-toward-humanoid-brain-body-co-design-joint-optim.md) — 待深读
- [Unitree H1 Humanoid Robot Whitepaper & Specifications](../entities/paper-notebook-unitree-h1-whitepaper.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/Unitree_H1_Whitepaper/Unitree_H1_Whitepaper.html)
- [X2-N](../tasks/hybrid-locomotion.md) — 见 wiki 实体页
- [Zeroth Bot](../entities/paper-notebook-zeroth-bot-https-github-com-zeroth-robotics-zero.md) — 待深读
- [iRonCub 3](../entities/paper-notebook-ironcub-3-the-jet-powered-flying-humanoid-robot.md) — 待深读

## 关联页面

- 本类在站点上的父级与同级关系见下。

## 与其他页面的关系

- 总索引：[humanoid-paper-notebooks-index.md](./humanoid-paper-notebooks-index.md)
- 笔记主页：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html>

## 参考来源

- [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)
- 站点来源归档：[humanoid-robot-learning-paper-notebooks](../../sources/sites/humanoid-robot-learning-paper-notebooks.md)
- 本类论文来源笔记（节选）：[ToddlerBot](../../sources/papers/loco_manip_161_survey_141_toddlerbot.md)、[CYJ Hand-0：21 自由度 SMA-电机混合驱动灵巧手](../../sources/papers/humanoid_pnb_a-21-dof-humanoid-dexterous-hand-with-hybrid-sma.md)、[Parallel_Ankle_Joint](../../sources/repos/parallel_ankle_joint.md)

## 推荐继续阅读

- 上游 README 学习路线图的工程层可作为硬件主题的阅读顺序参考。

- [机器人论文阅读笔记总站](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)
