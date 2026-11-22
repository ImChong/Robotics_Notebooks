---
type: overview
tags: [humanoid-paper-notebooks, paper-index, overview]
status: complete
updated: 2026-11-21
related:
  - ./humanoid-paper-notebooks-index.md
  - ../concepts/simulation-evaluation-infrastructure.md
sources:
  - ../../sources/sites/humanoid-robot-learning-paper-notebooks.md
original:
  - ../../sources/sites/humanoid-robot-learning-paper-notebooks.md
summary: "Paper Notebooks 分类 11：仿真与基准（涵盖站内既有 22 篇主索引并与上游最新 32 篇目录同步核对）。"
---

# Paper Notebooks · Simulation Benchmark

**Simulation Benchmark**（`11_Simulation_Benchmark`）是 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 主页面的第 **11** 类。

## 一句话定义

**本分类页收录《Robot Learning Paper Notebooks》第 11 专题（Simulation & Benchmark）下关于 GPU 并行仿真引擎、具身数据生成流水线与通用基准评测体系的论文深读导航。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- **数据与评测瓶颈决定策略上限：** 相比真机遥操作，GPU 并行仿真（如 Genesis、ManiSkill3、MuJoCo Playground）与自动化数据扩增框架（RoboCasa365、DexMimicGen、HumanoidGen）决定了策略预训练的吞吐和基准可复现性。
- **衔接上游持续更新：** 本站 2026-09 快照收录 22 篇主条目标注（对应下表）；截至 2026-11 核对上游 `index.html`，`11_Simulation_Benchmark` 已扩至 32 篇（新增含 Agent-World、SimGen、SimScale 等方向），可结合上游目录按需增补实体页。

## 核心机制与本类论文索引

- [Benchmarking Humanoid Imitation Learning with Motion Difficulty](../entities/paper-notebook-benchmarking-humanoid-imitation-learning-with-mo.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/Benchmarking_Humanoid_Imitation_Learning_with_Motion_Difficulty/Benchmarking_Humanoid_Imitation_Learning_with_Motion_Difficulty.html)
- [BiGym](../entities/paper-notebook-bigym-a-demo-driven-mobile-bi-manual-manipulatio.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/BiGym__A_Demo-Driven_Mobile_Bi-Manual_Manipulation_Benchmark/BiGym__A_Demo-Driven_Mobile_Bi-Manual_Manipulation_Benchmark.html)
- [ComFree-Sim](../entities/paper-notebook-comfree-sim.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/ComFree-Sim__GPU-Parallelized_Analytical_Contact_Physics_Engine/ComFree-Sim__GPU-Parallelized_Analytical_Contact_Physics_Engine.html)
- [DexMimicGen](../entities/paper-notebook-dexmimicgen-automated-data-generation-for-bimanu.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/DexMimicGen__Automated_Data_Generation_for_Bimanual_Dexterous_Manipulation/DexMimicGen__Automated_Data_Generation_for_Bimanual_Dexterous_Manipulation.html)
- [DualTHOR](../entities/paper-notebook-dualthor-a-dual-arm-humanoid-simulation-platform.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/DualTHOR__A_Dual-Arm_Humanoid_Simulation_Platform_for_Contingency-Aware_Planning/DualTHOR__A_Dual-Arm_Humanoid_Simulation_Platform_for_Contingency-Aware_Planning.html)
- [GRAIL](../entities/grail-locomanipulation-dataset.md) — 见 wiki 实体页
- [GRUtopia](../concepts/simulation-evaluation-infrastructure.md) — 见 wiki 实体页
- [Generative World Modelling for Humanoids](../methods/generative-world-models.md) — 见 wiki 实体页
- [Genesis](../entities/genesis-sim.md) — 见 wiki 实体页
- [Humanoid Everyday](../entities/humanoid-everyday-dataset.md) — 见 wiki 实体页
- [Humanoid World Models](../entities/paper-notebook-humanoid-world-models-open-world-foundation-mode.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/Humanoid_World_Models__Open_World_Foundation_Models_for_Humanoid_Robotics/Humanoid_World_Models__Open_World_Foundation_Models_for_Humanoid_Robotics.html)
- [HumanoidGen](../entities/paper-notebook-humanoidgen-data-generation-for-bimanual-dextero.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/HumanoidGen__Data_Generation_for_Bimanual_Dexterous_Manipulation_via_LLM_Reasoning/HumanoidGen__Data_Generation_for_Bimanual_Dexterous_Manipulation_via_LLM_Reasoning.html)
- [Learning with pyCub](../entities/paper-notebook-learning-with-pycub-a-simulation-and-exercise-fr.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/Learning_with_pyCub__A_Simulation_and_Exercise_Framework_for_Humanoid_Robotics/Learning_with_pyCub__A_Simulation_and_Exercise_Framework_for_Humanoid_Robotics.html)
- [ManiSkill-HAB](../entities/paper-notebook-maniskill-hab-a-benchmark-for-low-level-manipula.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/ManiSkill-HAB__A_Benchmark_for_Low-Level_Manipulation_in_Home_Rearrangement_Tasks/ManiSkill-HAB__A_Benchmark_for_Low-Level_Manipulation_in_Home_Rearrangement_Tasks.html)
- [ManiSkill3](../entities/paper-notebook-maniskill3-gpu-parallelized-robotics-simulation.md) — 待深读
- [Mimicking-Bench](../entities/paper-notebook-mimicking-bench-a-benchmark-for-generalizable-hu.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/Mimicking-Bench__A_Benchmark_for_Generalizable_Humanoid-Scene_Interaction_Learning/Mimicking-Bench__A_Benchmark_for_Generalizable_Humanoid-Scene_Interaction_Learning.html)
- [MolmoSpaces](../tasks/vision-language-navigation.md) — 见 wiki 实体页
- [MuJoCo Playground](../entities/mujoco-playground.md) — 见 wiki 实体页
- [RoboCasa365](../entities/paper-notebook-robocasa365-a-large-scale-simulation-framework-f.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/RoboCasa365__A_Large-Scale_Simulation_Framework_for_Training_and_Benchmarking_Generalist_Robots/RoboCasa365__A_Large-Scale_Simulation_Framework_for_Training_and_Benchmarking_Generalist_Robots.html)
- [RoboCasa](../entities/paper-notebook-robocasa-large-scale-simulation-of-everyday-task.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/RoboCasa__Large-Scale_Simulation_of_Everyday_Tasks_for_Generalist_Robots/RoboCasa__Large-Scale_Simulation_of_Everyday_Tasks_for_Generalist_Robots.html)
- [SIMPLE](../entities/paper-loco-manip-161-075-simple.md) — 见 wiki 实体页
- [Towards Motion Turing Test](../entities/paper-notebook-towards-motion-turing-test.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/Towards_Motion_Turing_Test__Evaluating_Human-Likeness_in_Humanoid_Robots/Towards_Motion_Turing_Test__Evaluating_Human-Likeness_in_Humanoid_Robots.html)

## 工程实践

- **按任务特征匹配仿真与数据生成底座：** 高吞吐刚体/操作预训练优先 ManiSkill3 / MuJoCo Playground / Genesis；面向家庭长程/灵巧双臂任务则结合 RoboCasa365、DexMimicGen 或 BiGym 做资产与轨迹扩增。
- **把「难易度分级与真实度校准」前置为准入门槛：** 参考 Benchmarking Humanoid IL with Motion Difficulty 与 Towards Motion Turing Test，切勿只看单一任务平均成功率，应在仿真基准里拆开运动复杂度、接触稳定性与类人性指标。

## 常见误区或局限

- **静态快照滞后于上游新增条目：** 本站静态分类表最初基于 22 篇快照整理，而上游 `11_Simulation_Benchmark` 仍在持续追加新笔记（如 2026-11 已见 32 篇），引用具体总数时需说明核对日期。
- **高 FPS 不自动等价于高质量接触物理：** 并行物理引擎的吞吐优势往往依赖简化的碰撞代理，跨接触操控与全身平衡迁移仍须回到实机或高保真模型校验。

## 关联页面

- 总索引：[humanoid-paper-notebooks-index.md](./humanoid-paper-notebooks-index.md)
- 协同概念：[Simulation Evaluation Infrastructure](../concepts/simulation-evaluation-infrastructure.md)
- 笔记主页：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html>

## 参考来源

- [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)
- 站点来源归档：[humanoid-robot-learning-paper-notebooks](../../sources/sites/humanoid-robot-learning-paper-notebooks.md)
- 本类论文来源笔记（节选）：[mjlab / Newton / UniLab / Genesis 出现的原因](../../sources/blogs/wechat_embodied_ai_lab_robot_training_stack_layers_2026.md)、[GRAIL（NVlabs）](../../sources/repos/grail_nvlabs.md)、[BiGym](../../sources/papers/humanoid_pnb_bigym.md)

## 推荐继续阅读

- [机器人论文阅读笔记总站](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)
