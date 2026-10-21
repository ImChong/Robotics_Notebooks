---
type: entity
project_id: embodied-human-simulation-interactive-robotics
tags: [paper, human-robot-interaction, biomechanics, exoskeleton, simulation, reinforcement-learning, tsinghua]
status: complete
updated: 2026-10-21
arxiv: "2603.09218"
venue: "IROS 2026"
project: https://lnsgroup.cc/research/HRI/
related:
  - ../tasks/loco-manipulation.md
  - ./paper-opencap-monocular.md
sources:
  - ../../sources/papers/embodied_human_simulation_arxiv_2603_09218.md
  - ../../sources/sites/embodied-human-simulation-hri.md
summary: "以受强化学习控制的全身肌肉骨骼数字人作为预测性代理，在耦合仿真中量化人体与外骨骼/人形机器人的交互，并以预训练人体策略支持结构和控制参数协同优化；项目页标注代码即将开放。"
---

# Embodied Human Simulation：面向交互机器人的具身人体仿真

**Embodied Human Simulation**（arXiv:2603.09218，清华大学 LNS Group，IROS 2026）用受强化学习控制的全身肌肉骨骼模型模拟人体对外骨骼及协作机器人的响应，并将人体内部生物力学量纳入机器人结构与控制设计评估。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HRI | Human-Robot Interaction | 人与机器人之间的物理或非物理交互 |
| RL | Reinforcement Learning | 通过交互奖励训练人体运动控制策略 |
| MS-HRI | Musculoskeletal Human-Robot Interaction | 项目页对该研究方向的称呼 |
| IROS | IEEE/RSJ International Conference on Intelligent Robots and Systems | 项目页标注的发表会议 |

## 核心思路

传统交互机器人设计常需反复开展真人实验，且难以直接测量肌肉力、关节载荷等人体内部状态。该工作使用全身肌肉骨骼数字人作为可计算的预测代理：先学习能跟踪运动且能响应扰动的人体控制策略，再将其与机器人放入耦合 MuJoCo 仿真，以交互成本评价设计方案。

预训练人体策略在设计搜索中保持为一致的评估器，优化器据此探索机器人结构参数与控制策略的组合。这样可以把「结构如何改变人机接触」和「控制如何影响任务表现」放进同一评估闭环。

## 展示与结果

- **外骨骼步行：** 项目页展示人体穿戴外骨骼的协同优化案例，报告优化后人体与机器人关节轴更对齐、接触力降低。
- **人形协作：** 展示人与 Unitree G1 协作搬桌，说明框架可扩展至不同交互机器人。
- **人体策略验证：** 项目页展示步态跟踪与受推扰后的恢复评估。

## 工程实践与开源状态

项目页使用 MuJoCo 进行耦合仿真，适合关注生物力学指标的人机交互设计研究。项目页截至 2026-10-21 标注 **“Code Coming Soon”**，未提供公开代码、模型权重或数据下载入口，因此目前无法据公开资料复现其具体训练与优化实现；不能将演示结果直接视为真实人体实验或临床安全性结论。

## 结论

**总判：** 该工作把可测的人体生物力学状态纳入交互机器人设计闭环，适合作为外骨骼与协作机器人仿真评估的研究范式参考。

- 设计评估时同时考虑机器人结构与控制，避免只优化单侧参数。
- 将肌肉力、关节负载、接触力及关节轴对齐作为互补指标，而非只看任务成功率。
- 先验证数字人的步态跟踪和扰动恢复，再信任其作为设计评估器。
- 目前代码未开放，落地复现需等待官方发布，或自行搭建肌肉骨骼模型与耦合控制环境。

## 关联页面

- [Loco-Manipulation](../tasks/loco-manipulation.md)：机器人全身运动与操作任务背景。
- [OpenCap Monocular](./paper-opencap-monocular.md)：从视频估计人体运动学与肌肉骨骼动力学的相关工作。

## 推荐继续阅读

- [项目主页](https://lnsgroup.cc/research/HRI/)
- [论文：arXiv:2603.09218](https://arxiv.org/abs/2603.09218)

## 参考来源

- [论文来源归档](../../sources/papers/embodied_human_simulation_arxiv_2603_09218.md)
- [官方项目页归档](../../sources/sites/embodied-human-simulation-hri.md)
