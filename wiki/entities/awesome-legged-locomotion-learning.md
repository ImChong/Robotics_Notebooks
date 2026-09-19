---
type: entity
tags: [curated-list, locomotion, legged-robots, reinforcement-learning, sim2real, urdf]
status: complete
updated: 2026-09-19
related:
  - ../tasks/locomotion.md
  - ./awesome-robot-descriptions.md
  - ./awesome-legged-robot-learning-clearlab.md
  - ../concepts/sim2real.md
  - ../methods/reinforcement-learning.md
  - ../entities/paper-rma-rapid-motor-adaptation.md
sources:
  - ../../sources/repos/awesome-legged-locomotion-learning.md
summary: "gaiyi7788 维护的腿足 locomotion learning 策展清单：机器人模型表、训练代码、Survey 与按会议分组的论文索引（~487 stars）。"
---

# awesome-legged-locomotion-learning

[`gaiyi7788/awesome-legged-locomotion-learning`](https://github.com/gaiyi7788/awesome-legged-locomotion-learning) 是一份 **腿足 locomotion learning** 的个人策展清单：除论文外还收录 **机器人 URDF/MJCF 模型表**、**legged_gym / GenLoco** 等代码入口、Sim2Real 综述与中文 Isaac Gym 入门博客。

## 一句话定义

**腿足 locomotion 入门向全栈索引** — 从机型描述、仿真训练框架到 perceptive locomotion / sim2real 代表论文的一站式 Markdown 目录。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 清单核心方法族 |
| Sim2Real | Simulation to Real | 大量 Survey 与论文标签 |
| URDF | Unified Robot Description Format | Robot models 表主格式 |
| MJCF | MuJoCo XML Format | 四足/人形 Menagerie 链接 |
| AMP | Adversarial Motion Priors | 相关论文与 character animation 交叉 |

## 为什么重要

- **模型 + 论文一体：** Robot models 表按双足/人形/四足给出格式、License 与 visual/inertia/collision 勾选，与 [Awesome Robot Descriptions](./awesome-robot-descriptions.md) 互补，适合刚选型仿真后端时对照。
- **代码入口集中：** legged_gym、terrain_benchmark、walk-these-ways、GenLoco 等链到可运行仓，比纯 paper list 更接近复现路径。
- **中文社区友好：** Technical blog 含 Isaac Gym 等中文长文，降低国内读者检索成本。
- **与 ClearLab 清单分工：** 本列表偏 **历史广度 + 模型/代码**；[Awesome-Legged-Robot-Learning（SUSTech）](./awesome-legged-robot-learning-clearlab.md) 偏 **2024–2025 人形/WBC 前沿 arXiv**。

## 核心结构

| 分区 | 用途 |
|------|------|
| Related awesome-lists | Isaac Gym、Hybrid Robotics、四足专题等 |
| Robot models | URDF/MJCF 表（License + 三项几何勾选） |
| Code | 仿真、RL 框架与 benchmark 仓 |
| Survey | Sim2Real、RL for robotics、perceptive locomotion |
| Papers | 按年份/venue；含 vision-guided、imitation 标签 |

## 局限与使用注意

- **维护节奏个人化：** 维护者自述逐步更新；2023 后 arXiv 需与 [ClearLab 清单](./awesome-legged-robot-learning-clearlab.md) 或 [awesome-humanoid-robot-learning](https://github.com/YanjieZe/awesome-humanoid-robot-learning) 交叉核对。
- **非运行时代码：** 列表本身不可 `pip install`；复现须跟链到具体项目页核权重/环境。
- **许可证未统一 SPDX：** 表内模型 License 各异，商用前逐条读上游。

## 关联页面

- [Locomotion](../tasks/locomotion.md)
- [Awesome Robot Descriptions](./awesome-robot-descriptions.md)
- [Awesome-Legged-Robot-Learning（ClearLab）](./awesome-legged-robot-learning-clearlab.md)
- [Sim2Real](../concepts/sim2real.md)
- [RMA](./paper-rma-rapid-motor-adaptation.md)

## 参考来源

- [sources/repos/awesome-legged-locomotion-learning.md](../../sources/repos/awesome-legged-locomotion-learning.md)

## 推荐继续阅读

- GitHub：<https://github.com/gaiyi7788/awesome-legged-locomotion-learning>
- [legged_gym](https://github.com/leggedrobotics/legged_gym) — 清单高频代码入口
