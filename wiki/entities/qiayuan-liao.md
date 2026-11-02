---
type: entity
tags: [humanoid, whole-body-control, reinforcement-learning, sim2real, berkeley]
status: complete
updated: 2026-11-01
summary: "Qiayuan Liao 的研究主页连接人形全身控制、移动操作和机器人开源工具。"
related:
  - ./paper-supermimic.md
  - ./workhorse-humanoid-loco-manipulation.md
  - ./mjlab.md
  - ./mujoco-playground.md
  - ../methods/beyondmimic.md
sources:
  - ../../sources/sites/qiayuan-liao-homepage.md
---

# Qiayuan Liao

**Qiayuan Liao** 是加州大学伯克利分校机器人研究者；其个人主页汇集人形全身控制、移动操作与机器人软件资源。身份和项目状态以 2026-11-01 主页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| RL | Reinforcement Learning | 从交互中学习行为的方法 |
| WBC | Whole-Body Control | 协调机器人全身运动和接触的控制 |
| NMPC | Nonlinear Model Predictive Control | 用非线性模型进行滚动优化的控制方法 |

## 研究脉络与阅读入口

主页自述聚焦敏捷、鲁棒的全身移动操作，以及数据、硬件和控制协同设计。阅读时可从以下相互关联但独立的项目入手：

- [SuperMimic](./paper-supermimic.md)：通过可微世界模型训练人形动作跟踪策略。
- [Workhorse](./workhorse-humanoid-loco-manipulation.md)：从人类数据学习人形全身移动操作。
- [BeyondMimic](../methods/beyondmimic.md)：从运动跟踪扩展到多样人形控制。
- [mjlab](./mjlab.md) 与 [MuJoCo Playground](./mujoco-playground.md)：机器人强化学习与仿真工具入口。

这些方向共同连接数据、仿真训练和机器人控制，但各论文的具体方法、实验与开源边界不同，应以各自项目资料为准。

## 开源与使用边界

主页列出 [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)、[legged_control](https://github.com/qiayuanl/legged_control)、[mjlab](https://github.com/mujocolab/mjlab) 等公开仓库。主页将 legged_control 和 MuJoCo Playground 标为 retired；whole_body_tracking 已归档。它们并非同一代码库，使用前应分别检查版本、依赖和维护状态。

## 关联页面

- [全身控制](../concepts/whole-body-control.md)
- [强化学习](../methods/reinforcement-learning.md)

## 参考来源

- [Qiayuan Liao 个人主页归档](../../sources/sites/qiayuan-liao-homepage.md)

## 推荐继续阅读

- [个人主页](https://qiayuanl.github.io/) — 研究与软件完整入口。
