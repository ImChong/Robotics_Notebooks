---
type: entity
tags: [humanoid, isaac-lab, rl, sim2sim, sim2real, roboparty]
status: complete
updated: 2026-04-25
related:
  - ./roboto-origin.md
  - ./robot-lab.md
  - ./isaac-gym-isaac-lab.md
sources:
  - ../../sources/repos/atom01_train.md
summary: "atom01_train 是 Atom01 的训练仓库，围绕 IsaacLab 训练配置、策略学习与仿真迁移流程构建。"
---

# Atom01 Train

**atom01_train** 是 Roboparty Atom01 项目的训练主仓库，聚焦 IsaacLab 场景下的策略学习、实验配置与迁移链路。

## 为什么重要

- 是 Atom01 从模型到策略的训练入口。
- 将硬件平台约束映射到训练环境，帮助减少部署落差。
- 与 `atom01_deploy` 联合构成训练→部署闭环。

## 核心结构/机制

- **训练配置**：任务参数、奖励项与训练超参数。
- **仿真迁移**：支持 Sim2Sim/Sim2Real 工作流。
- **工程接口**：与模型描述与部署链路对齐。

## 常见误区或局限

- 误区：只要训练收敛就能真机稳定。真机仍受通信、校准、硬件误差影响。
- 局限：训练仓库往往强调算法迭代，部署鲁棒性要靠额外工程补齐。

## 参考来源

- [sources/repos/atom01_train.md](../../sources/repos/atom01_train.md)
- [Roboparty/atom01_train](https://github.com/Roboparty/atom01_train)

## 关联页面

- [Roboto Origin（开源人形机器人基线）](./roboto-origin.md)
- [robot_lab (IsaacLab 扩展框架)](./robot-lab.md)
- [Isaac Gym / Isaac Lab](./isaac-gym-isaac-lab.md)

## 推荐继续阅读

- [Atom01 Deploy](./atom01-deploy.md)
- [Reinforcement Learning](../methods/reinforcement-learning.md)
