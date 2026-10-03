---
type: entity
tags: [paper, robot-learning, simulation, mujoco, benchmark, libero]
status: complete
updated: 2026-10-03
arxiv: "2009.12293"
summary: "robosuite 论文介绍了一个基于 MuJoCo 的模块化机器人学习仿真框架与基准，用可组合的机器人、任务、控制和传感器组件支持机器人学习研究。"
related:
  - ../entities/robosuite.md
  - ../entities/libero-benchmark.md
  - ../entities/awesome-sim2real.md
  - ../overview/lc-awesome-sim2real-technology-map.md
  - ../concepts/sim2real.md
  - ../methods/reinforcement-learning.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/lc_awesome_sim2real_2009_12293_robosuite-a-modular-simulation-framework.md
  - ../../sources/papers/lc_awesome_sim2real_catalog.md
  - ../../sources/repos/robosuite.md
  - ../../sources/repos/libero-benchmark.md
---

# robosuite: A modular simulation framework and benchmark for robot learning

[这篇论文](https://arxiv.org/abs/2009.12293)介绍 robosuite：一个基于 MuJoCo 的模块化仿真框架和机器人学习基准。项目将机器人模型、场景、操作物体、控制器和传感器作为可组合组件，使研究者可以在一致的仿真接口中构造任务并开展策略训练和评估。

论文收录于 [AwesomeSim2Real](https://github.com/LongchaoDa/AwesomeSim2Real) 的 **Action / Foundation Models** 分组（028/139）；此分组是策展标签，论文自身讨论的是仿真框架与机器人学习任务。

## 一句话概括

robosuite 通过统一的仿真 API 和模块化环境构造方式，降低创建机器人操作任务与比较学习方法的工程成本。它提供实验环境和基准任务，不替代策略算法本身。

## 主要思路

- **组合环境：** 通过机器人、场景、物体和任务组件构造仿真环境。
- **机器人控制：** 可配置控制器将策略动作转换为机器人控制命令；控制器选择与控制参数会影响任务行为。
- **多模态观测：** 环境可提供状态与传感器观测，具体可用数据依机器人、任务和配置而定。
- **任务与评估：** 仿真环境暴露任务交互和成功判定接口，便于训练与比较机器人学习策略。

这些要点概述项目设计，不是对论文所有实现细节或实验结果的替代；请以论文和对应代码版本为准。

## 与 LIBERO 的关系

[LIBERO](./libero-benchmark.md) 在 robosuite 之上构建语言条件机器人操作基准。LIBERO 的[安装文档](https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/installation.html)说明其底层仿真环境使用 robosuite，当前仓库 [requirements.txt](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/requirements.txt) 固定依赖 robosuite 1.4.0。

robosuite 官方[当前文档](https://robosuite.ai/docs/)对应 v1.5，而 LIBERO requirements 指定 v1.4.0。复现 LIBERO 时应遵循 LIBERO 的版本约束；不能把最新版文档与基准依赖版本混为一谈。

## 入口与复现

| 资源 | 用途 |
|---|---|
| [arXiv 论文](https://arxiv.org/abs/2009.12293) | 论文摘要、正文与引用信息 |
| [官方 GitHub](https://github.com/ARISE-Initiative/robosuite) | 源码、发布和安装信息 |
| [官方文档](https://robosuite.ai/docs/) | 当前 API 与使用指南 |
| [LIBERO 安装文档](https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/installation.html) | 基准安装方式 |
| [LIBERO requirements](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/requirements.txt) | LIBERO 声明的精确依赖版本 |

复现实验时记录 robosuite 和 MuJoCo 版本、任务配置、控制器、观测模态、随机种子及评估协议；如果升级依赖，需验证任务和指标仍与原设置兼容。

## 适用边界

- robosuite 是仿真框架，使用它不代表已有真机驱动或已验证仿真到现实迁移。
- 框架提供环境与接口，不规定唯一策略学习算法。
- 论文结果和当前软件版本可能不同；引用或复现实验时应标明采用的论文、代码与依赖版本。

## 关联页面

- 工程实体：[robosuite](./robosuite.md)
- 基准实体：[LIBERO](./libero-benchmark.md)
- 策展列表：[AwesomeSim2Real](./awesome-sim2real.md)
- 技术地图：[AwesomeSim2Real 技术地图](../overview/lc-awesome-sim2real-technology-map.md)
- 方法与任务：[Sim2Real](../concepts/sim2real.md)、[强化学习](../methods/reinforcement-learning.md)、[操作](../tasks/manipulation.md)

## 来源

- [arXiv: 2009.12293](https://arxiv.org/abs/2009.12293)
- [robosuite 官方项目](https://github.com/ARISE-Initiative/robosuite)
- [robosuite 官方文档](https://robosuite.ai/docs/)
- [LIBERO 安装文档](https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/installation.html)
- [LIBERO requirements.txt](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/requirements.txt)
- [站内来源归档](../../sources/papers/lc_awesome_sim2real_2009_12293_robosuite-a-modular-simulation-framework.md)
