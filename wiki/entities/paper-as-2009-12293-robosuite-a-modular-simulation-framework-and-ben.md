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

## 一句话定义

[这篇论文](https://arxiv.org/abs/2009.12293)介绍 robosuite：一个基于 MuJoCo 的模块化仿真框架和机器人学习基准。项目将机器人模型、场景、操作物体、控制器和传感器作为可组合组件，使研究者可以在统一仿真接口中构造任务并开展策略训练和评估。

论文收录于 [AwesomeSim2Real](https://github.com/LongchaoDa/AwesomeSim2Real) 的 **Action / Foundation Models** 分组（028/139）；这只是策展标签，论文主题是仿真框架与机器人学习任务。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| API | Application Programming Interface | 仿真环境交互接口 |
| MDP | Markov Decision Process | 状态、动作与转移的形式化模型 |
| MuJoCo | Multi-Joint dynamics with Contact | robosuite 使用的物理仿真引擎 |
| RL | Reinforcement Learning | 可在框架环境中训练的策略方法 |

## 方法

- **组合环境：** 通过机器人、场景、物体和任务组件构造仿真环境。
- **机器人控制：** 可配置控制器将策略动作转换为机器人控制命令；控制器与参数会影响任务行为。
- **多模态观测：** 环境可提供状态与传感器观测，实际内容依机器人、任务和配置而定。
- **任务接口：** 环境提供交互与任务完成判定接口，供训练和评估流程调用。

这些要点概述项目设计，不替代论文对实现和算法设置的完整描述。

## 评测

论文将 robosuite 定位为仿真框架与机器人学习 benchmark，并使用若干机器人操作任务展示框架如何支持研究和评估。此知识页不摘录论文表格中的具体数值；实验任务、指标及其设定请直接对照 [论文正文](https://arxiv.org/abs/2009.12293) 和对应代码版本。

复现实验应记录 robosuite 与 MuJoCo 版本、机器人和任务配置、控制器、观测模态、随机种子及评估协议。

## 与其他工作对比

robosuite 更接近可组合的仿真基础设施，和单一策略算法或任务数据集并非同一层级。比较机器人学习结果时，应固定环境与任务协议，再对比策略；跨仿真器或跨版本结果需要先核对动作、观测和成功判据是否一致。

[LIBERO](./libero-benchmark.md) 基于 robosuite 构建语言条件操作任务与知识迁移评估。LIBERO [requirements.txt](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/requirements.txt) 固定 robosuite 1.4.0；当前 robosuite 官方文档对应 v1.5，二者版本不可混同。

## 结论

robosuite 的核心贡献是提供可组合的仿真环境、机器人控制接口与任务基准，便于开展可重复的机器人学习实验。复现论文或 LIBERO 时应注明实际采用的代码与依赖版本，并以对应版本的接口和任务定义为准。

## 入口与复现

| 资源 | 用途 |
|---|---|
| [arXiv 论文](https://arxiv.org/abs/2009.12293) | 论文摘要、正文与引用信息 |
| [官方 GitHub](https://github.com/ARISE-Initiative/robosuite) | 源码、发布和安装信息 |
| [官方文档](https://robosuite.ai/docs/) | 当前 API 与使用指南 |
| [LIBERO 安装文档](https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/installation.html) | 基准安装方式 |
| [LIBERO requirements](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/requirements.txt) | LIBERO 声明的精确依赖版本 |

## 适用边界

- robosuite 是仿真框架，使用它不代表已有真机驱动或已验证仿真到现实迁移。
- 框架提供环境与接口，不规定唯一策略学习算法。
- 论文和当前软件版本可能不同；引用时应标明采用的论文、代码与依赖版本。

## 关联页面

- 工程实体：[robosuite](./robosuite.md)
- 基准实体：[LIBERO](./libero-benchmark.md)（论文页：[LIBERO 2306.03310](./paper-rcl-2306-03310-libero-benchmarking-knowledge-transfer-for-lifel.md)）
- 评测选型：[具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — robosuite 作为策略任务成功率评测层的仿真底座
- 策展列表：[AwesomeSim2Real](./awesome-sim2real.md)
- 技术地图：[AwesomeSim2Real 技术地图](../overview/lc-awesome-sim2real-technology-map.md)
- 方法与任务：[Sim2Real](../concepts/sim2real.md)、[强化学习](../methods/reinforcement-learning.md)、[操作](../tasks/manipulation.md)

## 参考来源

- [arXiv: 2009.12293](https://arxiv.org/abs/2009.12293)
- [robosuite 官方项目](https://github.com/ARISE-Initiative/robosuite)
- [robosuite 官方文档](https://robosuite.ai/docs/)
- [LIBERO 安装文档](https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/installation.html)
- [LIBERO requirements.txt](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/requirements.txt)
- [站内来源归档](../../sources/papers/lc_awesome_sim2real_2009_12293_robosuite-a-modular-simulation-framework.md)
