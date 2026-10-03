---
type: entity
tags: [simulation, mujoco, robot-learning, manipulation, libero, repo]
status: complete
updated: 2026-10-03
summary: "robosuite 是基于 MuJoCo 的模块化机器人仿真框架，提供机器人、场景、物体、控制器、传感器和操作任务组件；LIBERO 将其作为底层仿真环境，并在 requirements 中固定 robosuite 1.4.0。"
related:
  - ../entities/libero-benchmark.md
  - ../entities/paper-as-2009-12293-robosuite-a-modular-simulation-framework-and-ben.md
  - ../concepts/sim2real.md
  - ../entities/isaac-lab.md
  - ../entities/humanoid-motion-intelligence.md
sources:
  - ../../sources/repos/robosuite.md
  - ../../sources/papers/lc_awesome_sim2real_2009_12293_robosuite-a-modular-simulation-framework.md
  - ../../sources/repos/libero-benchmark.md
---

# robosuite

[robosuite](https://github.com/ARISE-Initiative/robosuite) 是一个以 MuJoCo 为后端的机器人学习仿真框架。它把机器人模型、操作场景、物体、控制器和传感器组合成可配置的任务环境，并为策略训练、评测和遥操作提供接口。

## 核心定位

robosuite 是仿真与任务环境层，不是一个单独的学习算法或真机驱动程序。研究者可以选择机器人和控制器，构造场景与任务，再通过环境 API 执行动作并取得观测、奖励和任务完成信息。具体接口与可用组件以所选版本的官方文档为准。

常见使用链路：

```text
机器人 / 场景 / 物体配置 → robosuite 仿真环境 → 动作与观测 → 策略训练或基准评估
```

官方项目资料：

- [GitHub 源码](https://github.com/ARISE-Initiative/robosuite)
- [项目主页](https://robosuite.ai/)
- [官方文档](https://robosuite.ai/docs/)
- [项目论文](https://arxiv.org/abs/2009.12293)
- [白皮书 PDF](https://robosuite.ai/assets/whitepaper.pdf)

## 与 LIBERO 的关系

[LIBERO](./libero-benchmark.md) 是终身机器人学习操作基准。其安装文档将 robosuite 作为底层仿真环境；LIBERO 仓库的 [requirements.txt](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/requirements.txt) 固定 `robosuite==1.4.0`。因此：

- **分工：** robosuite 提供仿真引擎和环境组件；LIBERO 提供语言条件任务、演示数据和知识迁移评估协议。
- **版本：** robosuite 当前官方文档对应 v1.5，而 LIBERO 依赖文件指定 v1.4.0。重跑 LIBERO 时应按其依赖文件和[安装文档](https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/installation.html)使用匹配版本。
- **复现：** 不要仅因能安装最新版就替换基准依赖；版本改变可能影响环境和评估的复现性。若升级，应检查 LIBERO 任务注册、控制接口及结果是否兼容。

## 适用场景

- 在 MuJoCo 中训练和评估机器人操作策略。
- 通过替换机器人、控制器、物体或场景组件构造实验。
- 为下游基准（如 LIBERO）提供物理仿真环境。
- 对比不同策略时，固定任务定义、环境版本和评估配置。

## 使用时核对

| 检查项 | 建议 |
|---|---|
| 版本 | 以目标基准的依赖锁定为准；LIBERO 当前 requirements 固定 1.4.0 |
| 仿真后端 | 按所选 robosuite 版本的安装文档核对 MuJoCo 要求 |
| 任务定义 | 区分 robosuite 自带任务与 LIBERO 的任务套件 |
| 评估复现 | 固定环境版本、随机种子、控制器、观测模态和成功条件 |
| 真机部署 | 仿真 API 本身不等同于真机驱动；另行核对机器人接口与迁移流程 |

## 来源与关联页面

- [robosuite 来源归档](../../sources/repos/robosuite.md)
- [robosuite 论文实体](./paper-as-2009-12293-robosuite-a-modular-simulation-framework-and-ben.md)
- [LIBERO 基准实体](./libero-benchmark.md)
- [Sim2Real](../concepts/sim2real.md)
- [官方源码](https://github.com/ARISE-Initiative/robosuite)
- [官方文档](https://robosuite.ai/docs/)
- [LIBERO requirements](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/requirements.txt)
