---
type: entity
tags: [mobile-robot, education, ros, autonomous-driving, open-source]
status: complete
updated: 2026-10-27
project_id: duckietown-duckiebot
code: https://github.com/duckietown/dt-duckiebot-interface
summary: "Duckiebot：Duckietown 生态的标准化开源小车，Raspberry Pi + ROS，配 gym-duckietown 仿真与真实道路系统，用于自主驾驶教育与研究。"
related:
  - ./turtlebot3.md
  - ./paper-openbot-smartphone-mobile-robot.md
  - ../concepts/sim2real.md
  - ./paper-as-130-duckietown-an-open-inexpensive-and-flexible-plat.md
  - ../overview/robot-open-source-wechat-githubstore-18-projects.md
sources:
  - ../../sources/blogs/wechat_githubstore_18_open_source_robot_projects.md
---

# Duckiebot（Duckietown）

## 一句话定义

**Duckiebot** 是 Duckietown 生态中的标准化开源小车，组织 [duckietown](https://github.com/duckietown)，关键仓库 [dt-duckiebot-interface](https://github.com/duckietown/dt-duckiebot-interface)、[dt-core](https://github.com/duckietown/dt-core)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ROS | Robot Operating System | 机器人中间件 |
| Gym | OpenAI Gym interface | 强化学习环境接口 |
| MARL | Multi-Agent RL | 多智能体强化学习 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- 自动驾驶教学领域的标准化平台；本库另有 Duckietown 论文清单条目，本页是平台节点。
- 在微信文章《想进入机器人领域吗？学习这18个开源项目就够了》中排第 **18** 位。

## 核心要点（据文章）

- 基于 Raspberry Pi，支持 ROS，有完整的仿真环境（gym-duckietown）和真实道路系统。
- 代码高度模块化，组织下有数百个仓库；适合学习感知、规划和多智能体。

## 工程实践

- 先从 dt-duckiebot-interface 与 dt-core 入手，在 gym-duckietown 里调通再上真车。
- 以上规格与许可证均出自文章叙述，**未逐项核对官方仓库**；动手前请以官方 README/BOM 为准。

## 局限与风险

- 本页仅基于一篇第三方公众号文章，规格数字可能随版本变化。
- 仓库迁移、分叉和许可证以官方当前状态为准。

## 关联页面

- [turtlebot3](./turtlebot3.md)
- [paper-openbot-smartphone-mobile-robot](./paper-openbot-smartphone-mobile-robot.md)
- [sim2real](../concepts/sim2real.md)
- [paper-as-130-duckietown-an-open-inexpensive-and-flexible-plat](./paper-as-130-duckietown-an-open-inexpensive-and-flexible-plat.md)
- [机器人开源项目 18 例（微信 GitHubStore）索引](../overview/robot-open-source-wechat-githubstore-18-projects.md)

## 推荐继续阅读

- [Duckietown 组织](https://github.com/duckietown)

## 参考来源

- [wechat_githubstore_18_open_source_robot_projects.md](../../sources/blogs/wechat_githubstore_18_open_source_robot_projects.md)
