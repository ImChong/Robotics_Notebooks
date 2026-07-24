---
type: entity
tags: [repo, unitree, unitreerobotics, challenge]
status: draft
updated: 2026-07-24
related:
  - ./unitree.md
  - ./unitree-lerobot.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/repos/unibot_submission.md
  - ../../sources/repos/unitree.md
summary: "UniBot Challenge 赛事资源仓。"
---

# unibot_submission

**unibot_submission** 是 [unitreerobotics](https://github.com/unitreerobotics) 组织下的官方仓库，归属 **赛事 / Challenge** 主线。

## 一句话定义

UniBot Challenge 赛事资源仓。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| API | Application Programming Interface | 应用程序编程接口 |
| IL | Imitation Learning | 模仿学习 |
| RL | Reinforcement Learning | 强化学习 |
| G1 | Unitree G1 Humanoid | 宇树入门级人形平台 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

赛事资源仓沉淀评测协议与提交约定，便于对齐社区基准。

在宇树官方开源地图中，本仓是 **赛事 / Challenge** 的独立节点；与其它仓的选型关系见 [Unitree](./unitree.md)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 仓库 | [`unitreerobotics/unibot_submission`](https://github.com/unitreerobotics/unibot_submission) |
| 组织分类 | 赛事 / Challenge |
| 星标（2026-07-24） | ~6 |
| 最近推送 | 2026-07-10 |
| 主要语言 | Python |

## 工程实践

- 从组织枢纽 [Unitree](./unitree.md) 确认本仓所属主线后再克隆。
- 对照上游 README 安装依赖，并与 SDK2 / ROS2 代际对齐。

- 组织级导航与五条研发主线见 [Unitree 软件生态](./unitree.md)；原始归档见 [sources/repos/unibot_submission.md](../../sources/repos/unibot_submission.md)。

## 局限与风险

- **不要脱离代际混用**：SDK1/ROS1 遗产仓与 SDK2/ROS2/DDS 新栈的消息与启动方式不兼容。
- **README 边界优先**：功能承诺以官方 README 为准；星标高不等于适合你的机型/仿真器。
- **外设/第三方手部仓**：驱动与固件版本需与具体灵巧手/雷达硬件对齐，否则 Serial↔DDS 桥会静默失败。

## 关联页面

- [Unitree 组织枢纽](./unitree.md)
- [unitree_lerobot](./unitree-lerobot.md)
- [Teleoperation](../tasks/teleoperation.md)

## 参考来源

- [sources/repos/unibot_submission.md](../../sources/repos/unibot_submission.md)
- [sources/repos/unitree.md](../../sources/repos/unitree.md) — 组织级仓库地图
- 上游仓库：<https://github.com/unitreerobotics/unibot_submission>

## 推荐继续阅读

- 官方开发者文档：<https://support.unitree.com/home/zh/developer>
- 组织总览：<https://github.com/unitreerobotics>
