---
type: entity
tags: [repo, unitree, unitreerobotics, meta]
status: draft
updated: 2026-07-24
related:
  - ./unitree.md
  - ./humanoid-robot.md
sources:
  - ../../sources/repos/Publications.md
  - ../../sources/repos/unitree.md
summary: "宇树发表论文索引仓。"
---

# Publications

**Publications** 是 [unitreerobotics](https://github.com/unitreerobotics) 组织下的官方仓库，归属 **组织元资料** 主线。

## 一句话定义

宇树发表论文索引仓。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| API | Application Programming Interface | 应用程序编程接口 |
| SDK | Software Development Kit | 软件开发工具包 |
| RL | Reinforcement Learning | 强化学习 |
| G1 | Unitree G1 Humanoid | 宇树入门级人形平台 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

组织级论文索引便于追溯官方发表脉络。

在宇树官方开源地图中，本仓是 **组织元资料** 的独立节点；与其它仓的选型关系见 [Unitree](./unitree.md)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 仓库 | [`unitreerobotics/Publications`](https://github.com/unitreerobotics/Publications) |
| 组织分类 | 组织元资料 |
| 星标（2026-07-24） | ~96 |
| 最近推送 | 2023-04-17 |
| 主要语言 | N/A |

## 工程实践

- 从组织枢纽 [Unitree](./unitree.md) 确认本仓所属主线后再克隆。
- 对照上游 README 安装依赖，并与 SDK2 / ROS2 代际对齐。

- 组织级导航与五条研发主线见 [Unitree 软件生态](./unitree.md)；原始归档见 [sources/repos/Publications.md](../../sources/repos/Publications.md)。

## 局限与风险

- **不要脱离代际混用**：SDK1/ROS1 遗产仓与 SDK2/ROS2/DDS 新栈的消息与启动方式不兼容。
- **README 边界优先**：功能承诺以官方 README 为准；星标高不等于适合你的机型/仿真器。
- **外设/第三方手部仓**：驱动与固件版本需与具体灵巧手/雷达硬件对齐，否则 Serial↔DDS 桥会静默失败。

## 关联页面

- [Unitree 组织枢纽](./unitree.md)
- [人形机器人](./humanoid-robot.md)

## 参考来源

- [sources/repos/Publications.md](../../sources/repos/Publications.md)
- [sources/repos/unitree.md](../../sources/repos/unitree.md) — 组织级仓库地图
- 上游仓库：<https://github.com/unitreerobotics/Publications>

## 推荐继续阅读

- 官方开发者文档：<https://support.unitree.com/home/zh/developer>
- 组织总览：<https://github.com/unitreerobotics>
