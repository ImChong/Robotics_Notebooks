---
type: entity
tags: [repo, unitree, unitreerobotics, slam]
status: draft
updated: 2026-07-24
related:
  - ./unitree.md
  - ./unitree-slam.md
  - ./point-lio-unilidar.md
sources:
  - ../../sources/repos/Python_unitree_demos.md
  - ../../sources/repos/unitree.md
summary: "Unitree SLAM 行业应用场景的 Python 示例例程。"
---

# Python_unitree_demos

**Python_unitree_demos** 是 [unitreerobotics](https://github.com/unitreerobotics) 组织下的官方仓库，归属 **SLAM / 行业示例** 主线。

## 一句话定义

Unitree SLAM 行业应用场景的 Python 示例例程。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SLAM | Simultaneous Localization and Mapping | 同步定位与建图 |
| LiDAR | Light Detection and Ranging | 激光雷达 |
| ROS 2 | Robot Operating System 2 | 机器人系统集成常用中间件 |
| SDK | Software Development Kit | 软件开发工具包 |
| API | Application Programming Interface | 应用程序编程接口 |

## 为什么重要

SLAM 接口与行业示例服务移动操作/巡检类落地场景。

在宇树官方开源地图中，本仓是 **SLAM / 行业示例** 的独立节点；与其它仓的选型关系见 [Unitree](./unitree.md)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 仓库 | [`unitreerobotics/Python_unitree_demos`](https://github.com/unitreerobotics/Python_unitree_demos) |
| 组织分类 | SLAM / 行业示例 |
| 星标（2026-07-24） | ~0 |
| 最近推送 | 2026-07-22 |
| 主要语言 | Python |

## 工程实践

- 从组织枢纽 [Unitree](./unitree.md) 确认本仓所属主线后再克隆。
- 对照上游 README 安装依赖，并与 SDK2 / ROS2 代际对齐。

- 组织级导航与五条研发主线见 [Unitree 软件生态](./unitree.md)；原始归档见 [sources/repos/Python_unitree_demos.md](../../sources/repos/Python_unitree_demos.md)。

## 局限与风险

- **不要脱离代际混用**：SDK1/ROS1 遗产仓与 SDK2/ROS2/DDS 新栈的消息与启动方式不兼容。
- **README 边界优先**：功能承诺以官方 README 为准；星标高不等于适合你的机型/仿真器。
- **外设/第三方手部仓**：驱动与固件版本需与具体灵巧手/雷达硬件对齐，否则 Serial↔DDS 桥会静默失败。

## 关联页面

- [Unitree 组织枢纽](./unitree.md)
- [unitree_slam](./unitree-slam.md)
- [point_lio_unilidar](./point-lio-unilidar.md)

## 参考来源

- [sources/repos/Python_unitree_demos.md](../../sources/repos/Python_unitree_demos.md)
- [sources/repos/unitree.md](../../sources/repos/unitree.md) — 组织级仓库地图
- 上游仓库：<https://github.com/unitreerobotics/Python_unitree_demos>

## 推荐继续阅读

- 官方开发者文档：<https://support.unitree.com/home/zh/developer>
- 组织总览：<https://github.com/unitreerobotics>
