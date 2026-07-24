---
type: entity
tags: [repo, unitree, unitreerobotics, hand]
status: draft
updated: 2026-07-24
related:
  - ./unitree.md
  - ./dex1-1-service.md
  - ./dfx-inspire-service.md
  - ./unitree-lerobot.md
  - ../tasks/manipulation.md
  - ./unitree-g1.md
sources:
  - ../../sources/repos/brainco_hand_service.md
  - ../../sources/repos/unitree.md
summary: "Brainco Revo2 灵巧手 Serial↔DDS 服务。"
---

# brainco_hand_service

**brainco_hand_service** 是 [unitreerobotics](https://github.com/unitreerobotics) 组织下的官方仓库，归属 **灵巧手 Serial↔DDS 服务** 主线。

## 一句话定义

Brainco Revo2 灵巧手 Serial↔DDS 服务。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DDS | Data Distribution Service | 分布式实时通信中间件 |
| SDK | Software Development Kit | 软件开发工具包 |
| API | Application Programming Interface | 应用程序编程接口 |
| G1 | Unitree G1 Humanoid | 宇树入门级人形平台 |
| IL | Imitation Learning | 模仿学习 |

## 为什么重要

人形双臂操作常依赖第三方/官方灵巧手桥接服务；Serial↔DDS 是常见集成摩擦点。

在宇树官方开源地图中，本仓是 **灵巧手 Serial↔DDS 服务** 的独立节点；与其它仓的选型关系见 [Unitree](./unitree.md)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 仓库 | [`unitreerobotics/brainco_hand_service`](https://github.com/unitreerobotics/brainco_hand_service) |
| 组织分类 | 灵巧手 Serial↔DDS 服务 |
| 星标（2026-07-24） | ~16 |
| 最近推送 | 2026-06-01 |
| 主要语言 | C++ |

## 工程实践

- Each hand (left or right) is controlled by a USB-to-serial device, and each generates a pair of topics: rt/brainco/(left or right)/(cmd or state).
- The position and speed of the fingers are normalized to the [0, 1] range.
- It is recommended to set the speed of all fingers to 1.0.
- The finger indices are mapped as follows: [Thumb, Thumbaux, Index, Middle, Ring, Pinky].

- 组织级导航与五条研发主线见 [Unitree 软件生态](./unitree.md)；原始归档见 [sources/repos/brainco_hand_service.md](../../sources/repos/brainco_hand_service.md)。

## 局限与风险

- **不要脱离代际混用**：SDK1/ROS1 遗产仓与 SDK2/ROS2/DDS 新栈的消息与启动方式不兼容。
- **README 边界优先**：功能承诺以官方 README 为准；星标高不等于适合你的机型/仿真器。
- **外设/第三方手部仓**：驱动与固件版本需与具体灵巧手/雷达硬件对齐，否则 Serial↔DDS 桥会静默失败。

## 关联页面

- [Unitree 组织枢纽](./unitree.md)
- [dex1_1_service](./dex1-1-service.md)
- [dfx_inspire_service](./dfx-inspire-service.md)
- [unitree_lerobot](./unitree-lerobot.md)
- [Manipulation](../tasks/manipulation.md)
- [Unitree G1](./unitree-g1.md)

## 参考来源

- [sources/repos/brainco_hand_service.md](../../sources/repos/brainco_hand_service.md)
- [sources/repos/unitree.md](../../sources/repos/unitree.md) — 组织级仓库地图
- 上游仓库：<https://github.com/unitreerobotics/brainco_hand_service>

## 推荐继续阅读

- 官方开发者文档：<https://support.unitree.com/home/zh/developer>
- 组织总览：<https://github.com/unitreerobotics>
