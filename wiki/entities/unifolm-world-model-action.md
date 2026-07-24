---
type: entity
tags: [repo, unitree, unitreerobotics, foundation]
status: draft
updated: 2026-07-24
related:
  - ./unitree.md
  - ./unifolm-vla.md
  - ./unitree-lerobot.md
  - ../concepts/world-action-models.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/repos/unifolm-world-model-action.md
  - ../../sources/repos/unitree.md
summary: "官方 UnifoLM-WMA-0：世界模型作仿真引擎 + 动作头策略增强。"
---

# unifolm-world-model-action

**unifolm-world-model-action** 是 [unitreerobotics](https://github.com/unitreerobotics) 组织下的官方仓库，归属 **基础模型（UnifoLM）** 主线。

## 一句话定义

官方 UnifoLM-WMA-0：世界模型作仿真引擎 + 动作头策略增强。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |
| WMA | World-Model-Action | 世界模型–动作架构 |
| IL | Imitation Learning | 模仿学习 |
| G1 | Unitree G1 Humanoid | 宇树入门级人形平台 |
| API | Application Programming Interface | 应用程序编程接口 |

## 为什么重要

官方 UnifoLM 代表宇树在 VLA/WMA 方向的开源落点，便于与自研策略对照。

在宇树官方开源地图中，本仓是 **基础模型（UnifoLM）** 的独立节点；与其它仓的选型关系见 [Unitree](./unitree.md)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 仓库 | [`unitreerobotics/unifolm-world-model-action`](https://github.com/unitreerobotics/unifolm-world-model-action) |
| 组织分类 | 基础模型（UnifoLM） |
| 星标（2026-07-24） | ~1083 |
| 最近推送 | 2026-03-18 |
| 主要语言 | Python |

## 工程实践

- Sep 22, 2025: 🚀 We released the deployment code for assisting experiments with Unitree robots.
- Sep 15, 2025: 🚀 We released the training and inference code along with the model weights of UnifoLM-WMA-0.
- [x] Training
- [x] Inference

- 组织级导航与五条研发主线见 [Unitree 软件生态](./unitree.md)；原始归档见 [sources/repos/unifolm-world-model-action.md](../../sources/repos/unifolm-world-model-action.md)。

## 局限与风险

- **不要脱离代际混用**：SDK1/ROS1 遗产仓与 SDK2/ROS2/DDS 新栈的消息与启动方式不兼容。
- **README 边界优先**：功能承诺以官方 README 为准；星标高不等于适合你的机型/仿真器。
- **外设/第三方手部仓**：驱动与固件版本需与具体灵巧手/雷达硬件对齐，否则 Serial↔DDS 桥会静默失败。

## 关联页面

- [Unitree 组织枢纽](./unitree.md)
- [unifolm-vla](./unifolm-vla.md)
- [unitree_lerobot](./unitree-lerobot.md)
- [World-Action Models](../concepts/world-action-models.md)
- [Imitation Learning](../methods/imitation-learning.md)

## 参考来源

- [sources/repos/unifolm-world-model-action.md](../../sources/repos/unifolm-world-model-action.md)
- [sources/repos/unitree.md](../../sources/repos/unitree.md) — 组织级仓库地图
- 上游仓库：<https://github.com/unitreerobotics/unifolm-world-model-action>

## 推荐继续阅读

- 官方开发者文档：<https://support.unitree.com/home/zh/developer>
- 组织总览：<https://github.com/unitreerobotics>
