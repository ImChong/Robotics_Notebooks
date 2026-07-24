---
type: entity
tags: [repo, unitree, unitreerobotics, sim]
status: draft
updated: 2026-07-24
related:
  - ./unitree.md
  - ./unitree-mujoco.md
  - ./unitree-guide.md
  - ./unitree-rl-mjlab.md
  - ../concepts/sim2real.md
  - ./mujoco.md
sources:
  - ../../sources/repos/unitree_model.md
  - ../../sources/repos/unitree.md
summary: "GitHub 侧模型仓已标注 deprecated；后续 USD/模型更新以 Hugging Face unitree_model 为准。"
---

# unitree_model

**unitree_model** 是 [unitreerobotics](https://github.com/unitreerobotics) 组织下的官方仓库，归属 **仿真与模型** 主线。

## 一句话定义

GitHub 侧模型仓已标注 deprecated；后续 USD/模型更新以 Hugging Face unitree_model 为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富的刚体物理引擎 |
| URDF | Unified Robot Description Format | 统一机器人描述格式 |
| Sim2Sim | Simulation to Simulation | 跨仿真器策略验证 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |
| SDK | Software Development Kit | 软件开发工具包 |

## 为什么重要

仿真与模型资产决定 Sim2Sim 成本；错误资产代际会让训练/验证结果无法对照真机。

在宇树官方开源地图中，本仓是 **仿真与模型** 的独立节点；与其它仓的选型关系见 [Unitree](./unitree.md)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 仓库 | [`unitreerobotics/unitree_model`](https://github.com/unitreerobotics/unitree_model) |
| 组织分类 | 仿真与模型 |
| 星标（2026-07-24） | ~147 |
| 最近推送 | 2026-06-01 |
| 主要语言 | N/A |
| 维护状态 | **deprecated**（GitHub）；资产改走 Hugging Face |

## 工程实践

- Select Movebale Base in Links
- Select Stiffness in Joint Configuration
- Select Force in Drive Type
- Allow Self-Collision

- 组织级导航与五条研发主线见 [Unitree 软件生态](./unitree.md)；原始归档见 [sources/repos/unitree_model.md](../../sources/repos/unitree_model.md)。

## 局限与风险

- **不要脱离代际混用**：SDK1/ROS1 遗产仓与 SDK2/ROS2/DDS 新栈的消息与启动方式不兼容。
- **README 边界优先**：功能承诺以官方 README 为准；星标高不等于适合你的机型/仿真器。
- **本仓已 deprecated**：勿再以 GitHub `unitree_model` 作为 USD 更新源，改用 Hugging Face 数据集。

## 关联页面

- [Unitree 组织枢纽](./unitree.md)
- [unitree_mujoco](./unitree-mujoco.md)
- [unitree_guide](./unitree-guide.md)
- [unitree_rl_mjlab](./unitree-rl-mjlab.md)
- [Sim2Real](../concepts/sim2real.md)
- [MuJoCo](./mujoco.md)

## 参考来源

- [sources/repos/unitree_model.md](../../sources/repos/unitree_model.md)
- [sources/repos/unitree.md](../../sources/repos/unitree.md) — 组织级仓库地图
- 上游仓库：<https://github.com/unitreerobotics/unitree_model>

## 推荐继续阅读

- 官方开发者文档：<https://support.unitree.com/home/zh/developer>
- 组织总览：<https://github.com/unitreerobotics>
