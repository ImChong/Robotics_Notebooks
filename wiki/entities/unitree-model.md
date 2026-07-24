---
type: entity
tags: [repo, unitree, unitreerobotics, assets, usd, deprecated, isaac-lab]
status: complete
updated: 2026-07-24
related:
  - ./unitree.md
  - ./unitree-rl-lab.md
  - ./unitree-ros.md
  - ./unitree-mujoco.md
sources:
  - ../../sources/repos/unitree_model.md
  - ../../sources/repos/unitree.md
summary: "GitHub unitree_model 已官方 deprecated；后续机器人 3D/USD 模型更新以 Hugging Face unitreerobotics/unitree_model 为准。URDF 仍大量来自 unitree_ros。"
---

# unitree_model（已弃用 → Hugging Face）

GitHub 仓库 **unitree_model** 曾提供多仿真环境的 Unitree 3D 模型；上游 README 以醒目声明标注：**本仓库已 deprecated，未来更新发布在 Hugging Face**。

## 一句话定义

模型资产迁移指针页——新项目不要再把 GitHub `unitree_model` 当更新源，改用 HF 数据集，并按需从 `unitree_ros` 取 URDF。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| USD | Universal Scene Description | Isaac 常用资产格式 |
| URDF | Unified Robot Description Format | 来自 unitree_ros |
| HF | Hugging Face | 现行模型托管 |
| Isaac Sim | NVIDIA Isaac Sim | URDF→USD 导入常用工具 |
| MJCF | MuJoCo XML Format | 另见 unitree_mujoco 内模型 |
| RL | Reinforcement Learning | Lab 训练依赖正确资产路径 |

## 为什么重要

- 避免克隆过时 GitHub 仓导致 USD 与文档不一致。
- [`unitree_rl_lab`](./unitree-rl-lab.md) 明确可从 HF `unitree_model` 或 `unitree_ros` URDF 取资产。
- 本页存在的价值是 **迁移动线**，不是继续维护 GitHub 树。

## 核心原理

| 来源 | 用途 |
|------|------|
| HF [`unitreerobotics/unitree_model`](https://huggingface.co/datasets/unitreerobotics/unitree_model) | USD 等后续更新 |
| [`unitree_ros`](./unitree-ros.md) | URDF/xacro 描述 |
| [`unitree_mujoco`](./unitree-mujoco.md) 内 `unitree_robots/` | MJCF |

上游仍给出 Isaac Sim **Direct Import** URDF 建议（Movable Base、Stiffness、Force、Allow Self-Collision），因部分版本 Python 导入脚本有 bug。

## 工程实践

1. `git clone https://huggingface.co/datasets/unitreerobotics/unitree_model`（或按 HF 网页指引）。
2. 在 `unitree_rl_lab` 中设置 `UNITREE_MODEL_DIR`。
3. 若仅需 URDF，克隆 `unitree_ros` 并设置 `UNITREE_ROS_DIR`。

## 局限与风险

- GitHub 仓内容可能冻结；发现不一致时以 HF 与 `unitree_ros` 为准。
- 坐标/关节命名跨 MuJoCo / Isaac / ROS 仍需人工对齐。

## 关联页面

- [unitree_rl_lab](./unitree-rl-lab.md)
- [unitree_ros](./unitree-ros.md)
- [unitree_mujoco](./unitree-mujoco.md)
- [Unitree](./unitree.md)

## 参考来源

- [sources/repos/unitree_model.md](../../sources/repos/unitree_model.md)
- HF：<https://huggingface.co/datasets/unitreerobotics/unitree_model>
- 上游（deprecated）：<https://github.com/unitreerobotics/unitree_model>

## 推荐继续阅读

- Isaac Sim import_urdf 教程（版本号以上游链接为准）

