---
type: entity
tags: [repo, unitree, unitreerobotics, isaac-lab, teleoperation, simulation, imitation-learning]
status: complete
updated: 2026-07-24
related:
  - ./unitree.md
  - ./xr-teleoperate.md
  - ./unitree-lerobot.md
  - ./unitree-rl-lab.md
  - ./unitree-dexterous-hand-services.md
  - ../tasks/teleoperation.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/repos/unitree_sim_isaaclab.md
  - ../../sources/repos/unitree.md
summary: "unitree_sim_isaaclab 在 Isaac Lab 上仿真 G1/H1-2 多执行器任务，用与真机相同的 DDS 主题支持遥操作采数、回放与模型验证；常与 xr_teleoperate 联用。"
---

# unitree_sim_isaaclab

**unitree_sim_isaaclab** 基于 Isaac Lab，为 Unitree **G1 / H1-2**（夹爪、Dex3、Inspire 等执行器组合）提供多任务仿真，服务数据采集、回放、生成与策略验证。

## 一句话定义

「长得像真机 DDS」的 Isaac Lab 仿真场——XR 遥操作或策略节点不用改话题就能在仿真里采数/验模型。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Isaac Lab | NVIDIA Isaac Lab | 仿真与学习框架 |
| DDS | Data Distribution Service | 与真机同构通信 |
| XR | Extended Reality | 常经 xr_teleoperate 接入 |
| DoF | Degrees of Freedom | 如 G1-29dof |
| IL | Imitation Learning | 采数下游用途 |
| Sim2Real | Simulation to Real | 仿真到真机 |

## 为什么重要

- 解决「没真机 / 真机档期不足」时的人形操作数据瓶颈。
- **Wholebody** 任务名表示可移动操作，不仅是固定基座桌面。
- 与 [`unitree_rl_lab`](./unitree-rl-lab.md) 同属 Isaac 生态，但本仓侧重 **遥操作采数与任务场景**，不是 locomotion RL 环境包。

## 核心原理

| 要点 | 说明 |
|------|------|
| 通信 | 启动后收发与真机相同 DDS 主题 |
| 机型×末端 | G1-29dof × Dex1/Dex3/Inspire；H1-2 × Inspire 等 |
| 任务示例 | `Isaac-PickPlace-Cylinder-G129-*-Joint`、`Isaac-PickPlace-RedBlock-...` |
| 权重 | 仓库提供的权重**仅供仿真测试** |

## 工程实践

1. 硬件按 Isaac Lab 官方推荐；已测 GPU：RTX 3080/3090/4090；50 系需 Isaac Sim 5.0.0（上游说明）。
2. 与 [`xr_teleoperate`](./xr-teleoperate.md) 联用采数；DDS 用法对照 SDK2 Python G1 示例与 Dex3 示例。
3. 同网有真机时必须隔离，防止仿真节点误控实机。
4. 首启加载慢属预期；主视角需在 Isaac 中切到 PerspectiveCamera。

## 局限与风险

- 仿真权重 ≠ 真机可部署策略。
- GPU/Isaac 版本矩阵变化快，以 README「Important Notes」为准。
- 执行器（夹爪 vs 多指手）与 [`unitree_lerobot`](./unitree-lerobot.md) 转换脚本必须一致。

## 关联页面

- [xr_teleoperate](./xr-teleoperate.md)
- [unitree_lerobot](./unitree-lerobot.md)
- [unitree_rl_lab](./unitree-rl-lab.md)
- [Teleoperation](../tasks/teleoperation.md)
- [Unitree](./unitree.md)

## 参考来源

- [sources/repos/unitree_sim_isaaclab.md](../../sources/repos/unitree_sim_isaaclab.md)
- 上游：<https://github.com/unitreerobotics/unitree_sim_isaaclab>

## 推荐继续阅读

- 仓内中文 README：`README_zh-CN.md`

