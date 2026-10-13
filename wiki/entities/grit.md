---
type: entity
tags: [humanoid, motion-tracking, teleoperation, sim2sim, sim2real, unitree-g1, open-source]
status: complete
updated: 2026-10-13
project_id: grit-v0-0-1
project: https://github.com/mrzuang/GRIT_teleop_deploy
code: https://github.com/mrzuang/GRIT_teleop_deploy
summary: "GRIT v0.0.1 是面向 Unitree G1 的全身动作跟踪策略部署项目，公开 ONNX 推理、MuJoCo sim2sim、PICO 遥操作和 SDK2 机载接口；训练与数据生成代码不在仓库范围内。"
related:
  - ./sim2real.md
  - ./grit-deploy-onboard-g1.md
  - ../concepts/sim2real.md
  - ../tasks/teleoperation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/repos/grit_teleop_deploy.md
  - ../../sources/sites/egalahad-sim2real.md
---

# GRIT v0.0.1

**一句话定义：** [GRIT_teleop_deploy](https://github.com/mrzuang/GRIT_teleop_deploy) 发布 Unitree G1 全身运动跟踪策略的部署实现，以 ONNX policy 驱动 MuJoCo 仿真、PICO 实时遥操作和原生 SDK2 真机桥接。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GRIT | 项目名称 | 榜单中的 G1 全身运动跟踪策略与其部署仓库 |
| ONNX | Open Neural Network Exchange | GRIT 推理模型格式 |
| SDK2 | Unitree SDK2 | G1 真机通信与低层控制接口 |
| PICO | PICO VR headset | 为实时全身遥操作提供人体动作输入 |
| WBC | Whole-Body Control | 该项目面向的多关节全身跟踪控制任务 |

## 为什么重要

- 同一策略通过离线参考动作和实时 VR 输入复用部署入口，适合研究者检查动作参考、策略输入和底层执行之间的接口。
- 发布的 MuJoCo sim2sim、Python 推理进程与 SDK2 bridge 可作为复现和部署起点。
- 它在 [sim2real Motion Tracking Leaderboard](./sim2real.md) 中单列为 GRIT v0.0.1，成为跨代码库基准中的一个可比较实现。

## 运行链路

| 环节 | 官方实现 | 作用 |
|---|---|---|
| 策略与 observation | sim2real/src/runtime/grit_policy.py、grit_observation.py | 定义 ONNX 输入输出和观测构造 |
| 推理控制 | sim2real/src/deploy.py | 以 50 Hz 执行策略与控制循环 |
| 仿真 | sim2real/src/sim2sim.py | MuJoCo G1 仿真端 |
| 遥操作 | sim2real/teleop/ | XRoboToolkit / PICO 动作流与重定向 |
| 真机桥接 | g1_sim2real/ | Unitree SDK2 硬件接口；提供 stale-command watchdog |

## 开源状态与局限

- **已开源：** 代码包含 GRIT ONNX 策略及部署契约、50 Hz Python runtime、MuJoCo sim2sim、PICO 遥操作和 SDK2 硬件桥。
- **训练 / 数据：** README 明确指出训练和数据生成不在仓库范围内，因此目前可按已发布接口复现部署与评测，不能仅凭此仓库复现训练。
- **发布计划：** README 中列出的 v0.0.2 / v1.0.0 是计划清单；除非仓库 release 或代码已经提供，不应视为已交付。
- **工程风险：** 真机控制需要匹配机器人关节顺序、策略 observation 合同、通信端口与固件；应先在 MuJoCo 测试，再做受保护的硬件验证。

## 关联页面

- [sim2real 运行栈与排行榜](./sim2real.md)
- [GRIT Deploy Onboard G1](./grit-deploy-onboard-g1.md) — 另一仓库的 G1 多策略机载集成实现
- [Sim2Real 工程概念](../concepts/sim2real.md)
- [Teleoperation](../tasks/teleoperation.md)
- [Locomotion](../tasks/locomotion.md)

## 参考来源

- [GRIT_teleop_deploy 仓库归档](../../sources/repos/grit_teleop_deploy.md)
- [GRIT_teleop_deploy README](https://github.com/mrzuang/GRIT_teleop_deploy)
- [动作跟踪排行榜](https://egalahad.github.io/sim2real/leaderboard/)

## 推荐继续阅读

- [GRIT v0.0.1 仓库](https://github.com/mrzuang/GRIT_teleop_deploy)
- [GRIT 上游模拟与部署框架](https://github.com/EGalahad/sim2real)
- [XRoboToolkit](https://github.com/BoosterRobotics/XRoboToolkit-PC-Service-Pybind)
