---
type: entity
tags: [repo, humanoid, unitree-g1, motion-tracking, teleoperation, sim2real, onnx, mujoco]
status: complete
updated: 2026-10-13
project_id: grit-v0-0-1
code: https://github.com/mrzuang/GRIT_teleop_deploy
summary: "GRIT v0.0.1 是一个全身人形动作跟踪策略发布；配套 G1 部署仓库提供 50 Hz ONNX runtime、MuJoCo sim2sim、PICO 遥操作与 SDK2 真机桥接。"
related:
  - ../concepts/sim2real.md
  - ../tasks/teleoperation.md
  - ./grit-deploy-onboard-g1.md
  - ./egalahad-sim2real.md
sources:
  - ../../sources/repos/grit-v0-0-1-deployment.md
---

# GRIT v0.0.1

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| ONNX | Open Neural Network Exchange | 发布策略模型格式 |
| UDP | User Datagram Protocol | 运行时统一通信接口 |
| SDK2 | Unitree SDK 2 | G1 真机通信接口 |
| PICO | PICO VR | 全身遥操作动作输入设备 |
| G1 | Unitree G1 | 部署目标人形机器人 |

[GRIT v0.0.1](https://github.com/mrzuang/GRIT_teleop_deploy) 是面向 Unitree G1 的全身动作跟踪策略发布及部署工程。仓库包含 ONNX policy 与部署 contract、50 Hz Python 推理/控制 runtime、MuJoCo sim2sim、PICO/XRoboToolkit 在线动作重定向，以及基于 Unitree SDK2 的真机 bridge，并设置 stale-command watchdog。

## 运行链路

```mermaid
flowchart LR
  A[PICO / motion reference] --> B[Retargeting / tracking command]
  B --> C[50 Hz ONNX policy runtime]
  C --> D[UDP command interface]
  D --> E[MuJoCo sim2sim / G1 SDK2 bridge]
```

README 将训练与数据生成明确列为仓库外部范围。硬件路径要求 G1 专用网络接口；PICO 路径需 XRoboToolkit、低延迟 Wi-Fi。上线前应先用 MuJoCo 验证 policy contract 和控制配置。它与本库中另一个 [GRIT Deploy Onboard G1](./grit-deploy-onboard-g1.md) 部署仓库分开记录，避免混淆代码来源。该策略出现在 [sim2real leaderboard](./egalahad-sim2real.md) 的 MotionDecode 评测中。
## 关联页面

- [GRIT Deploy Onboard G1](./grit-deploy-onboard-g1.md) — 另一套部署实现，仓库来源不同。
- [EGalahad/sim2real leaderboard](./egalahad-sim2real.md) — 该策略的统一动作跟踪评测入口。
