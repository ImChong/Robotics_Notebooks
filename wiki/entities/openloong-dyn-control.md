---
type: entity
tags: [repo, humanoid, mpc, wbc, mujoco, openloong]
status: complete
updated: 2026-05-27
related:
  - ./openloong.md
  - ../concepts/mpc-wbc-integration.md
  - ../overview/navigation-slam-autonomy-stack.md
sources:
  - ../../sources/repos/openloong_dyn_control.md
summary: "OpenLoong-Dyn-Control 是青龙人形 MuJoCo 动力学控制包：MPC+WBC 行走/跳跃/盲踩，与 OpenLoong Framework 并行。"
---

# OpenLoong-Dyn-Control

**OpenLoong-Dyn-Control** 提供青龙人形在 **MuJoCo** 上的 **MPC + 全身控制** 研究与仿真 demo。

## 为什么重要

- **与导航栈区分**：本仓属 **腿式人形运控**，非 Nav2/SLAM。
- **与 Framework 分工**：Framework 偏实机 C++ 全链；Dyn-Control 偏算法复现与论文级 demo。

## 核心结构/机制

| 能力 | 说明 |
|------|------|
| **MPC** | 质心/足端规划 |
| **WBC** | 关节力矩分配 |
| **MuJoCo** | 仿真验证与盲踩障碍 |

## 常见误区或局限

- **勿与 Autoware/Nav2 混栈** — 输出为关节/力矩级控制，非 `cmd_vel`。
- **硬件**：实机需配合 [OpenLoong](./openloong.md) Framework 与 EtherCAT 驱动。

## 参考来源

- [sources/repos/openloong_dyn_control.md](../../sources/repos/openloong_dyn_control.md)
- [loongOpen/OpenLoong-Dyn-Control](https://github.com/loongOpen/OpenLoong-Dyn-Control)

## 关联页面

- [OpenLoong](./openloong.md)
- [MPC 与 WBC 集成](../concepts/mpc-wbc-integration.md)
- [导航·SLAM·自动驾驶栈总览](../overview/navigation-slam-autonomy-stack.md)

## 推荐继续阅读

- https://www.openloong.org.cn/pages/api/html/index.html
