---
type: entity
tags: [quadruped, biped, hardware, open-source, odri, torque-control, qdd, actuator]
status: complete
updated: 2026-07-25
related:
  - ./quadruped-robot.md
  - ./open-source-humanoid-hardware.md
  - ../comparisons/open-source-qdd-actuator-projects.md
  - ./berkeley-humanoid-lite.md
  - ./moteus.md
  - ../overview/robot-open-source-wechat-issue01-curator.md
  - ../tasks/locomotion.md
  - ../../roadmap/depth-torque-motor-design.md
sources:
  - ../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md
  - ../../sources/repos/open_robot_actuator_hardware.md
  - ../../sources/sites/open_dynamic_robot_initiative.md
  - ../../sources/personal/open_source_qdd_actuator_learning_curator.md
summary: "Open Dynamic Robot Initiative（ODRI）：开源扭矩控制腿足平台（Solo/Bolt）与 open_robot_actuator_hardware 力控关节硬件——结构、行星/皮带减速、驱动 PCB、双编码器、电流/力矩环与测试资料的学术基线。"
---

# ODRI Solo / Bolt（开源腿式平台）

## 一句话定义

**ODRI（Open Dynamic Robot Initiative）** 提供面向研究的开源 **扭矩控制** 腿足平台：**Solo**（四足）与 **Bolt**（双足）常被引作 **低惯量、高带宽力控** 的学术基线；主线入口在 **[open-dynamic-robot-initiative](https://github.com/open-dynamic-robot-initiative)**，关节硬件深读仓为 **[open_robot_actuator_hardware](https://github.com/open-dynamic-robot-initiative/open_robot_actuator_hardware)**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ODRI | Open Dynamic Robot Initiative | 开源力控腿足与执行器硬件倡议 |
| QDD | Quasi-Direct Drive | 准直驱，低减速比、高背驱动性的作动方案 |
| FOC | Field-Oriented Control | 无刷电机的磁场定向控制 |
| CAN | Controller Area Network | 电机/关节常用的现场总线通信协议 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |

## 为什么重要

- **执行器与软件栈解耦**：组织下多仓覆盖驱动、固件、TriFinger 周边与 Solo 等子项目，是 **OCS2 / Pinocchio** 等学术栈常见底层参照（亦见 [开源人形硬件对比](./open-source-humanoid-hardware.md)）。
- **完整开源力控关节**：`open_robot_actuator_hardware` 覆盖机械结构、行星或皮带减速、电机驱动 PCB、编码器、电流环与关节力矩控制、CAN/以太网、装配与测试——是 [开源 QDD 学习路线](../comparisons/open-source-qdd-actuator-projects.md) 中**最优先**的完整关节教材。
- 架构可迁移到小型/中型人形髋、膝、踝原型（原平台为四足，不限于四足形态）。

## 执行器硬件要点（open_robot_actuator_hardware）

| 模块 | 学什么 |
|------|--------|
| 机械 | 关节结构；行星减速或皮带减速 |
| 电机策略 | 低减速比 + 高扭矩密度**成品**外转子无刷（典型 QDD） |
| 电气 | 驱动 PCB、编码器（含双编码器位置测量叙事） |
| 控制 | 电流控制与关节力矩控制 |
| 工程 | 装配步骤、测试方法、热管理、执行器测试台 |

**局限：** 电机本体一般采购现成无刷外转子，**不含完整电磁设计**。

## 开源入口

| 类型 | 链接 |
|------|------|
| 项目门户 | [open-dynamic-robot-initiative.github.io](https://open-dynamic-robot-initiative.github.io) |
| 执行器硬件仓 | [open_robot_actuator_hardware](https://github.com/open-dynamic-robot-initiative/open_robot_actuator_hardware)（BSD-3-Clause） |
| 架构论文 | [arXiv:1910.00093](https://arxiv.org/abs/1910.00093) |
| 组织 GitHub | [open-dynamic-robot-initiative](https://github.com/open-dynamic-robot-initiative) |

## 关联页面

- [开源 QDD 执行器项目对比](../comparisons/open-source-qdd-actuator-projects.md)
- [执行器驱动链选型闭环](../queries/actuator-drive-chain-selection-loop.md)
- [Berkeley Humanoid Lite](./berkeley-humanoid-lite.md)
- [四足机器人](./quadruped-robot.md)
- [开源人形硬件方案对比](./open-source-humanoid-hardware.md)
- [Locomotion](../tasks/locomotion.md)
- [力矩电机设计纵深](../../roadmap/depth-torque-motor-design.md)

## 推荐继续阅读

- 执行器硬件仓 README 与装配/测试文档（以当前主推版本为准）

## 参考来源

- [open_robot_actuator_hardware.md](../../sources/repos/open_robot_actuator_hardware.md)
- [open_dynamic_robot_initiative.md](../../sources/sites/open_dynamic_robot_initiative.md)
- [开源 QDD 执行器学习策展](../../sources/personal/open_source_qdd_actuator_learning_curator.md)
- [wechat_jixie_robot_open_source_treasury_issue01_10_robots.md](../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md)
