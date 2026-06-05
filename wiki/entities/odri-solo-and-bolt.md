---
type: entity
tags: [quadruped, biped, hardware, open-source, odri, torque-control]
status: complete
updated: 2026-05-18
related:
  - ./quadruped-robot.md
  - ./open-source-humanoid-hardware.md
  - ../overview/robot-open-source-wechat-issue01-curator.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md
summary: "Open Dynamic Robot Initiative（ODRI）：开源扭矩控制模块化腿足平台，典型代表为四足 Solo 与双足 Bolt；组织 GitHub 聚合多仓软硬件与仿真资源。"
---

# ODRI Solo / Bolt（开源腿式平台）

## 一句话定义

**ODRI（Open Dynamic Robot Initiative）** 提供面向研究的开源 **扭矩控制** 腿足平台：**Solo**（四足）与 **Bolt**（双足）常被引作 **低惯量、高带宽力控** 的学术基线；主线入口在 **[open-dynamic-robot-initiative](https://github.com/open-dynamic-robot-initiative)** 组织。

## 为什么重要

- **执行器与软件栈解耦**：组织下多仓覆盖驱动、固件、TriFinger 周边与 Solo 等子项目，是 **OCS2 / Pinocchio** 等学术栈常见底层参照（亦见 [开源人形硬件对比](./open-source-humanoid-hardware.md) 中的 ODRI 简述）。

## 开源入口（策展摘录）

| 类型 | 链接 |
|------|------|
| 架构论文（转载 PDF） | [arXiv:1910.00093](https://arxiv.org/abs/1910.00093) |
| 组织 GitHub | [open-dynamic-robot-initiative](https://github.com/open-dynamic-robot-initiative) |

## 关联页面

- [四足机器人](./quadruped-robot.md)
- [开源人形硬件方案对比](./open-source-humanoid-hardware.md)
- [Locomotion](../tasks/locomotion.md)
- [机器人开源宝库（微信策展第01期）索引](../overview/robot-open-source-wechat-issue01-curator.md)

## 推荐继续阅读

- ODRI 组织首页与各子仓库 README（以当前主推仓为准）

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |

## 参考来源

- [wechat_jixie_robot_open_source_treasury_issue01_10_robots.md](../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md)
