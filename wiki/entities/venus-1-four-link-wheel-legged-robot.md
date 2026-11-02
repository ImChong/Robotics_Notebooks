---
type: entity
project_id: venus-1-four-link-wheel-legged-robot
tags:
  - repo
  - independent
  - wheel-legged
  - four-link
  - stm32
  - hardware
  - pid
status: complete
updated: 2026-11-01
code: https://github.com/track-tech/Venus-1__Four-link-wheel-legged-robot
related:
  - ../comparisons/open-source-qdd-actuator-projects.md
  - ../tasks/locomotion.md
  - ../overview/quadruped-open-source-projects-technology-map.md
sources:
  - ../../sources/repos/venus_1_four_link_wheel_legged_robot.md
  - ../../sources/blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md
summary: "启明-1（Venus-1）是四连杆轮腿机器人的个人本科毕设开源项目：SolidWorks 机械、STM32H723 + FreeRTOS、小米 CyberGear 关节电机、DJI C610 + M2006 轮电机、全 PID 控制、SBUS 遥控。"
---

# 启明-1 Venus-1

**启明-1 Venus-1**（[GitHub](https://github.com/track-tech/Venus-1__Four-link-wheel-legged-robot)）来自 [古月居转载的四足机器人开源项目综述](https://mp.weixin.qq.com/s/pRGoPm41qYCNyHPWR6Dl6A)，属「轮足机器人」组。

## 一句话定义

**启明-1（Venus-1）是四连杆轮腿机器人的个人本科毕设开源项目：SolidWorks 机械、STM32H723 + FreeRTOS、小米 CyberGear 关节电机、DJI C610 + M2006 轮电机、全 PID 控制、SBUS 遥控。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PID | Proportional–Integral–Derivative | 比例积分微分控制 |
| SBUS | Serial Bus | 遥控接收机串行总线 |

## 为什么重要

- 综述把它放在「轮足机器人」这条技术路线下，作为可读源码的参照项目；横向对比见 [四足机器人开源项目技术地图：从极简运动学到研究级硬件（36 个 GitHub 项目）](../overview/quadruped-open-source-projects-technology-map.md)。
- 文中星标：60（文章表述，入库日未复核）。

## 核心信息

| 项 | 内容 |
|----|------|
| **仓库** | [track-tech/Venus-1__Four-link-wheel-legged-robot](https://github.com/track-tech/Venus-1__Four-link-wheel-legged-robot) |
| **最近提交** | 2025-05-09 |
| **可达性** | `git ls-remote` 可达（2026-11-01） |
| **文章归类** | 轮足机器人 |

## 文章说法核对

- **文章：** 文章称是启明-1 四连杆轮腿机器人，完整机械和控制开源。
- **README / 核查：** README 声明这是个人本科毕业设计，**存在设计缺陷，请谨慎复刻**；遵循 GPL 3.0，量产与商用需授权；结构为 3 mm 3240 环氧板，测试含重心自适应、跌落缓冲、主动悬挂、跳跃。
- **差异与保留：** GPL 3.0 且商用需授权；作者自述有设计缺陷。

## 关联页面

- [open-source-qdd-actuator-projects](../comparisons/open-source-qdd-actuator-projects.md)
- [locomotion](../tasks/locomotion.md)
- [quadruped-open-source-projects-technology-map](../overview/quadruped-open-source-projects-technology-map.md)

## 参考来源

- [venus_1_four_link_wheel_legged_robot.md](../../sources/repos/venus_1_four_link_wheel_legged_robot.md)
- [wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md](../../sources/blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md)
- [原文（微信公众号）](https://mp.weixin.qq.com/s/pRGoPm41qYCNyHPWR6Dl6A)
