---
type: entity
project_id: legged-control-ocs2
tags:
  - repo
  - independent
  - quadruped
  - mpc
  - wbc
  - ocs2
  - ros
status: complete
updated: 2026-11-01
code: https://github.com/qiayuanl/legged_control
related:
  - ../entities/ocs2.md
  - ../methods/model-predictive-control.md
  - ../concepts/whole-body-control.md
  - ../tasks/locomotion.md
  - ../overview/quadruped-open-source-projects-technology-map.md
sources:
  - ../../sources/repos/legged_control_ocs2.md
  - ../../sources/blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md
summary: "legged_control 是基于 OCS2 与 ros-control 的足式机器人 NMPC + WBC + 状态估计 + sim2real 框架（作者 Qiayuan Liao 等）。"
---

# legged_control

**legged_control**（[GitHub](https://github.com/qiayuanl/legged_control)）来自 [古月居转载的四足机器人开源项目综述](https://mp.weixin.qq.com/s/pRGoPm41qYCNyHPWR6Dl6A)，属「优化控制（MPC / WBC）」组。

## 一句话定义

**legged_control 是基于 OCS2 与 ros-control 的足式机器人 NMPC + WBC + 状态估计 + sim2real 框架（作者 Qiayuan Liao 等）。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OCS2 | Optimal Control for Switched Systems | ETH 切换系统最优控制库 |
| NMPC | Nonlinear Model Predictive Control | 非线性模型预测控制 |
| WBC | Whole-Body Control | 全身控制 |

## 为什么重要

- 综述把它放在「优化控制（MPC / WBC）」这条技术路线下，作为可读源码的参照项目；横向对比见 [四足机器人开源项目技术地图：从极简运动学到研究级硬件（36 个 GitHub 项目）](../overview/quadruped-open-source-projects-technology-map.md)。
- 文中星标：（文中未给）（文章表述，入库日未复核）。

## 核心信息

| 项 | 内容 |
|----|------|
| **仓库** | [qiayuanl/legged_control](https://github.com/qiayuanl/legged_control) |
| **最近提交** | 2025-02-13 |
| **可达性** | `git ls-remote` 可达（2026-11-01） |
| **文章归类** | 优化控制（MPC / WBC） |

## 文章说法核对

- **文章：** 文章称其在 Gazebo 中运行，包含 EKF/互补滤波状态估计、基于 SRBD 的 MPC、WBC 执行器与步态调度器；要求 ROS1 Noetic / Ubuntu 20.04。
- **README / 核查：** README 标题引用为「NMPC, WBC, state estimation, and sim2real framework for legged robots based on OCS2 and ros-controls」；顶部声明**软件已不再维护**，作者转向全新框架，并推荐带感知的 legged_perceptive。
- **差异与保留：** 文章未提「不再维护」；SRBD / SQP 等细节来自文章，未在 README 中逐项核对。

## 关联页面

- [ocs2](../entities/ocs2.md)
- [model-predictive-control](../methods/model-predictive-control.md)
- [whole-body-control](../concepts/whole-body-control.md)
- [locomotion](../tasks/locomotion.md)
- [quadruped-open-source-projects-technology-map](../overview/quadruped-open-source-projects-technology-map.md)

## 参考来源

- [legged_control_ocs2.md](../../sources/repos/legged_control_ocs2.md)
- [wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md](../../sources/blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md)
- [原文（微信公众号）](https://mp.weixin.qq.com/s/pRGoPm41qYCNyHPWR6Dl6A)
