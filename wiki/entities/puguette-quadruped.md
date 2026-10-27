---
type: entity
tags: [quadruped, open-source, bldc, actuator, 3d-printing]
status: complete
updated: 2026-10-27
project_id: puguette-quadruped
code: https://github.com/haruto89610/puguette
summary: "Puguette：日本开源 12 自由度四足，BLDC + 行星齿轮箱，单关节输出约 55 Nm，执行器可全 3D 打印，MIT。"
related:
  - ./quadruped-robot.md
  - ./stanford-doggo-and-pupper.md
  - ../comparisons/open-source-qdd-actuator-projects.md
  - ../queries/actuator-drive-chain-selection-loop.md
  - ../overview/robot-open-source-wechat-githubstore-18-projects.md
sources:
  - ../../sources/blogs/wechat_githubstore_18_open_source_robot_projects.md
---

# Puguette

## 一句话定义

**Puguette** 是日本开源 12 自由度四足机器人，尺寸约 80×40×60 cm（腿伸直时），仓库 [haruto89610/puguette](https://github.com/haruto89610/puguette)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BLDC | Brushless DC Motor | 无刷直流电机 |
| DoF | Degrees of Freedom | 自由度 |
| Nm | Newton metre | 力矩单位 |
| MIT | MIT License | 宽松开源许可证 |

## 为什么重要

- 把「3D 打印行星减速器 + BLDC」做成整机，是 QDD/高扭矩执行器学习的整机案例。
- 在微信文章《想进入机器人领域吗？学习这18个开源项目就够了》中排第 **11** 位。

## 核心要点（据文章）

- 使用 BLDC 电机配合行星齿轮箱，单关节输出扭矩约 55 Nm。
- 执行器可全 3D 打印，方便调整齿数改变减速比；MIT 许可，提供模型文件和说明。

## 工程实践

- 适合想研究高扭矩执行器设计的人：可通过改齿数试验不同减速比。
- 以上规格与许可证均出自文章叙述，**未逐项核对官方仓库**；动手前请以官方 README/BOM 为准。

## 局限与风险

- 本页仅基于一篇第三方公众号文章，规格数字可能随版本变化。
- 仓库迁移、分叉和许可证以官方当前状态为准。

## 关联页面

- [quadruped-robot](./quadruped-robot.md)
- [stanford-doggo-and-pupper](./stanford-doggo-and-pupper.md)
- [open-source-qdd-actuator-projects](../comparisons/open-source-qdd-actuator-projects.md)
- [执行器驱动链选型闭环](../queries/actuator-drive-chain-selection-loop.md)
- [机器人开源项目 18 例（微信 GitHubStore）索引](../overview/robot-open-source-wechat-githubstore-18-projects.md)

## 推荐继续阅读

- [puguette 仓库](https://github.com/haruto89610/puguette)

## 参考来源

- [wechat_githubstore_18_open_source_robot_projects.md](../../sources/blogs/wechat_githubstore_18_open_source_robot_projects.md)
