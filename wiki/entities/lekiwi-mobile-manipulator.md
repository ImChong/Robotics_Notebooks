---
type: entity
tags: [mobile-manipulation, open-source, lerobot, imitation-learning, raspberry-pi]
status: complete
updated: 2026-10-27
project_id: lekiwi-mobile-manipulator
code: https://github.com/SIGRobotics-UIUC/LeKiwi
summary: "LeKiwi：低成本开源移动操作机器人，全向三轮（Kiwi）底盘 + SO-ARM 机械臂，Raspberry Pi，原生支持 LeRobot，Apache-2.0。"
related:
  - ./lerobot.md
  - ./painode-170-soarm100.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
  - ../overview/robot-open-source-wechat-githubstore-18-projects.md
sources:
  - ../../sources/blogs/wechat_githubstore_18_open_source_robot_projects.md
---

# LeKiwi

## 一句话定义

**LeKiwi** 是低成本开源移动操作机器人，仓库 [SIGRobotics-UIUC/LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BOM | Bill of Materials | 物料清单 |
| LeRobot | Hugging Face LeRobot | 开源机器人学习框架 |
| IL | Imitation Learning | 模仿学习 |
| DoF | Degrees of Freedom | 自由度 |

## 为什么重要

- LeRobot 生态里把 SO-ARM 扩展到移动操作的低成本样板。
- 在微信文章《想进入机器人领域吗？学习这18个开源项目就够了》中排第 **15** 位。

## 核心要点（据文章）

- 全向三轮（Kiwi）底盘 + SO-ARM 系列机械臂，使用 Raspberry Pi，原生支持 LeRobot（遥操作、数据采集、模仿学习）。
- 仓库提供 BOM、3D 打印文件、组装指南和软件说明；Apache-2.0。

## 工程实践

- 适合同时学习移动与操作：先装 SO-ARM 与底盘，再用 LeRobot 采集数据训练模仿学习策略。
- 以上规格与许可证均出自文章叙述，**未逐项核对官方仓库**；动手前请以官方 README/BOM 为准。

## 局限与风险

- 本页仅基于一篇第三方公众号文章，规格数字可能随版本变化。
- 仓库迁移、分叉和许可证以官方当前状态为准。

## 关联页面

- [lerobot](./lerobot.md)
- [painode-170-soarm100](./painode-170-soarm100.md)
- [manipulation](../tasks/manipulation.md)
- [imitation-learning](../methods/imitation-learning.md)
- [机器人开源项目 18 例（微信 GitHubStore）索引](../overview/robot-open-source-wechat-githubstore-18-projects.md)

## 推荐继续阅读

- [LeKiwi 仓库](https://github.com/SIGRobotics-UIUC/LeKiwi)

## 参考来源

- [wechat_githubstore_18_open_source_robot_projects.md](../../sources/blogs/wechat_githubstore_18_open_source_robot_projects.md)
