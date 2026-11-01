---
type: entity
tags: [manipulator, arm, open-source, esp32, desktop]
status: complete
updated: 2026-10-27
project_id: waveshare-roarm-m2
code: https://github.com/waveshareteam/roarm_m2
summary: "RoArm-M2：Waveshare 桌面级约 4 自由度机械臂（3 轴 + 夹爪），ESP32 + 串行总线舵机，<900 g，载荷约 0.5 kg@0.5 m；代码开源、硬件为商业套件。"
related:
  - ./painode-170-soarm100.md
  - ./robotis-open-manipulator-line.md
  - ../tasks/manipulation.md
  - ../overview/robot-open-source-wechat-githubstore-18-projects.md
sources:
  - ../../sources/blogs/wechat_githubstore_18_open_source_robot_projects.md
---

# RoArm-M2

## 一句话定义

**RoArm-M2** 是 Waveshare 的桌面级机械臂，约 4 自由度（3 轴 + 夹爪），仓库 [waveshareteam/roarm_m2](https://github.com/waveshareteam/roarm_m2)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ESP32 | Espressif ESP32 | 低成本 MCU |
| DoF | Degrees of Freedom | 自由度 |
| API | Application Programming Interface | 应用编程接口 |
| SDK | Software Development Kit | 开发工具包 |

## 为什么重要

- 代码开放、价格亲民的桌面臂，补足 SO-ARM100 之外的选择。
- 在微信文章《想进入机器人领域吗？学习这18个开源项目就够了》中排第 **13** 位。

## 核心要点（据文章）

- 基于 ESP32 和串行总线舵机，整体重量小于 900 g，有效载荷约 0.5 kg@0.5 m，结构使用碳纤维和铝合金。
- 控制代码、通信接口和示例完全开源，支持二次开发；硬件以商业套件形式销售。

## 工程实践

- 适合做桌面二次开发和通信接口实验；不应作为开源硬件设计参考。
- 以上规格与许可证均出自文章叙述，**未逐项核对官方仓库**；动手前请以官方 README/BOM 为准。

## 局限与风险

- 本页仅基于一篇第三方公众号文章，规格数字可能随版本变化。
- 仓库迁移、分叉和许可证以官方当前状态为准。

## 关联页面

- [painode-170-soarm100](./painode-170-soarm100.md)
- [robotis-open-manipulator-line](./robotis-open-manipulator-line.md)
- [manipulation](../tasks/manipulation.md)
- [机器人开源项目 18 例（微信 GitHubStore）索引](../overview/robot-open-source-wechat-githubstore-18-projects.md)

## 推荐继续阅读

- [roarm_m2 仓库](https://github.com/waveshareteam/roarm_m2)

## 参考来源

- [wechat_githubstore_18_open_source_robot_projects.md](../../sources/blogs/wechat_githubstore_18_open_source_robot_projects.md)
