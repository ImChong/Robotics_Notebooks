---
type: entity
title: Gen-HumanWholeBody（简智机器人全身操作数据集）
project_id: gen-human-whole-body
project: https://huggingface.co/datasets/genrobot2025/Gen-HumanWholeBody
tags: [dataset, human-motion, whole-body, genrobot, jzrobot, imitation-learning]
summary: "简智机器人的全身操作数据集，官方门户标注 600 小时和约 3 cm 全身重建 MPJPE。"
updated: 2026-10-27
status: draft
institutions:
  - jzrobot
related:
  - ./gen-human-ego-dataset.md
  - ../concepts/motion-retargeting.md
sources:
  - ../../sources/datasets/gen-human-whole-body-genrobot.md
  - ../../sources/sites/genrobot-open-dataset.md
---

# Gen-HumanWholeBody（全身操作数据集）

## 一句话定义

Gen-HumanWholeBody 是简智机器人发布的全身操作数据集，官方门户标注 **600 小时**及重建 **MPJPE 约 3 cm**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MPJPE | Mean Per Joint Position Error | 关节点平均位置误差 |
| HF | Hugging Face | 数据集托管平台 |
| IL | Imitation Learning | 从示范中学习动作的模仿学习 |

## 为什么重要

全身操作示范可为全身运动学习与人类技能向机器人迁移提供研究素材；实际使用时还需要处理人和机器人形态差异，参见[运动重定向](../concepts/motion-retargeting.md)。同机构的 [Gen-HumanEgo](./gen-human-ego-dataset.md) 主要面向第一视角手部与场景数据，两者应按各自数据卡确认模态后选用。

## 数据与使用边界

| 项目 | 官方门户信息 |
|------|--------------|
| 时长 | 600 小时 |
| 重建精度 | MPJPE 约 3 cm；具体测试条件待查数据卡 |
| 获取 | [Hugging Face 数据集](https://huggingface.co/datasets/genrobot2025/Gen-HumanWholeBody) |
| 代码 | 官方数据门户未列出本数据集专属训练代码 |

数据卡本次无法读取，许可、访问条件、episode 数量、场景域、数据模态和精度评测细节尚待核实。不要将人类动作直接视作机器人关节指令；迁移时应检查坐标系、身体形态和接触约束。

## 关联页面

- [Gen-HumanEgo](./gen-human-ego-dataset.md)
- [运动重定向](../concepts/motion-retargeting.md)

## 参考来源

- [Gen-HumanWholeBody 数据集资料归档](../../sources/datasets/gen-human-whole-body-genrobot.md)
- [GenRobot 开放数据门户归档](../../sources/sites/genrobot-open-dataset.md)

## 推荐继续阅读

- [官方数据门户](https://www.genrobot.ai/data/open-dataset)
- [Hugging Face 数据集](https://huggingface.co/datasets/genrobot2025/Gen-HumanWholeBody)
