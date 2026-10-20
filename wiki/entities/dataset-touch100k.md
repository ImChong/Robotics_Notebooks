---
type: entity
tags:
  - dataset
  - tactile
  - language
  - vision-language
status: draft
updated: 2026-10-20
summary: "约 10 万组触觉-语言-视觉配对数据，提供句子级与短语级多粒度触感描述。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../concepts/tactile-sensing.md
  - ../concepts/visuo-tactile-fusion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# Touch100k

## 一句话定义

**Touch100k**：基于已有视觉触觉数据、经生成与校正得到的触觉语言视觉配对数据集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TLV-Link | 文中预训练方法名 | 课程式链接预训练 |
| GelSight | Gel-based tactile sensor | 视觉触觉传感器 |
| VLM | Vision-Language Model | 视觉语言模型 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | 北京交通大学、腾讯微信 AI、北京邮电大学 |
| 时间 | 2024-06 |
| 模态 | 触觉 + 语言 + 视觉 |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 约 10 万组 |
| 标注流程 | GPT-4V 生成、Gemini 校验、人工校正（文章） |
| 任务 | 材质属性识别、机器人抓取预测 |

## 为什么重要

- 多粒度描述可同时支持自然语言理解与关键触感特征提取，并配套 TLV-Link 预训练方法。

## 核心原理

在 Touch and Go、VisGel 等数据基础上做多阶段生成与校正，并形成课程式训练。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | GitHub 数据页（文章给出链接） |
| 代码 | cocacola-lab/TLV-Link 仓库 |

## 局限与风险

- 标注由大模型生成并校正，描述风格偏向模型语言（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [tactile-sensing](../concepts/tactile-sensing.md)
- [visuo-tactile-fusion](../concepts/visuo-tactile-fusion.md)
- [manipulation](../tasks/manipulation.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://cocacola-lab.github.io/Touch100k/)
