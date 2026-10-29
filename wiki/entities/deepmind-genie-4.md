---
type: entity
project_id: deepmind-genie-4
project: https://deepmind.google/models/genie/
tags: [world-model, generative-model, simulation, robotics, google-deepmind]
status: complete
updated: 2026-10-28
summary: "Google DeepMind Genie 4 是实时生成交互画面与音频的世界模型；Gemini 可在循环中规划并引导场景，机器人训练应用尚属早期探索。"
related:
  - ../concepts/functional-taxonomy-world-models.md
  - ../concepts/video-as-simulation.md
  - ../methods/generative-world-models.md
  - ../overview/robot-world-models-training-loop-taxonomy.md
sources:
  - ../../sources/blogs/deepmind_genie_4_world_simulation.md
---

# Genie 4（Google DeepMind · 交互世界模拟）

## 一句话定义

Genie 4 是实时生成交互画面与音频的世界模型；Gemini 可观察生成画面、规划步骤并以文本引导模拟。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 生成可交互环境的世界模型 |
| AI | Artificial Intelligence | 通过环境交互学习或执行任务的智能系统 |
| API | Application Programming Interface | 官方计划面向早期合作伙伴提供的接入方式 |

## 核心机制与能力

| 部分 | 官方描述 |
|------|----------|
| 交互生成 | 键盘导航和文本提示实时改变场景；同步生成音乐、环境声和对话 |
| Gemini Omni | 为模拟提供现实世界外观和行为基础 |
| Gemini 3.8 Flash | 在循环中观察画面、分解目标、核查步骤并调整计划 |
| Gemini 4 Argon | 机器人探索示例中将任务分解成文本步骤，供 Genie 4 渲染 |

官方展示 Apollo 人形机器人下跳棋的模拟，说明文本分步引导生成的可能性；文章没有报告机器人策略训练、真机迁移或任务成功率。按[世界模型功能分类](../concepts/functional-taxonomy-world-models.md)理解，已展示的输出主要是生成观测和交互，不能由视觉逼真推断其提供可查询的物理状态。

## 工程实践与局限

- 研究预览先向可信测试者开放，后续计划开放给 Project Genie 的 Google AI Ultra 用户与 Gemini API / AI Studio 早期合作伙伴。
- 截至 2026-10-28，博客和模型页未列训练或推理代码、权重及数据集下载链接；未确认可复现的部署接口。
- 官方指出视觉细节与物体持久性可能漂移，复杂物理、空间精度、音频准确性与长时一致性仍在研究；模型页说明连续交互为数分钟，演示片段经过剪辑。
- 机器人训练和评测是潜在用途，实际使用仍需验证动作接口、接触动力学、策略闭环效果以及与真机的相关性。

## 关联页面

- [世界模型功能分类](../concepts/functional-taxonomy-world-models.md) — 区分观测、状态与动作输出
- [Video-as-Simulation](../concepts/video-as-simulation.md) — 视频生成与模拟的关系
- [生成式世界模型](../methods/generative-world-models.md) — 相关方法
- [机器人世界模型训练闭环](../overview/robot-world-models-training-loop-taxonomy.md) — 训练和评测接口

## 参考来源

- [DeepMind 博客归档](../../sources/blogs/deepmind_genie_4_world_simulation.md)

## 推荐继续阅读

- [Genie 4 官方博客](https://deepmind.google/blog/genie-4-a-new-generation-of-world-simulation/)
- [Genie 4 官方模型页](https://deepmind.google/models/genie/)
