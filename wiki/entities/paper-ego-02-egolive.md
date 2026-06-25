---

type: entity
tags: [paper, egocentric, ego-survey, embodied-ai, jd]
status: complete
updated: 2026-06-25
arxiv: "2604.23570"
code: https://robotdata-market.jdcloud.com/console/market
summary: "大规模真实家政/零售等任务导向 Ego 数据，把「人类视频」推向机器人任务数据。"
related:
  - ../overview/ego-9-papers-technology-map.md
  - ../overview/ego-category-01-data-collection.md
sources:
  - ../../sources/papers/ego_survey_02_egolive.md
  - ../../sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md
  - ../../sources/papers/ego_9_papers_catalog.md
---

# EgoLive

**EgoLive** 收录于 [具身智能研究室 · Ego 9 篇专题](https://mp.weixin.qq.com/s/4JQ1xa-cJ7J1ep_e4txNnA) **第 02/9** 篇，归类为 **01 数据采集**。

## 一句话定义

大规模真实家政/零售等任务导向 Ego 数据，把「人类视频」推向机器人任务数据。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称视角感知与控制 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |

## 为什么重要

- EgoLive 更像是在回答另一个问题：第一视角数据不能只停留在“日常视频”，它要尽量靠近机器人未来要做的任务。
- 论文强调的是大规模、真实场景、任务导向和多模态标注。它覆盖家政、零售等真实工作场景，目标很明确：给机器人学习提供更接近部署环境的数据。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 02/9 |
| 分组 | 01 数据采集 |
| 出处 | 2026 · arXiv |
| 论文/项目 | <https://arxiv.org/abs/2604.23570> |
| 代码/项目 | <https://robotdata-market.jdcloud.com/console/market> |

## 核心机制（归纳）

### 1）策展导读要点

EgoLive 更像是在回答另一个问题：第一视角数据不能只停留在“日常视频”，它要尽量靠近机器人未来要做的任务。

### 2）策展导读要点

论文强调的是大规模、真实场景、任务导向和多模态标注。它覆盖家政、零售等真实工作场景，目标很明确：给机器人学习提供更接近部署环境的数据。

### 3）策展导读要点

我觉得这类数据集有一个重要意义：它把“人类视频”从泛泛的视频资源，往 **机器人任务数据** 方向推了一步。机器人不只是需要看到世界，还要知道一个任务通常怎么开始、怎么推进、哪里容易失败、人在执行时会怎样调整。

## 常见误区

1. Ego 视频不会天然等于机器人策略数据；须经过重建、对齐、重定向与物理过滤。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 技术地图：[ego-9-papers-technology-map.md](../overview/ego-9-papers-technology-map.md)
- 分类 hub：[ego-category-01-data-collection.md](../overview/ego-category-01-data-collection.md)
- 原始 source：[ego_survey_02_egolive.md](../../sources/papers/ego_survey_02_egolive.md)

## 参考来源

- [ego_survey_02_egolive.md](../../sources/papers/ego_survey_02_egolive.md) — Ego 9 篇策展摘录
- [ego_9_papers_catalog.md](../../sources/papers/ego_9_papers_catalog.md) — 9 篇总表
- [wechat_embodied_ai_lab_ego_9_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md) — 微信公众号编译导读
- 论文/项目：<https://arxiv.org/abs/2604.23570>

## 推荐继续阅读

- [Ego 9 篇技术地图](../overview/ego-9-papers-technology-map.md)
- [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
