---

type: entity
tags: [paper, egocentric, ego-survey, embodied-ai, stanford]
status: complete
updated: 2026-06-25
arxiv: "2605.30282"
code: https://zuo-kuangji.github.io/Gaze2Act/
summary: "将人类第一视角 gaze 映射到机器人视角，作为 VLA 条件输入，补语言难以精确描述的空间意图。"
related:
  - ../overview/ego-9-papers-technology-map.md
  - ../overview/ego-category-02-human-to-robot.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/ego_survey_05_gaze2act.md
  - ../../sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md
  - ../../sources/papers/ego_9_papers_catalog.md
---

# Gaze2Act

**Gaze2Act** 收录于 [具身智能研究室 · Ego 9 篇专题](https://mp.weixin.qq.com/s/4JQ1xa-cJ7J1ep_e4txNnA) **第 05/9** 篇，归类为 **02 人→机器人**。

## 一句话定义

将人类第一视角 gaze 映射到机器人视角，作为 VLA 条件输入，补语言难以精确描述的空间意图。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| Ego | Egocentric Vision | 第一人称视角感知与控制 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |

## 为什么重要

- 很多时候，人不一定能把意图说清楚。桌上有几个相似杯子，语言很难精确描述“拿左前方那个杯子的边缘”；但人看向哪里，往往已经透露了目标和动作区域。
- Gaze2Act 做的事情，是把人的第一视角 gaze 映射到机器人视角里，再作为 VLA 策略的条件输入。这样一来，人和机器人之间的交互就不只靠语言，而多了一条更自然的意图通道。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 05/9 |
| 分组 | 02 人→机器人 |
| 出处 | 2026 · arXiv |
| 论文/项目 | <https://arxiv.org/abs/2605.30282> |
| 代码/项目 | <https://zuo-kuangji.github.io/Gaze2Act/> |

## 核心机制（归纳）

### 1）策展导读要点

很多时候，人不一定能把意图说清楚。桌上有几个相似杯子，语言很难精确描述“拿左前方那个杯子的边缘”；但人看向哪里，往往已经透露了目标和动作区域。

### 2）策展导读要点

Gaze2Act 做的事情，是把人的第一视角 gaze 映射到机器人视角里，再作为 VLA 策略的条件输入。这样一来，人和机器人之间的交互就不只靠语言，而多了一条更自然的意图通道。

### 3）策展导读要点

对于机器人来说，真正难的常常不是“听懂拿杯子”，而是知道到底拿哪个、从哪里接近、什么时候目标变了。视线就是一种低负担、高密度的信号。

## 常见误区

1. Ego 视频不会天然等于机器人策略数据；须经过重建、对齐、重定向与物理过滤。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 技术地图：[ego-9-papers-technology-map.md](../overview/ego-9-papers-technology-map.md)
- 分类 hub：[ego-category-02-human-to-robot.md](../overview/ego-category-02-human-to-robot.md)
- 原始 source：[ego_survey_05_gaze2act.md](../../sources/papers/ego_survey_05_gaze2act.md)

## 参考来源

- [ego_survey_05_gaze2act.md](../../sources/papers/ego_survey_05_gaze2act.md) — Ego 9 篇策展摘录
- [ego_9_papers_catalog.md](../../sources/papers/ego_9_papers_catalog.md) — 9 篇总表
- [wechat_embodied_ai_lab_ego_9_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md) — 微信公众号编译导读
- 论文/项目：<https://arxiv.org/abs/2605.30282>

## 推荐继续阅读

- [Ego 9 篇技术地图](../overview/ego-9-papers-technology-map.md)
- [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
