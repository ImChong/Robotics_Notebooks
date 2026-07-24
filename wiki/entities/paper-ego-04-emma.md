---

type: entity
tags: [paper, egocentric, ego-survey, embodied-ai, georgia-tech]
status: complete
updated: 2026-07-24
arxiv: "2509.04443"
code: https://ego-moma.github.io/
summary: "人类移动操作 Ego 数据 + 静态机器人数据共训，绕开大规模移动机器人遥操作成本。"
related:
  - ../overview/ego-9-papers-technology-map.md
  - ../overview/ego-category-02-human-to-robot.md
sources:
  - ../../sources/papers/ego_survey_04_emma.md
  - ../../sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md
  - ../../sources/papers/ego_9_papers_catalog.md
---

# EMMA

**EMMA** 收录于 [具身智能研究室 · Ego 9 篇专题](https://mp.weixin.qq.com/s/4JQ1xa-cJ7J1ep_e4txNnA) **第 04/9** 篇，归类为 **02 人→机器人**。

> **命名注意：** 勿与 Waymo 驾驶端到端模型 [EMMA（End-to-End Multimodal Model for Autonomous Driving）](./paper-emma-waymo-e2e.md)（arXiv:2410.23262）混淆。

## 一句话定义

人类移动操作 Ego 数据 + 静态机器人数据共训，绕开大规模移动机器人遥操作成本。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称视角感知与控制 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |

## 为什么重要

- 移动操作最贵的地方在于：机器人不只是动手，还要移动身体。让机器人完成移动操作遥操作，成本比固定机械臂高很多。EMMA 的思路是用 **人类移动操作数据 + 静态机器人数据** 共同训练，绕开大规模移动机器人遥操作。
- 这篇让我意识到，Ego 的重要性不是局限在手部操作。只要任务涉及“人怎么走过去、怎么靠近物体、怎么把身体对准操作对象”，第一视角都会比外部视频更贴近执行过程。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 04/9 |
| 分组 | 02 人→机器人 |
| 出处 | 2025 · arXiv |
| 论文/项目 | <https://arxiv.org/abs/2509.04443> |
| 代码/项目 | <https://ego-moma.github.io/> |

## 核心机制（归纳）

### 1）策展导读要点

移动操作最贵的地方在于：机器人不只是动手，还要移动身体。让机器人完成移动操作遥操作，成本比固定机械臂高很多。EMMA 的思路是用 **人类移动操作数据 + 静态机器人数据** 共同训练，绕开大规模移动机器人遥操作。

### 2）策展导读要点

这篇让我意识到，Ego 的重要性不是局限在手部操作。只要任务涉及“人怎么走过去、怎么靠近物体、怎么把身体对准操作对象”，第一视角都会比外部视频更贴近执行过程。

### 3）策展导读要点

**未来移动操作最缺的，可能不是某一个抓取动作，而是人如何在空间里组织身体和任务。** 这恰好是 Ego 数据最容易记录下来的部分。

## 常见误区

1. Ego 视频不会天然等于机器人策略数据；须经过重建、对齐、重定向与物理过滤。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 技术地图：[ego-9-papers-technology-map.md](../overview/ego-9-papers-technology-map.md)
- 分类 hub：[ego-category-02-human-to-robot.md](../overview/ego-category-02-human-to-robot.md)
- 原始 source：[ego_survey_04_emma.md](../../sources/papers/ego_survey_04_emma.md)

## 参考来源

- [ego_survey_04_emma.md](../../sources/papers/ego_survey_04_emma.md) — Ego 9 篇策展摘录
- [ego_9_papers_catalog.md](../../sources/papers/ego_9_papers_catalog.md) — 9 篇总表
- [wechat_embodied_ai_lab_ego_9_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md) — 微信公众号编译导读
- 论文/项目：<https://arxiv.org/abs/2509.04443>

## 推荐继续阅读

- [Ego 9 篇技术地图](../overview/ego-9-papers-technology-map.md)
- [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
