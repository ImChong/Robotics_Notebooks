---

type: entity
tags: [paper, egocentric, ego-survey, embodied-ai, alibaba, baai, pku, ucas, zju]
status: complete
updated: 2026-06-25
arxiv: "2602.23893"
summary: "颈挂手机 + 端云协同：把「人人可采」的第一视角交互数据做成系统，而非实验室专用设备。"
related:
  - ../overview/ego-9-papers-technology-map.md
  - ../overview/ego-category-01-data-collection.md
sources:
  - ../../sources/papers/ego_survey_01_aoe.md
  - ../../sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md
  - ../../sources/papers/ego_9_papers_catalog.md
---

# AoE

**AoE** 收录于 [具身智能研究室 · Ego 9 篇专题](https://mp.weixin.qq.com/s/4JQ1xa-cJ7J1ep_e4txNnA) **第 01/9** 篇，归类为 **01 数据采集**。

## 一句话定义

颈挂手机 + 端云协同：把「人人可采」的第一视角交互数据做成系统，而非实验室专用设备。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称视角感知与控制 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |

## 为什么重要

- 它用的是更低门槛的方式：颈挂式手机支架、移动端应用、云端自动标注和过滤。这个设计背后的判断很直接：如果具身模型要继续 scaling，不能只依赖实验室里的机器人采集。
- **人类每天都在真实世界里完成大量操作任务。** 做饭、整理、搬放、购物、清洁，这些动作本身就包含丰富的视觉、接触和任务顺序。Ego 数据的价值，是把这些原本没有被机器人系统记录下来的经验，变成可以被整理、筛选和训练的数据源。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 01/9 |
| 分组 | 01 数据采集 |
| 出处 | 2026 · arXiv |
| 论文/项目 | <https://arxiv.org/abs/2602.23893> |

## 核心机制（归纳）

### 1）策展导读要点

它用的是更低门槛的方式：颈挂式手机支架、移动端应用、云端自动标注和过滤。这个设计背后的判断很直接：如果具身模型要继续 scaling，不能只依赖实验室里的机器人采集。

### 2）策展导读要点

**人类每天都在真实世界里完成大量操作任务。** 做饭、整理、搬放、购物、清洁，这些动作本身就包含丰富的视觉、接触和任务顺序。Ego 数据的价值，是把这些原本没有被机器人系统记录下来的经验，变成可以被整理、筛选和训练的数据源。

### 3）策展导读要点

这里也有风险。手机采集的数据肯定不如专业设备干净，视角会抖，手会挡住物体，动作也不一定严格标准。但这类工作真正打开的，是一个规模问题：**如果人人都能低成本采集第一视角任务数据，具身数据的增长方式会发生变化。**

## 常见误区

1. Ego 视频不会天然等于机器人策略数据；须经过重建、对齐、重定向与物理过滤。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 技术地图：[ego-9-papers-technology-map.md](../overview/ego-9-papers-technology-map.md)
- 分类 hub：[ego-category-01-data-collection.md](../overview/ego-category-01-data-collection.md)
- 原始 source：[ego_survey_01_aoe.md](../../sources/papers/ego_survey_01_aoe.md)

## 参考来源

- [ego_survey_01_aoe.md](../../sources/papers/ego_survey_01_aoe.md) — Ego 9 篇策展摘录
- [ego_9_papers_catalog.md](../../sources/papers/ego_9_papers_catalog.md) — 9 篇总表
- [wechat_embodied_ai_lab_ego_9_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md) — 微信公众号编译导读
- 论文/项目：<https://arxiv.org/abs/2602.23893>

## 推荐继续阅读

- [Ego 9 篇技术地图](../overview/ego-9-papers-technology-map.md)
- [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
