---

type: entity
tags: [paper, egocentric, ego-survey, embodied-ai, meta]
status: complete
updated: 2026-06-25
venue: "2026 · 项目"
code: https://e3c-videogen.github.io/
summary: "3D 环境记忆 + ego/exo 人体姿态控制的第一视角视频生成，服务可推演的世界片段而非短视频观感。"
related:
  - ../overview/ego-9-papers-technology-map.md
  - ../overview/ego-category-04-ego-exo-fusion.md
sources:
  - ../../sources/papers/ego_survey_09_e3c.md
  - ../../sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md
  - ../../sources/papers/ego_9_papers_catalog.md
---

# E³C

**E³C** 收录于 [具身智能研究室 · Ego 9 篇专题](https://mp.weixin.qq.com/s/4JQ1xa-cJ7J1ep_e4txNnA) **第 09/9** 篇，归类为 **04 Ego+Exo**。

## 一句话定义

3D 环境记忆 + ego/exo 人体姿态控制的第一视角视频生成，服务可推演的世界片段而非短视频观感。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称视角感知与控制 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |

## 为什么重要

- E³C 讨论的是可控第一视角视频生成。它的关键词有三个：3D environmental memory、ego human control、exo human control。
- 这篇让我比较在意的是，第一视角视频生成不能只追求画面像。Ego 视频里相机和身体绑定，视角变化快，遮挡多，人的动作还经常只露出一部分。如果没有 3D 环境记忆，生成视频很容易在空间上漂。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 09/9 |
| 分组 | 04 Ego+Exo |
| 出处 | 2026 · 项目 |
| 论文/项目 | <https://e3c-videogen.github.io/> |

## 核心机制（归纳）

### 1）策展导读要点

E³C 讨论的是可控第一视角视频生成。它的关键词有三个：3D environmental memory、ego human control、exo human control。

### 2）策展导读要点

这篇让我比较在意的是，第一视角视频生成不能只追求画面像。Ego 视频里相机和身体绑定，视角变化快，遮挡多，人的动作还经常只露出一部分。如果没有 3D 环境记忆，生成视频很容易在空间上漂。

### 3）策展导读要点

放到机器人里看，这件事更明显。机器人要靠世界模型想象未来，生成出来的未来不能只是好看，还要能保持场景结构、身体动作和物体变化的一致。

## 常见误区

1. Ego 视频不会天然等于机器人策略数据；须经过重建、对齐、重定向与物理过滤。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 技术地图：[ego-9-papers-technology-map.md](../overview/ego-9-papers-technology-map.md)
- 分类 hub：[ego-category-04-ego-exo-fusion.md](../overview/ego-category-04-ego-exo-fusion.md)
- 原始 source：[ego_survey_09_e3c.md](../../sources/papers/ego_survey_09_e3c.md)

## 参考来源

- [ego_survey_09_e3c.md](../../sources/papers/ego_survey_09_e3c.md) — Ego 9 篇策展摘录
- [ego_9_papers_catalog.md](../../sources/papers/ego_9_papers_catalog.md) — 9 篇总表
- [wechat_embodied_ai_lab_ego_9_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md) — 微信公众号编译导读
- 论文/项目：<https://e3c-videogen.github.io/>

## 推荐继续阅读

- [Ego 9 篇技术地图](../overview/ego-9-papers-technology-map.md)
- [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
