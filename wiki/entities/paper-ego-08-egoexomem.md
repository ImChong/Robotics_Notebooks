---

type: entity
tags: [paper, egocentric, ego-survey, embodied-ai, eth]
status: complete
updated: 2026-06-25
arxiv: "2605.18734"
code: https://github.com/RuipingL/EgoExoMem
summary: "同步 ego-exo 视频上的跨视角记忆推理；提醒 Ego 需 Exo 补全空间结构。"
related:
  - ../overview/ego-9-papers-technology-map.md
  - ../overview/ego-category-04-ego-exo-fusion.md
sources:
  - ../../sources/papers/ego_survey_08_egoexomem.md
  - ../../sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md
  - ../../sources/papers/ego_9_papers_catalog.md
---

# EgoExoMem

**EgoExoMem** 收录于 [具身智能研究室 · Ego 9 篇专题](https://mp.weixin.qq.com/s/4JQ1xa-cJ7J1ep_e4txNnA) **第 08/9** 篇，归类为 **04 Ego+Exo**。

## 一句话定义

同步 ego-exo 视频上的跨视角记忆推理；提醒 Ego 需 Exo 补全空间结构。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称视角感知与控制 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |

## 为什么重要

- 它研究的是同步的 ego-exo 视频记忆推理。很多问题单靠第一视角答不出来，单靠外部视角也答不出来。第一视角知道操作者看到了什么，外部视角知道人与物体的整体关系。
- 这个结论对机器人也很有启发。只看机器人自己的相机，很多空间关系会丢；只看外部摄像头，又容易丢掉机器人的局部意图和接触细节。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 08/9 |
| 分组 | 04 Ego+Exo |
| 出处 | 2026 · arXiv |
| 论文/项目 | <https://arxiv.org/abs/2605.18734> |
| 代码/项目 | <https://github.com/RuipingL/EgoExoMem> |

## 核心机制（归纳）

### 1）策展导读要点

它研究的是同步的 ego-exo 视频记忆推理。很多问题单靠第一视角答不出来，单靠外部视角也答不出来。第一视角知道操作者看到了什么，外部视角知道人与物体的整体关系。

### 2）策展导读要点

这个结论对机器人也很有启发。只看机器人自己的相机，很多空间关系会丢；只看外部摄像头，又容易丢掉机器人的局部意图和接触细节。

### 3）策展导读要点

所以我觉得未来的数据系统，很可能不是单纯押 Ego 或 Exo，而是走向 **Ego + Exo + 3D 环境记忆**。Ego 负责临场执行，Exo 负责补全结构，环境记忆负责把短片段连接成长时程理解。

## 常见误区

1. Ego 视频不会天然等于机器人策略数据；须经过重建、对齐、重定向与物理过滤。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 技术地图：[ego-9-papers-technology-map.md](../overview/ego-9-papers-technology-map.md)
- 分类 hub：[ego-category-04-ego-exo-fusion.md](../overview/ego-category-04-ego-exo-fusion.md)
- 原始 source：[ego_survey_08_egoexomem.md](../../sources/papers/ego_survey_08_egoexomem.md)

## 参考来源

- [ego_survey_08_egoexomem.md](../../sources/papers/ego_survey_08_egoexomem.md) — Ego 9 篇策展摘录
- [ego_9_papers_catalog.md](../../sources/papers/ego_9_papers_catalog.md) — 9 篇总表
- [wechat_embodied_ai_lab_ego_9_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md) — 微信公众号编译导读
- 论文/项目：<https://arxiv.org/abs/2605.18734>

## 推荐继续阅读

- [Ego 9 篇技术地图](../overview/ego-9-papers-technology-map.md)
- [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
