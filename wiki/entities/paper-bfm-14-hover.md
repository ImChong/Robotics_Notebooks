---

type: entity
tags: [paper, bfm, behavior-foundation-model, awesome-bfm-papers, nvidia]
status: complete
updated: 2026-06-25
arxiv: "2410.21229"
venue: "2025 · ICRA"
code: https://github.com/NVlabs/HOVER/
summary: "统一头/手/身体/根目标的神经全身接口，供上层规划器调用。"
related:
  - ../concepts/behavior-foundation-model.md
  - ../overview/bfm-41-papers-technology-map.md
  - ../overview/bfm-category-02-goal-conditioned-learning.md
  - ../concepts/whole-body-control.md
sources:
  - ../../sources/papers/bfm_awesome_hover_arxiv_2410_21229.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
---

# HOVER

**HOVER** 收录于 [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) **第 14/41** 篇，归类为 **02 Goal-conditioned 学习**（2025 · ICRA）。

## 一句话定义

统一头/手/身体/根目标的神经全身接口，供上层规划器调用。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFM | Behavior Foundation Model | 大规模行为数据预训练的可复用全身行为先验 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 统一头/手/身体/根目标的神经全身接口，供上层规划器调用。
- 在 [BFM 41 篇技术地图](../overview/bfm-41-papers-technology-map.md) 中属于 **02 Goal-conditioned 学习**（#14/41）。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 14/41 |
| 分组 | 02 Goal-conditioned 学习 |
| 出处 | 2025 · ICRA |
| 论文 | <https://arxiv.org/abs/2410.21229> |
- **代码/项目：** <https://github.com/NVlabs/HOVER/>

## 核心机制（归纳）

### 1）策展导读要点

以 **goal / reference / command** 为条件训练全身跟踪或交互策略，扩展人形可执行动作库。

### 2）策展导读要点

数据侧常融合 MoCap、视频、遥操作与 HOI；控制侧强调 **抗扰、恢复与跨参考泛化**。

### 3）策展导读要点

在 BFM taxonomy 中回答「身体能覆盖多少目标条件技能」。

## 常见误区

1. Goal-conditioned 跟踪不等于 unlimited skills：仍受数据分布、接触建模与实机 Sim2Real 约束。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 技术地图：[bfm-41-papers-technology-map.md](../overview/bfm-41-papers-technology-map.md)
- BFM 概念：[behavior-foundation-model.md](../concepts/behavior-foundation-model.md)
- 原始 source：[bfm_awesome_hover_arxiv_2410_21229.md](../../sources/papers/bfm_awesome_hover_arxiv_2410_21229.md)

## 参考来源

- [bfm_awesome_hover_arxiv_2410_21229.md](../../sources/papers/bfm_awesome_hover_arxiv_2410_21229.md) — awesome-bfm 策展摘录
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md) — 41+10 总表
- [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md) — 微信公众号编译导读
- 论文：<https://arxiv.org/abs/2410.21229>

## 推荐继续阅读

- [机器人论文阅读笔记：HOVER Versatile Neural Whole-Body Controller](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/HOVER_Versatile_Neural_Whole-Body_Controller/HOVER_Versatile_Neural_Whole-Body_Controller.html)
- [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) — 完整列表与数据集表
- [A Survey of Behavior Foundation Model](https://arxiv.org/abs/2506.20487) — TPAMI 2025 综述
