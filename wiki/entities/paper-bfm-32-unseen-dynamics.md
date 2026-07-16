---

type: entity
tags: [paper, bfm, behavior-foundation-model, awesome-bfm-papers, meta]
status: complete
updated: 2026-06-25
arxiv: "2505.13150"
venue: "2025 · arXiv"
summary: "负载/地面/硬件参数变化下的零样本动力学适配。"
related:
  - ../concepts/behavior-foundation-model.md
  - ../overview/bfm-41-papers-technology-map.md
  - ../overview/bfm-category-04-adaptation.md
  - ../concepts/sim2real.md
  - ../entities/paper-any2any-cross-embodiment-wbt.md
sources:
  - ../../sources/papers/bfm_awesome_unseen_dynamics_arxiv_2505_13150.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
---

# Zero-Shot Adaptation of Behavioral Foundation Models to Unseen Dynamics

**Zero-Shot Adaptation of Behavioral Foundation Models to Unseen Dynamics** 收录于 [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) **第 32/41** 篇，归类为 **04 Adaptation**（2025 · arXiv）。

> **同主题深读：** [paper-any2any-cross-embodiment-wbt](../entities/paper-any2any-cross-embodiment-wbt.md) — 同题材（跨动力学/跨具身适配）的另一篇论文深读页，两者并非同一工作；本页保留本文的 survey 坐标与交叉引用。

## 一句话定义

负载/地面/硬件参数变化下的零样本动力学适配。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| BFM | Behavior Foundation Model | 大规模行为数据预训练的可复用全身行为先验 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 负载/地面/硬件参数变化下的零样本动力学适配。
- 在 [BFM 41 篇技术地图](../overview/bfm-41-papers-technology-map.md) 中属于 **04 Adaptation**（#32/41）。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 32/41 |
| 分组 | 04 Adaptation |
| 出处 | 2025 · arXiv |
| 论文 | <https://arxiv.org/abs/2505.13150> |

## 核心机制（归纳）

### 1）策展导读要点

在预训练 BFM 上通过 **task token、动力学适配或少量示范** 快速迁移到新任务/新机体。

### 2）策展导读要点

核心问题是 **保留基座能力的同时** 以低成本吸收新约束，而非从零重训。

## 常见误区

1. Fast adaptation 论文通常假设 **已有强预训练基座**；弱基座上 adapter 收益有限。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 技术地图：[bfm-41-papers-technology-map.md](../overview/bfm-41-papers-technology-map.md)
- BFM 概念：[behavior-foundation-model.md](../concepts/behavior-foundation-model.md)
- 原始 source：[bfm_awesome_unseen_dynamics_arxiv_2505_13150.md](../../sources/papers/bfm_awesome_unseen_dynamics_arxiv_2505_13150.md)

## 参考来源

- [bfm_awesome_unseen_dynamics_arxiv_2505_13150.md](../../sources/papers/bfm_awesome_unseen_dynamics_arxiv_2505_13150.md) — awesome-bfm 策展摘录
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md) — 41+10 总表
- [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md) — 微信公众号编译导读
- 论文：<https://arxiv.org/abs/2505.13150>

## 推荐继续阅读

- [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) — 完整列表与数据集表
- [A Survey of Behavior Foundation Model](https://arxiv.org/abs/2506.20487) — TPAMI 2025 综述
