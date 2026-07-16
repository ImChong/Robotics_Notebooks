---

type: entity
tags: [paper, bfm, behavior-foundation-model, awesome-bfm-papers, meta]
status: complete
updated: 2026-06-25
venue: "2024 · NeurIPS"
summary: "有行为基座后新动作应少走弯路；降低技能扩展的真机数据与训练成本。"
related:
  - ../concepts/behavior-foundation-model.md
  - ../overview/bfm-41-papers-technology-map.md
  - ../overview/bfm-category-01-forward-backward-representation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/bfm_awesome_fast_imitation_bfm_neurips_2024.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
---

# Fast Imitation via Behavior Foundation Models

**Fast Imitation via Behavior Foundation Models** 收录于 [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) **第 04/41** 篇，归类为 **01 Forward-backward 表征**（2024 · NeurIPS）。

> **方法背景：** [imitation-learning](../methods/imitation-learning.md) — 通用方法页（非本文专属深读）；本文机制与实验以原文为准，本页保留 survey 坐标与交叉引用。

## 一句话定义

有行为基座后新动作应少走弯路；降低技能扩展的真机数据与训练成本。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFM | Behavior Foundation Model | 大规模行为数据预训练的可复用全身行为先验 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 有行为基座后新动作应少走弯路；降低技能扩展的真机数据与训练成本。
- 在 [BFM 41 篇技术地图](../overview/bfm-41-papers-technology-map.md) 中属于 **01 Forward-backward 表征**（#04/41）。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 04/41 |
| 分组 | 01 Forward-backward 表征 |
| 出处 | 2024 · NeurIPS |
| 论文 | <https://openreview.net/pdf?id=qnWtw3l0jb> |

## 核心机制（归纳）

### 1）策展导读要点

在无监督或多任务 RL 中学习 **forward-backward（FB）** 或 successor 结构，把异构任务压进可调用的身体潜空间。

### 2）策展导读要点

上层通过 **目标姿态、奖励向量或 latent prompt** 在潜空间中检索/组合行为，而非为每个技能单独训练策略。

### 3）策展导读要点

与单技能 motion tracking 对照：BFM 关心 **覆盖面与可组合性**，不只单一参考跟踪成功率。

## 常见误区

1. BFM-Zero 类工作不是「更大动作数据集」本身，而是 **潜空间可被 prompt 检索** 的身体接口。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 技术地图：[bfm-41-papers-technology-map.md](../overview/bfm-41-papers-technology-map.md)
- BFM 概念：[behavior-foundation-model.md](../concepts/behavior-foundation-model.md)
- 原始 source：[bfm_awesome_fast_imitation_bfm_neurips_2024.md](../../sources/papers/bfm_awesome_fast_imitation_bfm_neurips_2024.md)

## 参考来源

- [bfm_awesome_fast_imitation_bfm_neurips_2024.md](../../sources/papers/bfm_awesome_fast_imitation_bfm_neurips_2024.md) — awesome-bfm 策展摘录
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md) — 41+10 总表
- [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md) — 微信公众号编译导读
- 论文：<https://openreview.net/pdf?id=qnWtw3l0jb>

## 推荐继续阅读

- [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) — 完整列表与数据集表
- [A Survey of Behavior Foundation Model](https://arxiv.org/abs/2506.20487) — TPAMI 2025 综述
