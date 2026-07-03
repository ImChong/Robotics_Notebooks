---
type: entity
tags: [paper, loco-manip-contact-survey, humanoid, loco-manipulation, hoi, generative-models, zero-shot, behavior-foundation-model]
status: complete
updated: 2026-07-03
arxiv: "2605.22272"
venue: "2026 · arXiv"
summary: "Imagine2Real 用统一 4D 点轨迹与 BFM 潜空间稀疏关键点 Tracker 实现零样本人形 HOI，绕过密集重定向与 CAD 几何先验。"
related:
  - ../overview/loco-manip-contact-technology-map.md
  - ../overview/loco-manip-contact-category-03-generative-data.md
  - ./paper-loco-manip-03-genhoi.md
  - ../concepts/behavior-foundation-model.md
sources:
  - ../../sources/papers/imagine2real_arxiv_2605_22272.md
  - ../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md
---

# Imagine2Real

**Imagine2Real**（arXiv:2605.22272，ZJU / Shanghai AI Lab 等）收录于 [具身智能研究室 · Loco-Manip 接触专题](https://mp.weixin.qq.com/s/UjShbwl8p1h9ukymfiRNaw) **03 生成式补数** 组：用 **视频生成先验** 实现 **零样本人形-物体交互**。

## 一句话定义

给定图像与文本，合成交互视频 → 提取统一 **4D 点轨迹**（机器人+物体）→ **BFM 潜空间** 上的稀疏关键点 Tracker（基座、双手、物体）跟踪参考，三阶段渐进训练后在 mocap 系统 **零样本** 物理部署。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HOI | Human-Object Interaction | 全身人-物交互 |
| BFM | Behavior Foundation Model | Tracker 搜索域与全身跟踪底座 |
| AMP | Adversarial Motion Priors | 对比路线；本文用 BFM + 稀疏关键点 |
| Loco-Manip | Loco-Manipulation | 含行走与操作的全身 HOI |
| RL | Reinforcement Learning | 渐进训练后段与适配器 |

## 为什么重要

- **几何无关：** 统一点轨迹表示避免 CAD 模型与独立估计带来的 **表示错位**。
- **绕过重定向：** 只跟踪交互关键三点，规避 HOI 场景下复杂 morphing 误差放大。
- **与 GenHOI 并列：** 同属生成视频驱动 HOI；Imagine2Real 强调 **稀疏关键点 + BFM**，GenHOI 强调从生成视频恢复可执行轨迹。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 分组 | Loco-Manip 接触专题 · 03 生成式补数 |
| 原文题目 | Imagine2Real: Towards Zero-shot Humanoid-Object Interaction via Video Generative Priors |
| 机构 | Zhejiang University；Shanghai AI Laboratory；CUHK 等 |
| 论文/项目 | <https://arxiv.org/abs/2605.22272> |

## 与其他页面的关系

- 分类 hub：[loco-manip-contact-category-03-generative-data.md](../overview/loco-manip-contact-category-03-generative-data.md)
- 姊妹：[GenHOI](paper-loco-manip-03-genhoi.md)

## 参考来源

- [imagine2real_arxiv_2605_22272.md](../../sources/papers/imagine2real_arxiv_2605_22272.md)
- [wechat_embodied_ai_lab_loco_manip_contact_survey.md](../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)

## 推荐继续阅读

- [GenHOI](paper-loco-manip-03-genhoi.md)
- [BFM 概念页](../concepts/behavior-foundation-model.md)
