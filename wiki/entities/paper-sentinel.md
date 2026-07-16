---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, bfm, behavior-foundation-model, vla, beingbeyond, pku]
status: complete
updated: 2026-07-16
arxiv: "2511.19236"
venue: "2025 · arXiv"
summary: "SENTINEL：语言–全身动作端到端映射；在 RL 身体系统栈属视觉闭环/任务接口层，在 BFM 谱系属 hierarchical control。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../overview/bfm-41-papers-technology-map.md
  - ../overview/bfm-category-05-hierarchical-control.md
  - ../concepts/behavior-foundation-model.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/humanoid_rl_stack_31_sentinel_a_fully_end_to_end_language_action_mode.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/papers/bfm_awesome_sentinel_arxiv_2511_19236.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
---

# SENTINEL

**SENTINEL**（*A Fully End-to-End Language-Action Model for Humanoid Robots*，arXiv:2511.19236）将自然语言与本体感知直接映射到全身低层动作。

## 一句话定义

SENTINEL 直接把自然语言和本体感知映射到全身低层动作。它看起来是这批论文里最“端到端”的方向之一。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| BFM | Behavior Foundation Model | 大规模行为数据预训练的可复用全身行为先验 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一建模的机器人策略范式 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **04 视觉闭环 · 任务接口 · 世界模型**（#31/42）。
- SENTINEL 直接把自然语言和本体感知映射到全身低层动作。它看起来是这批论文里最“端到端”的方向之一。
- 但读细之后会发现，它并没有绕过运动控制。它先训练一个能跟踪人类动作的全身控制器，然后用这个控制器在仿真里 rollout，得到机器人自己的 state-action trajectories，再用语言标注训练 language-action model。
- 也就是说，SENTINEL 的监督信号不是“人体应该怎么动”，而是“机器人在动力学里实际能怎么动”。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 论文 | <https://arxiv.org/abs/2511.19236> |
| 机构 | 北京大学；BeingBeyond |

### 在 42 篇 RL 运动控制身体系统栈中

| 字段 | 内容 |
|------|------|
| 编号 | 31/42 |
| 系统栈层 | 04 视觉闭环 · 任务接口 · 世界模型 |
| 索引来源 | [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) |

### 在 BFM 41 篇技术地图中

| 字段 | 内容 |
|------|------|
| 编号 | 34/41 |
| 分组 | 05 Hierarchical control |
| 索引来源 | [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) |

## 核心机制（归纳）

### 1）策展导读要点

SENTINEL 直接把自然语言和本体感知映射到全身低层动作。它看起来是这批论文里最“端到端”的方向之一。

### 2）策展导读要点

但读细之后会发现，它并没有绕过运动控制。它先训练一个能跟踪人类动作的全身控制器，然后用这个控制器在仿真里 rollout，得到机器人自己的 state-action trajectories，再用语言标注训练 language-action model。

### 3）策展导读要点

也就是说，SENTINEL 的监督信号不是“人体应该怎么动”，而是“机器人在动力学里实际能怎么动”。

### 4）策展导读要点

论文用 Transformer + flow matching action head 预测 action chunk，并用 residual action head 做后训练，提升真实部署和扰动下稳定性。它还讨论了长期观测、done prediction、classifier-free guidance、action chunk horizon 等细节。

## 常见误区

1. VLA/世界模型条目解决 **接口与预测**，不自动替代已封装的底层 WBC 能力。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- VLA 语境：[vla.md](../methods/vla.md)
- RL 身体系统栈：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- BFM 技术地图：[bfm-41-papers-technology-map.md](../overview/bfm-41-papers-technology-map.md)
- BFM 概念：[behavior-foundation-model.md](../concepts/behavior-foundation-model.md)

## 参考来源

- [humanoid_rl_stack_31_sentinel_a_fully_end_to_end_language_action_mode.md](../../sources/papers/humanoid_rl_stack_31_sentinel_a_fully_end_to_end_language_action_mode.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 42 篇总表
- [bfm_awesome_sentinel_arxiv_2511_19236.md](../../sources/papers/bfm_awesome_sentinel_arxiv_2511_19236.md) — awesome-bfm 策展摘录
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md) — 41+10 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — RL 运动控制微信公众号编译导读
- [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md) — BFM 41 篇微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)
- 论文：<https://arxiv.org/abs/2511.19236>

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) — 完整列表与数据集表
- [A Survey of Behavior Foundation Model](https://arxiv.org/abs/2506.20487) — TPAMI 2025 综述
