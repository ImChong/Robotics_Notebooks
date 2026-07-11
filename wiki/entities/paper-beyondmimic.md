---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, bfm, behavior-foundation-model, stanford, berkeley]
status: complete
updated: 2026-07-11
arxiv: "2508.08241"
venue: "2025 · arXiv"
code: https://github.com/HybridRobotics/whole_body_tracking
summary: "BeyondMimic：从 motion tracking 到 guided diffusion 的通用人形控制；在 RL 身体系统栈属参考跟踪层，在 BFM 谱系属 hierarchical control。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-04-wbt-base.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../overview/bfm-41-papers-technology-map.md
  - ../overview/bfm-category-05-hierarchical-control.md
  - ../concepts/behavior-foundation-model.md
  - ../methods/beyondmimic.md
sources:
  - ../../sources/papers/humanoid_rl_stack_15_beyondmimic_from_motion_tracking_to_versatile_hu.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/papers/bfm_awesome_beyondmimic_arxiv_2508_08241.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/loco_manip_161_survey_004_beyondmimic.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# BeyondMimic

**BeyondMimic**（*From Motion Tracking to Versatile Humanoid Control via Guided Diffusion*，arXiv:2508.08241）将 guided diffusion 引入全身人形控制。

> **深读页：** [beyondmimic](../methods/beyondmimic.md) — 方法机制与实验细节见链接页；本页保留 survey 坐标与交叉引用。

## 一句话定义

BeyondMimic 的完整题目是 From Motion Tracking to Versatile Humanoid Control via Guided Diffusion。它放在 OmniTrack、RGMT 旁边很合适，因为它同样在处理“参考动作如何变成可执行控制”这个问题，但切入点更偏生成式。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| BFM | Behavior Foundation Model | 大规模行为数据预训练的可复用全身行为先验 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **02 参考跟踪 · 通用控制**（#15/42）。
- BeyondMimic 的完整题目是 From Motion Tracking to Versatile Humanoid Control via Guided Diffusion。它放在 OmniTrack、RGMT 旁边很合适，因为它同样在处理“参考动作如何变成可执行控制”这个问题，但切入点更偏生成式。
- 普通 motion tracking 通常假设参考轨迹已经给定，控制器只需要尽量追踪。但真实任务里，参考动作可能缺局部细节，可能和当前环境不匹配，也可能需要根据任务目标在线调整。
- BeyondMimic 的思路是引入 guided diffusion，让策略不只是被动追踪参考，而是能在约束和目标引导下生成、修正更合适的动作。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 论文 | <https://arxiv.org/abs/2508.08241> |
| 项目页 | <https://beyondmimic.github.io/> |
| 代码 | <https://github.com/HybridRobotics/whole_body_tracking> |
| 机构 | 伯克利；斯坦福大学 |
| 作者 | Qiayuan Liao、Takara E. Truong、Xiaoyu Huang、Yuman Gao、Guy Tevet、Koushil Sreenath、C. Karen Liu |
| arXiv 版本 | v1 2025-08-11 → v4 2025-11-13 |

### 在 42 篇 RL 运动控制身体系统栈中

| 字段 | 内容 |
|------|------|
| 编号 | 15/42 |
| 系统栈层 | 02 参考跟踪 · 通用控制 |
| 索引来源 | [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) |

### 在 BFM 41 篇技术地图中

| 字段 | 内容 |
|------|------|
| 编号 | 35/41 |
| 分组 | 05 Hierarchical control |
| 索引来源 | [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) |

### 在人形 Loco-Manip 161 篇中

| 字段 | 内容 |
|------|------|
| 槽位 | 004/161 |
| 分组 | 01 运控基座与通用全身跟踪 |
| 分类 hub | [loco-manip-161-category-01-motion-base-wbt](../overview/loco-manip-161-category-01-motion-base-wbt.md) |
| 索引来源 | [具身智能研究室 · 161 篇人形 Loco-Manip 长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) |

## 核心机制（归纳）

### 1）两阶段框架：紧凑跟踪 → 引导扩散

论文摘要口径的两段式结构：**阶段 ①** 用一个 **紧凑的 motion-tracking 公式** 覆盖高动态技能；**阶段 ②** 把这些技能升格为 **统一潜空间扩散模型**，支持多样目标指定、任务无缝切换与技能动态组合——「模仿」由此走向「可控生成」。

### 2）阶段 ①：单一设置跟踪高动态动作

在 **LAFAN1 的 14 段约 3 分钟长序列** 上逐条训练跟踪策略，**全部使用同一 MDP 设置与共享超参**（不做逐动作调参），覆盖侧空翻（aerial cartwheel）、旋踢（spin-kick）、翻转踢（flip-kick）、冲刺跑等技能，同时保持 SOTA 级人类相似度——回应了此前方法「动作不自然或依赖逐动作调参」的痛点。

### 3）阶段 ②：classifier guidance 零样本下游任务

多条跟踪策略蒸馏进 **单一 latent diffusion policy** 后，测试时用 **classifier guidance**——以简单代价函数（路点距离、障碍距离等）的梯度引导扩散采样——解决 **训练中从未出现** 的任务：**motion inpainting、joystick 遥操、waypoint 导航、障碍规避**，并 **零样本迁移到真实硬件**。任务语义在测试时注入，策略本体不动。

### 4）策展定位

这篇论文的价值在于，它把“模仿”往“可控生成”推进了一步。DeepMimic 那条线解决的是“如何跟着人类动作学”；OmniTrack 和 RGMT 解决的是“参考动作怎样更物理、更鲁棒”；BeyondMimic 则进一步问：如果参考本身不够，能不能让模型在物理约束下补出更可执行的行为？

## 常见误区

1. Motion tracking 论文的泛化常指 **参考分布内**；换数据源或接触条件仍可能崩塌。

## 实验与评测

- **跟踪阶段**：LAFAN1 14 段约 3 分钟序列，单一 MDP 设置 + 共享超参；项目页展示 jumping spin、sprint、cartwheel 等技能在实机上的稳定复现。
- **下游阶段**：joystick 遥操、waypoint 导航、障碍规避与 motion inpainting 零样本实机验证；动捕系统用于路点 / 障碍定位与辅助状态估计。
- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 方法深读：[beyondmimic.md](../methods/beyondmimic.md)
- RL 身体系统栈：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- BFM 技术地图：[bfm-41-papers-technology-map.md](../overview/bfm-41-papers-technology-map.md)
- BFM 概念：[behavior-foundation-model.md](../concepts/behavior-foundation-model.md)

## 参考来源

- [humanoid_rl_stack_15_beyondmimic_from_motion_tracking_to_versatile_hu.md](../../sources/papers/humanoid_rl_stack_15_beyondmimic_from_motion_tracking_to_versatile_hu.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 42 篇总表
- [bfm_awesome_beyondmimic_arxiv_2508_08241.md](../../sources/papers/bfm_awesome_beyondmimic_arxiv_2508_08241.md) — awesome-bfm 策展摘录
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md) — 41+10 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — RL 运动控制微信公众号编译导读
- [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md) — BFM 41 篇微信公众号编译导读
- [loco_manip_161_survey_004_beyondmimic.md](../../sources/papers/loco_manip_161_survey_004_beyondmimic.md) — Loco-Manip 161 #004 策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md) — Loco-Manip 161 总表
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)
- 论文：<https://arxiv.org/abs/2508.08241>（v4，2025-11-13）；项目页：<https://beyondmimic.github.io/>

## 推荐继续阅读

- [机器人论文阅读笔记：BeyondMimic](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/01_Foundational_RL/BeyondMimic/BeyondMimic.html)
- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) — 完整列表与数据集表
- [A Survey of Behavior Foundation Model](https://arxiv.org/abs/2506.20487) — TPAMI 2025 综述
