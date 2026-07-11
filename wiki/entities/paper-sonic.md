---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, bfm, behavior-foundation-model, nvidia, loco-manipulation, loco-manip-161-survey]
status: complete
updated: 2026-07-11
arxiv: "2511.07820"
venue: "2025 · arXiv"
summary: "SONIC：规模化运动跟踪人形全身控制；在 RL 身体系统栈属参考跟踪层，在 BFM 谱系强调 goal-conditioned 与运控基座覆盖面。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-01-motion-base-wbt.md
  - ../overview/loco-manip-161-category-04-generative-language-trajectory.md
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-04-wbt-base.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../overview/bfm-41-papers-technology-map.md
  - ../overview/bfm-category-02-goal-conditioned-learning.md
  - ../concepts/behavior-foundation-model.md
  - ../tasks/loco-manipulation.md
  - ../methods/sonic-motion-tracking.md
sources:
  - ../../sources/papers/humanoid_rl_stack_17_sonic_supersizing_motion_tracking_for_natural_hu.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/papers/bfm_awesome_sonic_arxiv_2511_07820.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/loco_manip_161_survey_019_sonic.md
  - ../../sources/papers/loco_manip_161_survey_103_sonic.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
  - ../../sources/repos/sonic-humanoid-motion-tracking.md
---

# SONIC

**SONIC**（*Supersizing Motion Tracking for Natural Humanoid Whole-Body Control*，arXiv:2511.07820）把 humanoid whole-body motion tracker 当作可扩展的基础模型来研究。

> **深读页：** [sonic-motion-tracking](../methods/sonic-motion-tracking.md) — 方法机制与实验细节见链接页；本页保留 survey 坐标与交叉引用。

## 一句话定义

SONIC 的题目是 Supersizing Motion Tracking for Natural Humanoid Whole-Body Control。它把 humanoid whole-body motion tracker 当成基础模型来扩展，研究参数规模、数据规模、训练计算对控制能力的影响。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| BFM | Behavior Foundation Model | 大规模行为数据预训练的可复用全身行为先验 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **02 参考跟踪 · 通用控制**（#17/42）。
- SONIC 的题目是 Supersizing Motion Tracking for Natural Humanoid Whole-Body Control。它把 humanoid whole-body motion tracker 当成基础模型来扩展，研究参数规模、数据规模、训练计算对控制能力的影响。
- 这篇论文和传统 motion tracking 工作的区别在于，它不只问“某个策略能不能跟某些动作”，而是问：**把网络容量、MoCap 数据量与训练算力同时放大之后，单一 tracker 的动作覆盖面、自然度与对未见动作的泛化能到什么程度**——这是把 LLM 式 scaling law 提问方式搬到运动控制上。
- 它还讨论下游任务、交互式运动控制，以及 motion 和 VLA 表示的迁移价值。这意味着 motion tracker 不再只是一个执行模块，而可能成为上层任务模型的底层表征。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 论文 | <https://arxiv.org/abs/2511.07820> |
| 项目页 | <https://nvlabs.github.io/SONIC/>（主站 <https://nvlabs.github.io/GEAR-SONIC/>） |
| 机构 | NVIDIA（GEAR Lab 等）；CMU 等合作者 |
| 作者 | Zhengyi Luo、Ye Yuan、Tingwu Wang 等 28 人；含 Tairan He、Jan Kautz、Linxi "Jim" Fan、Yuke Zhu |
| arXiv 版本 | v1 2025-11-11 → v2 2025-12-04 → v3 2026-05-21 |

### 在 42 篇 RL 运动控制身体系统栈中

| 字段 | 内容 |
|------|------|
| 编号 | 17/42 |
| 系统栈层 | 02 参考跟踪 · 通用控制 |
| 索引来源 | [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) |

### 在 BFM 41 篇技术地图中

| 字段 | 内容 |
|------|------|
| 编号 | 07/41 |
| 分组 | 02 Goal-conditioned 学习 |
| 索引来源 | [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) |

### 在人形 Loco-Manip 161 篇中

同一篇论文在 [Loco-Manip 161 篇技术地图](../overview/humanoid-loco-manip-161-papers-technology-map.md) 里出现 **两次**（策展分类不同，canonical 实体仅此页）：

| 槽位 | 分组 | 分类 hub |
|------|------|----------|
| 019/161 | 01 运控基座与通用全身跟踪 | [loco-manip-161-category-01-motion-base-wbt](../overview/loco-manip-161-category-01-motion-base-wbt.md) |
| 103/161 | 04 生成式运动、语言控制与轨迹规划 | [loco-manip-161-category-04-generative-language-trajectory](../overview/loco-manip-161-category-04-generative-language-trajectory.md) |

索引来源：[具身智能研究室 · 161 篇人形 Loco-Manip 长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A)

## 核心机制（归纳）

### 1）三轴 scaling：模型 × 数据 × 算力

把 motion tracking 当作可规模化的预训练任务：网络容量从 **1.2M 扩到 42M 参数**，监督数据为 **1 亿+ MoCap 帧（约 700 小时）** 的密集轨迹监督，训练投入约 **2.1 万 GPU 小时**；论文报告性能随算力与数据多样性 **稳步改善**，且策略能 **泛化到未见动作**。

### 2）统一 token 接口与共享潜表征

**专用编码器族 → 统一 token 空间**：单一策略经共享潜表征同时处理 **机器人运动、人体运动与混合运动** 三类指令。VR（三点 / 全身）、视频（GEM 姿态估计）、文本 / 音乐（GEM 生成人体运动）与 VLA 等上游都汇入同一接口，换上游不需要重写奖励或重训低层。

### 3）下游系统形态

**实时运动学规划器** 与 tracking 衔接，支持手柄导航、风格化步态与受限空间动作（蹲行、爬行）；**GR00T N1.5** 经同一接口接入，实现 **自主 VLA 驱动的全身 loco-manipulation**——「慢推理 / 快反射」分层控制的公开工程形态之一。

### 4）策展提醒

SONIC 的方向很重要，但也要谨慎：运动控制的 scaling 不会和语言模型完全一样，因为机器人还受硬件、物理和实时闭环约束。参数变大不一定能直接解决接触和安全问题。

## 常见误区

1. Motion tracking 论文的泛化常指 **参考分布内**；换数据源或接触条件仍可能崩塌。

## 实验与评测

- **规模量级**（论文摘要口径）：1 亿+ 帧 / 约 700 小时 MoCap、1.2M→42M 参数、约 2.1 万 GPU 小时；scaling 曲线显示性能随算力与数据多样性稳步改善。
- **系统演示**（项目页）：单一统一策略完成 VR 三点 / 全身遥操、视频 + GEM 跟踪、文本 / 音乐条件、运动学规划器导航与 GR00T N1.5 VLA loco-manip；扰动下跟踪单列展示。
- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 方法深读：[sonic-motion-tracking.md](../methods/sonic-motion-tracking.md)
- RL 身体系统栈：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- BFM 技术地图：[bfm-41-papers-technology-map.md](../overview/bfm-41-papers-technology-map.md)
- Loco-Manip 161 篇：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- BFM 概念：[behavior-foundation-model.md](../concepts/behavior-foundation-model.md)

## 参考来源

- [humanoid_rl_stack_17_sonic_supersizing_motion_tracking_for_natural_hu.md](../../sources/papers/humanoid_rl_stack_17_sonic_supersizing_motion_tracking_for_natural_hu.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 42 篇总表
- [bfm_awesome_sonic_arxiv_2511_07820.md](../../sources/papers/bfm_awesome_sonic_arxiv_2511_07820.md) — awesome-bfm 策展摘录
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md) — 41+10 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — RL 运动控制微信公众号编译导读
- [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md) — BFM 41 篇微信公众号编译导读
- [loco_manip_161_survey_019_sonic.md](../../sources/papers/loco_manip_161_survey_019_sonic.md) — Loco-Manip 161 #019 策展摘录
- [loco_manip_161_survey_103_sonic.md](../../sources/papers/loco_manip_161_survey_103_sonic.md) — Loco-Manip 161 #103 策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md) — Loco-Manip 161 总表
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md) — Loco-Manip 161 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)
- 论文：<https://arxiv.org/abs/2511.07820>（v3，2026-05-21）
- [sonic-humanoid-motion-tracking.md](../../sources/repos/sonic-humanoid-motion-tracking.md) — 论文 + 官网公开材料对照整理

## 推荐继续阅读

- [机器人论文阅读笔记：SONIC Supersizing Motion Tracking for Natural Humanoid Control](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/SONIC_Supersizing_Motion_Tracking_for_Natural_Humanoid_Control/SONIC_Supersizing_Motion_Tracking_for_Natural_Humanoid_Control.html)
- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) — 完整列表与数据集表
- [A Survey of Behavior Foundation Model](https://arxiv.org/abs/2506.20487) — TPAMI 2025 综述
