---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, bfm, behavior-foundation-model, teleoperation, sfu, stanford]
status: complete
updated: 2026-06-25
arxiv: "2505.02833"
venue: "2025 · CoRL"
code: https://github.com/YanjieZe/TWIST
summary: "TWIST：全身遥操作模仿系统；在 RL 身体系统栈属数据/遥操作层，在 BFM 谱系强调遥操作作为持续数据生产方式。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-06-cross-embodiment-teleop.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../overview/bfm-41-papers-technology-map.md
  - ../overview/bfm-category-02-goal-conditioned-learning.md
  - ../concepts/behavior-foundation-model.md
  - ../tasks/teleoperation.md
  - ../entities/paper-twist2.md
sources:
  - ../../sources/papers/humanoid_rl_stack_09_twist_teleoperated_whole_body_imitation_system.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/papers/bfm_awesome_twist_corl_2025.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
---

# TWIST

**TWIST**（*Teleoperated Whole-Body Imitation System*，arXiv:2505.02833，CoRL 2025）是全身人形遥操作与模仿学习系统。

## 一句话定义

TWIST 和 H2O / OmniH2O 属于同一条技术线，但它强调的是另一个问题：**人形机器人的遥操作不能只控制手，也不能只控制底盘，而是要控制整具身体。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| BFM | Behavior Foundation Model | 大规模行为数据预训练的可复用全身行为先验 |
| BC | Behavior Cloning | 从专家示范直接回归动作的行为克隆 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **01 数据 · 重定向 · 遥操作**（#09/42）。
- TWIST 和 H2O / OmniH2O 属于同一条技术线，但它强调的是另一个问题：**人形机器人的遥操作不能只控制手，也不能只控制底盘，而是要控制整具身体。**
- 很多移动操作系统会把问题拆开：底盘负责移动，手臂负责操作，头部或相机负责观察。这样做工程上可控，但人形机器人不是简单的“移动底盘 + 双臂机械臂”。
- 它的腰、腿、手臂、头部和重心会互相影响。人一边走一边伸手、一边转身一边操作时，身体协调本身就是任务的一部分。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 论文 | <https://arxiv.org/abs/2505.02833> |
| 项目页 | <https://yanjieze.com/TWIST/> |
| 代码 | <https://github.com/YanjieZe/TWIST> |
| 机构 | 斯坦福大学；西蒙弗雷泽大学 |

### 在 42 篇 RL 运动控制身体系统栈中

| 字段 | 内容 |
|------|------|
| 编号 | 09/42 |
| 系统栈层 | 01 数据 · 重定向 · 遥操作 |
| 索引来源 | [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) |

### 在 BFM 41 篇技术地图中

| 字段 | 内容 |
|------|------|
| 编号 | 11/41 |
| 分组 | 02 Goal-conditioned 学习 |
| 索引来源 | [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) |

## 核心机制（归纳）

### 1）策展导读要点

TWIST 和 H2O / OmniH2O 属于同一条技术线，但它强调的是另一个问题：**人形机器人的遥操作不能只控制手，也不能只控制底盘，而是要控制整具身体。**

### 2）策展导读要点

很多移动操作系统会把问题拆开：底盘负责移动，手臂负责操作，头部或相机负责观察。这样做工程上可控，但人形机器人不是简单的“移动底盘 + 双臂机械臂”。

### 3）策展导读要点

它的腰、腿、手臂、头部和重心会互相影响。人一边走一边伸手、一边转身一边操作时，身体协调本身就是任务的一部分。

### 4）策展导读要点

TWIST 的思路是，用运动捕捉系统采集人的全身动作，再通过重定向和强化学习控制器，把人的运动变成机器人可以执行的全身动作。它不是只追求“姿态像人”，而是要让机器人在真实任务里完成全身操作、腿式操作、移动和表达性动作。

## 常见误区

1. 重定向/遥操作不是「训练前脚本」——参考质量上限往往 **早于** RL 策略决定。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 后继系统：[paper-twist2.md](./paper-twist2.md)
- 任务语境：[teleoperation.md](../tasks/teleoperation.md)
- RL 身体系统栈：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- BFM 技术地图：[bfm-41-papers-technology-map.md](../overview/bfm-41-papers-technology-map.md)

## 参考来源

- [humanoid_rl_stack_09_twist_teleoperated_whole_body_imitation_system.md](../../sources/papers/humanoid_rl_stack_09_twist_teleoperated_whole_body_imitation_system.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 42 篇总表
- [bfm_awesome_twist_corl_2025.md](../../sources/papers/bfm_awesome_twist_corl_2025.md) — awesome-bfm 策展摘录
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md) — 41+10 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — RL 运动控制微信公众号编译导读
- [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md) — BFM 41 篇微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)
- 论文：<https://arxiv.org/abs/2505.02833>

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) — 完整列表与数据集表
- [TWIST2](./paper-twist2.md) — 可扩展数据采集后继系统
