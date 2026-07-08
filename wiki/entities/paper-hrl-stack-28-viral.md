---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, cuhk, nvidia, berkeley, cmu]
status: complete
updated: 2026-07-08
venue: curated
summary: "VIRAL 做的是 Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation。它要让人形机器人仅凭机载 RGB 摄像头，在仿真中训练后零样本迁移到真实机器人，完成移动抓取和放置任务。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../entities/paper-viral-humanoid-visual-sim2real.md
sources:
  - ../../sources/papers/humanoid_rl_stack_28_viral_visual_sim_to_real_at_scale_for_humanoid_l.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
---

# VIRAL

**VIRAL** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 28/42** 篇，归类为 **04 视觉闭环 · 任务接口 · 世界模型**。

> **深读页：** [paper-viral-humanoid-visual-sim2real](../entities/paper-viral-humanoid-visual-sim2real.md) — 方法机制与实验细节见链接页；本页保留 survey 坐标与交叉引用。

## 一句话定义

VIRAL 做的是 Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation。它要让人形机器人仅凭机载 RGB 摄像头，在仿真中训练后零样本迁移到真实机器人，完成移动抓取和放置任务。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| Manipulation | Robot Manipulation | 抓取、移动、操作物体的任务总称 |
| RGB | Red-Green-Blue | 彩色图像通道，常与深度 (RGB-D) 配合 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **04 视觉闭环 · 任务接口 · 世界模型**（#28/42）。
- VIRAL 做的是 Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation。它要让人形机器人仅凭机载 RGB 摄像头，在仿真中训练后零样本迁移到真实机器人，完成移动抓取和放置任务。
- VIRAL 的整体结构是 teacher-student。教师在仿真里拥有 privileged state，学习长时程 loco-manipulation；学生只看 RGB 和本体感知，通过 DAgger 和行为克隆蒸馏教师能力。
- 论文还展示了训练规模的重要性。低计算规模下 teacher 和 student 都容易失败；扩大仿真规模后，视觉策略才更稳定。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 28/42 |
| 系统栈层 | 04 视觉闭环 · 任务接口 · 世界模型 |
| 机构 | NVIDIA；CMU；伯克利；香港中文大学 |
| 出处 | curated |
| 链接 | <https://viral-humanoid.github.io> |

## 核心机制（归纳）

### 1）策展导读要点

VIRAL 做的是 Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation。它要让人形机器人仅凭机载 RGB 摄像头，在仿真中训练后零样本迁移到真实机器人，完成移动抓取和放置任务。

### 2）策展导读要点

VIRAL 的整体结构是 teacher-student。教师在仿真里拥有 privileged state，学习长时程 loco-manipulation；学生只看 RGB 和本体感知，通过 DAgger 和行为克隆蒸馏教师能力。

### 3）策展导读要点

论文还展示了训练规模的重要性。低计算规模下 teacher 和 student 都容易失败；扩大仿真规模后，视觉策略才更稳定。

### 4）策展导读要点

我的判断**VIRAL 的价值不只是一个任务成功，而是把视觉 sim-to-real 变成一套可扩展工程流程。**

## 常见误区

1. VLA/世界模型条目解决 **接口与预测**，不自动替代已封装的底层 WBC 能力。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_28_viral_visual_sim_to_real_at_scale_for_humanoid_l.md](../../sources/papers/humanoid_rl_stack_28_viral_visual_sim_to_real_at_scale_for_humanoid_l.md)

## 参考来源

- [humanoid_rl_stack_28_viral_visual_sim_to_real_at_scale_for_humanoid_l.md](../../sources/papers/humanoid_rl_stack_28_viral_visual_sim_to_real_at_scale_for_humanoid_l.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
