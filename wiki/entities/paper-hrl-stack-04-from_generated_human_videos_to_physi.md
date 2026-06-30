---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, jku, nyu, berkeley]
status: complete
updated: 2026-06-30
venue: curated
summary: "这篇论文关注一个很值得追踪的问题：视频生成模型越来越强，可以生成各种人类动作视频，那机器人能不能直接执行生成视频里的动作？"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-03-data-pipeline.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_04_from_generated_human_videos_to_physically_plausi.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
---

# From Generated Human Videos to Physically Plausible Robot Trajectories

**From Generated Human Videos to Physically Plausible Robot Trajectories** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 04/42** 篇，归类为 **01 数据 · 重定向 · 遥操作**。

## 一句话定义

这篇论文关注一个很值得追踪的问题：视频生成模型越来越强，可以生成各种人类动作视频，那机器人能不能直接执行生成视频里的动作？

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **01 数据 · 重定向 · 遥操作**（#04/42）。
- 这篇论文关注一个很值得追踪的问题：视频生成模型越来越强，可以生成各种人类动作视频，那机器人能不能直接执行生成视频里的动作？
- 生成视频可能在视觉上合理，但会有形变、遮挡、肢体穿模、动作不连续、人体比例不稳定等问题。对人眼来说，这些瑕疵可能可以忽略；但对机器人来说，一点姿态错误就可能变成无法执行的轨迹。
- 论文提出两阶段管线：先把生成视频 lift 成 4D human representation，再重定向到 humanoid morphology；之后用 GenMimic 这样的 physics-aware RL policy 来跟踪 3D keypoints，并引入 symmetry 和 keypoint-weighted rewards。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 04/42 |
| 系统栈层 | 01 数据 · 重定向 · 遥操作 |
| 机构 | 伯克利；纽约大学；约翰内斯开普勒大学 |
| 出处 | curated |
| 链接 | <https://genmimic.github.io/> |

## 核心机制（归纳）

### 1）策展导读要点

这篇论文关注一个很值得追踪的问题：视频生成模型越来越强，可以生成各种人类动作视频，那机器人能不能直接执行生成视频里的动作？

### 2）策展导读要点

生成视频可能在视觉上合理，但会有形变、遮挡、肢体穿模、动作不连续、人体比例不稳定等问题。对人眼来说，这些瑕疵可能可以忽略；但对机器人来说，一点姿态错误就可能变成无法执行的轨迹。

### 3）策展导读要点

论文提出两阶段管线：先把生成视频 lift 成 4D human representation，再重定向到 humanoid morphology；之后用 GenMimic 这样的 physics-aware RL policy 来跟踪 3D keypoints，并引入 symmetry 和 keypoint-weighted rewards。

### 4）策展导读要点

我的判断**未来视频生成模型可能会成为机器人动作创意来源，但不会直接成为机器人控制器。中间必须有物理过滤和机器人化过程。**

## 常见误区

1. 重定向/遥操作不是「训练前脚本」——参考质量上限往往 **早于** RL 策略决定。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_04_from_generated_human_videos_to_physically_plausi.md](../../sources/papers/humanoid_rl_stack_04_from_generated_human_videos_to_physically_plausi.md)

## 参考来源

- [humanoid_rl_stack_04_from_generated_human_videos_to_physically_plausi.md](../../sources/papers/humanoid_rl_stack_04_from_generated_human_videos_to_physically_plausi.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
