---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, amp, motion-prior, tsinghua, shanghai-pil]
status: complete
updated: 2026-06-18
venue: curated
summary: "Deep Whole-body Parkour：全身跑酷与 PHP 同簇但侧重点不同；在 RL 身体系统栈属感知式高动态运动，在 AMP 专题属交互与长时程簇。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-01-locomotion-base.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../tasks/stair-obstacle-perceptive-locomotion.md
  - ../entities/project-instinct.md
sources:
  - ../../sources/papers/humanoid_rl_stack_23_deep_whole_body_parkour.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/papers/humanoid_amp_survey_18_deep_whole_body_parkour.md
  - ../../sources/papers/humanoid_amp_survey_19_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
---

# Deep Whole-body Parkour

**Deep Whole-body Parkour** 与 PHP 同属全身跑酷簇，但侧重点不同：强调深度感知下的全身协调穿越障碍。本页为知识库 **策展摘要**；方法细节以论文 PDF 与项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |
| PHP | Perceptive Humanoid Parkour | 感知式人形跑酷，与本页同属 Project Instinct 跑酷簇 |

## 为什么重要

- 在 [运动小脑 64 篇技术地图](../overview/humanoid-motion-cerebellum-technology-map.md) 中归类为 **A 走路底座**（8/64）：底座：手脚躯干一起参与跑酷。
- Deep Whole-body Parkour 和 PHP 都做跑酷，但侧重点不同。
- 不是 AMP 主线论文，但对理解人形运动系统与感知 locomotion 很重要。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 项目页 | <https://project-instinct.github.io/deep-whole-body-parkour> |
| 机构 | 清华大学交叉信息研究院；上海期智研究院 |

### 在 42 篇 RL 运动控制身体系统栈中

| 字段 | 内容 |
|------|------|
| 编号 | 23/42 |
| 系统栈层 | 03 感知式高动态运动 |
| 索引来源 | [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) |

### 在 AMP 19 篇运动先验专题中

| 字段 | 内容 |
|------|------|
| 编号 | 18/19 |
| 叙事段 | 04 交互与长时程 |
| 索引来源 | [具身智能研究室 · AMP 运动先验专题](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w) |

## 与其他页面的关系

- RL 身体系统栈：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 综述：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 任务语境：[stair-obstacle-perceptive-locomotion.md](../tasks/stair-obstacle-perceptive-locomotion.md)
- 团队：[project-instinct.md](./project-instinct.md)

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **原文 PDF / 项目页** 为准（链接见 [参考来源](#参考来源) 与上文 **核心信息** 表）。
- 若需与姊妹篇对照，请回到对应 **技术地图 / 42 篇栈 / AMP 专题** 总览中的实验段落。

## 参考来源

- [humanoid_rl_stack_23_deep_whole_body_parkour.md](../../sources/papers/humanoid_rl_stack_23_deep_whole_body_parkour.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 42 篇总表
- [humanoid_amp_survey_18_deep_whole_body_parkour.md](../../sources/papers/humanoid_amp_survey_18_deep_whole_body_parkour.md) — AMP 专题策展摘录
- [humanoid_amp_survey_19_catalog.md](../../sources/papers/humanoid_amp_survey_19_catalog.md) — AMP 19 篇总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — RL 运动控制微信公众号编译导读
- [wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md) — AMP 专题微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)、[wechat_humanoid_amp_19_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_amp_19_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [AMP 专题长文（微信公众号）](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
