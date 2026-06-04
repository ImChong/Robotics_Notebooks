---
type: entity
tags: [paper, humanoid, amp, motion-prior, adversarial-imitation]
status: complete
updated: 2026-06-04
venue: curated
summary: "PhysHSI 把 AMP 用到 humanoid-scene interaction 里：搬箱子、坐下、躺下、站起，以及风格化行走。"
related:
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ./paper-splitadapter-load-aware-loco-manipulation.md
  - ../tasks/loco-manipulation.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/humanoid_amp_survey_15_physhsi_towards_a_real_world_generalizable_and_n.md
  - ../../sources/papers/humanoid_amp_survey_19_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md
---

# PhysHSI

**PhysHSI** 收录于 [具身智能研究室 · AMP 运动先验专题](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w) **第 15/19** 篇，归类为 **04 交互与长时程**。本页为知识库 **策展摘要**。

## 为什么重要

- PhysHSI 把 AMP 用到 humanoid-scene interaction 里：搬箱子、坐下、躺下、站起，以及风格化行走。
- 在 [人形 AMP 运动先验综述](../overview/humanoid-amp-motion-prior-survey.md) 中与 mimic / 身体系统栈分工对照阅读。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 15/19 |
| 叙事段 | 04 交互与长时程 |
| 机构 | 上海人工智能实验室、香港科技大学 |
| 出处 | curated |
| 链接 | 见论文标题检索 |

## 与其他页面的关系

- AMP 综述：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 身体系统栈：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- 原始 source：[humanoid_amp_survey_15_physhsi_towards_a_real_world_generalizable_and_n.md](../../sources/papers/humanoid_amp_survey_15_physhsi_towards_a_real_world_generalizable_and_n.md)
- **下游适配：** [SplitAdapter](./paper-splitadapter-load-aware-loco-manipulation.md)（arXiv:2606.03297）以 PhysHSI 类 **AMP 搬箱策略为冻结基线**，用因子化世界模型/FiLM 做负载感知 sim2real 适配。

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **原文 PDF / 项目页** 为准（链接见 [参考来源](#参考来源) 与上文 **核心信息** 表）。
- 若需与姊妹篇对照，请回到对应 **技术地图 / 42 篇栈 / AMP 专题** 总览中的实验段落。


## 参考来源

- [humanoid_amp_survey_15_physhsi_towards_a_real_world_generalizable_and_n.md](../../sources/papers/humanoid_amp_survey_15_physhsi_towards_a_real_world_generalizable_and_n.md)
- [humanoid_amp_survey_19_catalog.md](../../sources/papers/humanoid_amp_survey_19_catalog.md)
- [wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)
- 原始抓取：[wechat_humanoid_amp_19_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_amp_19_survey_2026-05-26.md)

## 推荐继续阅读

- [AMP 专题长文（微信公众号）](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
- [42 篇 RL 身体系统栈姊妹篇](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
