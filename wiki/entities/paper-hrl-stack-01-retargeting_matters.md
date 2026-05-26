---
type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack]
status: complete
updated: 2026-05-26
summary: "GMR 的核心命题很直接：**retargeting matters**。论文指出，humanoid motion tracking policies 依赖人类动作重定向，但人和机器人之间存在 **embodiment gap**。重定向阶段留下的脚滑、不可行姿态、起始姿态不合理等问题，会直接影响后面的 **RL 控制器**。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../methods/motion-retargeting-gmr.md
sources:
  - ../../sources/papers/humanoid_rl_stack_01_retargeting_matters_general_motion_retargeting_f.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
---

# Retargeting Matters

**Retargeting Matters** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 01/42** 篇，归类为 **01 数据 · 重定向 · 遥操作**。本页为知识库 **策展摘要**；方法细节以论文 PDF 与项目页为准。

## 为什么重要

- GMR 的核心命题很直接：**retargeting matters**。论文指出，humanoid motion tracking policies 依赖人类动作重定向，但人和机器人之间存在 **embodiment gap**。重定向阶段留下的脚滑、不可行姿态、起始姿态不合理等问题，会直接影响后面的 **RL 控制器**。
- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 的八层框架中，属于 **01 数据 · 重定向 · 遥操作** 簇。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 01/42 |
| 系统栈层 | 01 数据 · 重定向 · 遥操作 |
| 机构 | 斯坦福大学 |
| 出处 | curated |
| 链接 | <https://jaraujo98.github.io/retargeting\_matters/> |

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_01_retargeting_matters_general_motion_retargeting_f.md](../../sources/papers/humanoid_rl_stack_01_retargeting_matters_general_motion_retargeting_f.md)

## 参考来源

- [humanoid_rl_stack_01_retargeting_matters_general_motion_retargeting_f.md](../../sources/papers/humanoid_rl_stack_01_retargeting_matters_general_motion_retargeting_f.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
