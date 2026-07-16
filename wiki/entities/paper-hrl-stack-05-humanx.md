---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, shanghai-ai-lab, hkust]
status: complete
updated: 2026-07-16
venue: curated
summary: "HumanX 也从视频出发，但它关心的是 agile and generalizable humanoid interaction skills。它想把人类视频转成机器人可学习的交互技能，覆盖篮球、足球、羽毛球等任务。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-03-data-pipeline.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_05_humanx_toward_agile_and_generalizable_humanoid_i.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
---

# HumanX

**HumanX** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 05/42** 篇，归类为 **01 数据 · 重定向 · 遥操作**。

## 一句话定义

HumanX 也从视频出发，但它关心的是 agile and generalizable humanoid interaction skills。它想把人类视频转成机器人可学习的交互技能，覆盖篮球、足球、羽毛球等任务。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **01 数据 · 重定向 · 遥操作**（#05/42）。
- HumanX 也从视频出发，但它关心的是 agile and generalizable humanoid interaction skills。它想把人类视频转成机器人可学习的交互技能，覆盖篮球、足球、羽毛球等任务。
- 它的框架包括两个部分：XGen 和 XMimic。XGen 用来从视频中合成物理合理的交互数据，并支持对象 mesh、尺寸、轨迹等增强；XMimic 则学习这些数据里的泛化交互技能。
- 这篇论文最重要的不是“机器人会打球”，而是数据生产思路：真实机器人交互数据贵，人类视频多，但视频里的动作和机器人身体不匹配，物体状态也不一定完整。HumanX 试图把视频中的人-物交互转换成机器人可以训练的数据。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 05/42 |
| 系统栈层 | 01 数据 · 重定向 · 遥操作 |
| 机构 | 香港科技大学；上海人工智能实验室 |
| 出处 | curated |
| 链接 | <https://wyhuai.github.io/human-x/> |

## 核心机制（归纳）

### 1）策展导读要点

HumanX 也从视频出发，但它关心的是 agile and generalizable humanoid interaction skills。它想把人类视频转成机器人可学习的交互技能，覆盖篮球、足球、羽毛球等任务。

### 2）策展导读要点

它的框架包括两个部分：XGen 和 XMimic。XGen 用来从视频中合成物理合理的交互数据，并支持对象 mesh、尺寸、轨迹等增强；XMimic 则学习这些数据里的泛化交互技能。

### 3）策展导读要点

这篇论文最重要的不是“机器人会打球”，而是数据生产思路：真实机器人交互数据贵，人类视频多，但视频里的动作和机器人身体不匹配，物体状态也不一定完整。HumanX 试图把视频中的人-物交互转换成机器人可以训练的数据。

### 4）策展导读要点

它和 OmniRetarget 的区别在于：OmniRetarget 更像一个交互保留的重定向引擎，HumanX 更像从人类视频到机器人交互技能的完整管线。

## 常见误区

1. 重定向/遥操作不是「训练前脚本」——参考质量上限往往 **早于** RL 策略决定。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_05_humanx_toward_agile_and_generalizable_humanoid_i.md](../../sources/papers/humanoid_rl_stack_05_humanx_toward_agile_and_generalizable_humanoid_i.md)

## 参考来源

- [humanoid_rl_stack_05_humanx_toward_agile_and_generalizable_humanoid_i.md](../../sources/papers/humanoid_rl_stack_05_humanx_toward_agile_and_generalizable_humanoid_i.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [机器人论文阅读笔记：HumanX](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/HumanX__Toward_Agile_and_Generalizable_Humanoid_Interaction_Skills_from_Human_Vi/HumanX__Toward_Agile_and_Generalizable_Humanoid_Interaction_Skills_from_Human_Vi.html)
- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
