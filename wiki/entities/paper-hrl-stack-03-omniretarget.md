---
type: entity
tags: [paper, humanoid, rl, motion-retargeting, motion-control, interaction-mesh, loco-manipulation, data-generation, amazon-far, body-system-stack]
status: complete
updated: 2026-05-31
arxiv: "2509.26633"
venue: curated
summary: "OmniRetarget 用 interaction mesh + 硬约束优化生成交互保留的人形运动学参考，并支持单演示增广；是 PHP 等下游感知跑酷/操作论文的原子技能重定向上游。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ./paper-hrl-stack-22-perceptive_humanoid_parkour.md
  - ../concepts/motion-retargeting.md
  - ../methods/motion-retargeting-gmr.md
  - ./unitree-g1.md
sources:
  - ../../sources/papers/omniretarget_arxiv_2509_26633.md
  - ../../sources/papers/humanoid_rl_stack_03_omniretarget_interaction_preserving_data_generat.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
---

# OmniRetarget

**OmniRetarget**（OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction，arXiv:[2509.26633](https://arxiv.org/abs/2509.26633)，[项目页](https://omniretarget.github.io/)）是 Amazon FAR 团队的**交互保留人形运动重定向与数据生成**引擎：用 **interaction mesh** 显式建模人/机器人与物体、地形之间的空间关系，在**硬运动学约束**下最小化 Laplacian 形变能，生成可下游 RL 跟踪的高质量参考，并能把**单条人类演示**增广到多种 embodiment、地形与物体配置。

本页在 [42 篇 humanoid RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中编号 **03/42**（**01 数据 · 重定向 · 遥操作**）。下游应用示例：[Perceptive Humanoid Parkour（PHP）](./paper-hrl-stack-22-perceptive_humanoid_parkour.md) 用其构建跑酷**原子技能库**。

## 为什么重要

- **关键词是 interaction-preserving：** 不仅匹配关键点，而是保留 agent–object–terrain 的相对几何与接触关系，缓解传统 PHC/GMR 的穿透、脚滑与「场景盲」。
- **硬约束 + 可增广：** Sequential SOCP 每帧求解；固定源 mesh、变化目标场景即可批量生成新轨迹——论文报告 **8+ 小时**数据，kinematic 指标优于常用基线。
- **极简下游 RL：** 与 BeyondMimic 叙事一致，**5 项 reward** + 轻量 DR、**无 curriculum** 即可在 G1 上零样本实机长时程 parkour / loco-manipulation（最长约 **30 s**）。

## 核心机制（简表）

| 组件 | 作用 |
|------|------|
| Interaction mesh | 关键关节 + 物体/环境采样点 Delaunay 四面体；Laplacian 坐标差最小化 |
| 硬约束 | 非穿透（SDF）、关节/速度界、stance 脚位置固定（防 foot skating） |
| 增广 | 物体位姿/形状、地形高度等变化 → 重新优化目标 mesh |
| 下游 | DeepMimic 式 tracking + 共享 DR；多任务共享奖励形式 |

## 实验与评测

- **数据规模：** OMOMO、LAFAN1、自采 MoCap → **8+ 小时**重定向轨迹。
- **Kinematic 指标（论文 Table II 摘要）：** 相对 PHC / GMR / VideoMimic，**穿透时长、脚滑、接触保留**与下游 RL **成功率**更优（具体数值以原文为准）。
- **下游 RL：** Unitree G1 上多类 loco-manipulation / parkour；**5 项 reward**、共享 DR、**无 curriculum**；最长约 **30 s** 实机序列（含搬运、攀台、翻滚等）。

## 与其他工作对比

| 方法 | 硬约束 | 物体/地形交互 | 数据增广 |
|------|--------|---------------|----------|
| PHC / GMR | 弱/无 | 通常无 | 无 |
| VideoMimic | 软惩罚 | 地形为主 | 无 |
| **OmniRetarget** | 有 | 有 | 有 |

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 03/42 |
| 系统栈层 | 01 数据 · 重定向 · 遥操作 |
| 机构 | Amazon FAR；MIT；UC Berkeley；Stanford；CMU |
| 链接 | <https://omniretarget.github.io/> · <https://arxiv.org/abs/2509.26633> |

## 与其他页面的关系

- 下游跑酷：[PHP（2602.15827）](./paper-hrl-stack-22-perceptive_humanoid_parkour.md)
- 问题域：[Motion Retargeting](../concepts/motion-retargeting.md)、[GMR](../methods/motion-retargeting-gmr.md)
- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Manipulation | Robot Manipulation | 抓取、移动、操作物体的任务总称 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| GMR | General Motion Retargeting | 把人体/视频动作重定向为机器人可执行参考 |
| DR | Domain Randomization | 训练时随机化仿真参数以提升跨域鲁棒迁移 |
| G1 | Unitree G1 Humanoid | 宇树入门级教育科研人形平台 |
| MoCap | Motion Capture | 动作捕捉，参考动作与演示数据的主要来源 |
| Retargeting | Motion Retargeting | 将人体/动物动作映射到目标机器人骨架 |

## 参考来源

- [omniretarget_arxiv_2509_26633.md](../../sources/papers/omniretarget_arxiv_2509_26633.md) — 论文摘要与方法摘录（主归档）
- [humanoid_rl_stack_03_omniretarget_interaction_preserving_data_generat.md](../../sources/papers/humanoid_rl_stack_03_omniretarget_interaction_preserving_data_generat.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读

## 推荐继续阅读

- arXiv：<https://arxiv.org/abs/2509.26633>
- 项目页：<https://omniretarget.github.io/>
- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
