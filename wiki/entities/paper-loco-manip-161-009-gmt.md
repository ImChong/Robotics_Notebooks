---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid, motion-tracking, gmt, ucsd, sfu]
status: complete
updated: 2026-07-21
venue: curated
summary: "Loco-Manip 161 篇 #009 索引：GMT（arXiv:2506.14770）用 Adaptive Sampling + Motion MoE 训练单一统一全身跟踪策略；完整方法见 paper-gmt.md。注意：公众号策展曾误写扩散/流匹配，应以 arXiv/项目页为准。"
related:
  - ./paper-gmt.md
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-01-motion-base-wbt.md
  - ../tasks/loco-manipulation.md
  - ../concepts/whole-body-tracking-pipeline.md
sources:
  - ../../sources/papers/gmt_arxiv_2506_14770.md
  - ../../sources/papers/loco_manip_161_survey_009_gmt.md
  - ../../sources/sites/gmt-humanoid-github-io.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# GMT（Loco-Manip 161 · #009 索引）

**GMT** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 009/161** 篇，归类为 **01 运控基座与通用全身跟踪**。本页为 **策展地图坐标**；方法深读与开源边界见正式实体页 [GMT（arXiv:2506.14770）](./paper-gmt.md)。

## 一句话定义

**161 篇地图中的 GMT 条目：指向「Adaptive Sampling + Motion MoE 的统一人形全身跟踪」正式实体页。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GMT | General Motion Tracking | 统一全身运动跟踪框架（Chen et al., 2025） |
| MoE | Mixture-of-Experts | GMT 教师策略的运动专家混合结构 |
| WBC | Whole-Body Control | 协调全身关节的控制层语境 |
| Loco-Manip | Loco-Manipulation | 161 篇地图的任务域标签 |

## 为什么重要

- 在 161 篇 **运控基座** 分组中占 **#009**，常被下游（ResMimic / PhyGile / EGM 等）当作「通用 tracker」参照。
- 公众号一句话曾误写为扩散/流匹配生成——**应以 [正式实体页](./paper-gmt.md) 与 arXiv 为准**。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 009/161 |
| 分组 | 01 运控基座与通用全身跟踪 |
| 原文题目 | GMT: General Motion Tracking for Humanoid Whole-Body Control |
| 机构 | 加州大学圣地亚哥分校（UCSD）、西蒙菲莎大学（SFU） |
| 发表 | arXiv:2506.14770（2025） |
| 论文/项目 | <https://gmt-humanoid.github.io/> |
| 正式实体 | [paper-gmt.md](./paper-gmt.md) |

## 核心机制（归纳）

- **正确要点（来自论文，非公众号误述）：** 两阶段教师–学生；**Adaptive Sampling** 平衡难易片段；**Motion MoE** 提升广谱动作表达；局部 key body + 未来窗运动输入；AMASS+LAFAN1 策展约 **8925 clips**。
- **策展误述（勿沿用）：** 「扩散策略/流匹配采样可执行轨迹」——那是动作生成族叙述，**不是 GMT 本体**。

## 评测与指标（索引级）

- 量化指标与真机视频见 [paper-gmt.md](./paper-gmt.md) 与 [项目页](https://gmt-humanoid.github.io/)。
- 横向地图：[分类 hub](../overview/loco-manip-161-category-01-motion-base-wbt.md)、[技术地图](../overview/humanoid-loco-manip-161-papers-technology-map.md)。

## 常见误区

1. 把 161 篇一句话当方法定义 → 先读 [paper-gmt.md](./paper-gmt.md)。
2. 把 GMT 当成扩散动作生成器 → 它是 **RL 跟踪策略**；MDM 仅是下游跟踪试验。

## 与其他页面的关系

- 正式实体：[paper-gmt.md](./paper-gmt.md)
- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-01-motion-base-wbt.md](../overview/loco-manip-161-category-01-motion-base-wbt.md)

## 参考来源

- [gmt_arxiv_2506_14770.md](../../sources/papers/gmt_arxiv_2506_14770.md) — 论文摘录（权威）
- [gmt-humanoid-github-io.md](../../sources/sites/gmt-humanoid-github-io.md) — 项目页
- [loco_manip_161_survey_009_gmt.md](../../sources/papers/loco_manip_161_survey_009_gmt.md) — 161 策展摘录（含历史误述，已交叉更正）
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)

## 推荐继续阅读

- [GMT 正式实体页](./paper-gmt.md)
- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
- 项目页：<https://gmt-humanoid.github.io/>
