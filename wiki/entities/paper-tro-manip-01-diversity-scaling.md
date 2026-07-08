---
type: entity
tags: [paper, manipulation, tro-manip-survey, imitation-learning, data-scaling, hku, agibot, buaa]
status: complete
updated: 2026-07-08
arxiv: "2507.06219"
summary: "实证任务/本体/演示者三维数据多样性对操作 scaling 的影响；分布去偏得 GO-1-Pro（+15%，等效 2.5× 预训练数据）。"
related:
  - ../overview/tro-manip-5-papers-technology-map.md
  - ../overview/tro-manip-category-01-data-scaling.md
  - ../tasks/manipulation.md
  - ../methods/behavior-cloning.md
sources:
  - ../../sources/papers/tro_manip_survey_01_diversity_scaling.md
  - ../../sources/blogs/wechat_shenlan_tro_manip_5_papers_survey.md
  - ../../sources/papers/tro_manip_5_papers_catalog.md
---

# Is Diversity All You Need for Scalable Robotic Manipulation?

**Is Diversity All You Need** 收录于 [深蓝具身智能 · T-RO 2026 操作学习精选](https://mp.weixin.qq.com/s/nswA-jCGC3kr9iQjhRRuXQ) **第 01/5** 篇，归类为 **01 数据规模化**。

## 一句话定义

系统实证 **任务 / 本体 / 演示者** 三维数据多样性对操作 scaling 的影响，并提出分布去偏方法得到 **GO-1-Pro**（相对基线 +15%，等效 2.5× 预训练数据）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IL | Imitation Learning | 模仿学习 / 从演示学习 |
| LfD | Learning from Demonstrations | 从演示中学习 |
| OOD | Out-of-Distribution | 分布外泛化 |

## 为什么重要

- 挑战「数据多样性越高越好」的直觉，给出 **可操作的 scaling 指南**。
- **任务多样性** 比单任务演示堆量更关键；场景多样性对分布偏移鲁棒性尤其重要。
- **跨本体预训练非必要**：高质量单本体数据微调时的 scaling 可能优于多本体预训练，降低采集成本。
- **演示者多样性可能有害**：人类示教的速度多模态性（velocity multimodality）会干扰策略学习。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 01/5 |
| 分组 | 01 数据规模化 |
| 机构 | 香港大学（HKU）；智元机器人（AgiBot）；北京航空航天大学（BUAA）等 |
| 出处 | IEEE T-RO 2026, vol. 42, pp. 1872–1883 · arXiv:2507.06219 |
| 论文/项目 | <https://arxiv.org/abs/2507.06219> |

## 核心机制（归纳）

### 1）三维多样性分解

- **任务多样性**：做什么（场景/技能组合）。
- **本体多样性**：哪款机器人示教与部署。
- **演示者多样性**：不同人类操作者的习惯与随机性。

### 2）分布去偏 → GO-1-Pro

- 针对演示者引入的 **速度歧义 / 速度多模态性**，提出 **distribution debiasing**。
- **GO-1-Pro** 在策展导读中报告 **+15%** 性能提升，等效 **2.5×** 预训练数据效率。

## 常见误区

1. 把「更多演示者」等同于「更多多样性收益」——文内指出演示者差异可能 **confound** 策略学习。
2. 策展编译不能替代原文消融；量化指标以 T-RO / arXiv PDF 为准。

## 实验与评测

- 多机器人平台上的大量实验；具体 benchmark 与消融以原文为准（见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 技术地图：[tro-manip-5-papers-technology-map.md](../overview/tro-manip-5-papers-technology-map.md)
- 分类 hub：[tro-manip-category-01-data-scaling.md](../overview/tro-manip-category-01-data-scaling.md)
- 任务页：[Manipulation](../tasks/manipulation.md)

## 参考来源

- [tro_manip_survey_01_diversity_scaling.md](../../sources/papers/tro_manip_survey_01_diversity_scaling.md) — T-RO 5 篇策展摘录
- [tro_manip_5_papers_catalog.md](../../sources/papers/tro_manip_5_papers_catalog.md)
- [wechat_shenlan_tro_manip_5_papers_survey.md](../../sources/blogs/wechat_shenlan_tro_manip_5_papers_survey.md)
- 论文：<https://arxiv.org/abs/2507.06219>

## 推荐继续阅读

- [T-RO 5 篇技术地图](../overview/tro-manip-5-papers-technology-map.md)
- [Behavior Cloning](../methods/behavior-cloning.md)
