---
type: entity
tags: [paper, survey, tactile, tactile-sensing, representation-learning, foundation-model, sim2real, review]
status: complete
updated: 2026-10-15
project_id: tactile-guided-robot-learning-survey
arxiv: "2610.20134"
project: https://sites.google.com/view/touch-in-robot-learning-review/home
related:
  - ../concepts/tactile-sensing.md
  - ../concepts/visuo-tactile-fusion.md
  - ./paper-tf-art-tactile-force-survey.md
  - ./paper-vision-based-tactile-intelligence.md
  - ../concepts/foundation-policy.md
sources:
  - ../../sources/papers/tactile_guided_survey_arxiv_2610_20134.md
summary: "触觉引导机器人学习综述（arXiv:2610.20134）：按触觉在学习与控制流程中的功能组织文献，覆盖表示学习、策略、预测控制与触觉基础模型，并讨论跨物体/传感器/形态与 sim2real 泛化；综述无代码，配套文献库为资源索引。"
---

# 触觉引导机器人学习综述：从表示学习到基础模型

**Tactile-Guided Robot Learning: From Representation Learning to Foundation Models**（[arXiv:2610.20134](https://arxiv.org/abs/2610.20134)，[配套文献库](https://sites.google.com/view/touch-in-robot-learning-review/home)）是一篇按触觉功能而非传感器类型组织的综述，目的是给研究者一张触觉机器人学习的研究地图。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| sim2real | Simulation to Real | 综述列为泛化维度之一的仿真到真实迁移 |
| FM | Foundation Model | 综述末尾讨论的触觉基础模型方向 |
| WM | World Model | 综述提出的触觉基础模型与世界模型方向 |

## 为什么重要

- **触觉从辅助信号走向核心模态：** 综述主张触觉正在成为通用具身智能的基础模态，而不只是任务专用的反馈。
- **按功能组织便于选型：** 表示、策略、预测控制与基础模型各自的问题不同，按功能分章比按传感器分章更利于判断该读哪一块。
- **明确列出开放问题：** 物理表示迁移、语义推理与接触级控制结合、异构数据扩展，是后续研究可以直接对接的题目。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 未核实（页面未列机构） |
| **作者** | Kiyanoush Nazari, Luchen Li, Haoshu Fang, Nima Fazeli, Zhibin Li, Fumiya Iida, Thomas George Thuruthel |
| **类型** | 综述；配套文献数据库按条目链接论文、项目页与代码仓库 |
| **开源** | **未开源**（综述无自有代码；截至 2026-10-15 配套站点为文献索引，条目级链接未逐条核实） |

## 核心原理

综述不提出单一方法，核心是一套按功能组织的框架：

| 章节主题（按功能） | 要解决的问题 | 读法 |
|--------------------|--------------|------|
| 触觉表示学习 | 如何把触觉信号编码成可迁移的特征 | 先读，是后续策略与基础模型的前提 |
| 触觉条件策略学习 | 触觉如何进入策略的观测与动作 | 对照视觉策略，看哪些设计可复用 |
| 预测与自适应控制 | 触觉反馈如何用于接触预测与调节 | 与经典接触控制方法衔接 |
| 触觉基础模型 | 大规模预训练能否覆盖跨任务与跨传感器 | 偏前沿，证据较少 |
| 泛化与 sim2real | 跨物体、传感器、形态与仿真迁移 | 作为选型时的约束清单 |

## 工程实践

| 项 | 建议 |
|----|------|
| 入门路径 | 先读触觉表示与感知概念，再看策略与基础模型章节 |
| 选型约束 | 以「跨传感器、跨形态」作为检查项，避免只在单一传感器上验证 |
| 文献跟踪 | 用配套文献库跟踪条目，但需逐条核对原文与代码状态 |
| 源码运行时序图 | **不适用**（综述无可运行代码） |

## 实验与评测

综述本身不做新实验。论文关注的评价维度是跨物体、任务、传感器、形态与 sim2real 的泛化能力；具体各方法的数字以原文与被引论文为准，本页不引用摘要之外的数值。

## 结论

**一句总判：触觉学习已从单点反馈走向表示与基础模型，但跨传感器和跨形态的泛化仍是主要缺口。**

1. **先看表示，再看策略** — 触觉表示是多数下游方法的瓶颈，综述的组织方式也体现这一点。
2. **泛化维度要单独评估** — 跨物体、传感器、形态与 sim2real 各自独立，不能用单一场景的结果代替。
3. **语义与接触需要结合** — 综述指出，高层语义推理与接触级控制之间的衔接仍不成熟。
4. **基础模型方向证据较少** — 触觉基础模型与世界模型仍处于提出与探索阶段，选型时应视为研究方向。
5. **综述适合做地图，不适合做性能排名** — 本页不给出方法排名，需回到原文核对条目。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [触觉感知](../concepts/tactile-sensing.md) | 概念页讲传感原理；本综述讲触觉如何进入学习系统 |
| [视触觉融合](../concepts/visuo-tactile-fusion.md) | 融合是综述的一个子问题，综述覆盖范围更广 |
| [TF-ART 触觉力综述](./paper-tf-art-tactile-force-survey.md) | 同为触觉综述，本综述更侧重学习与基础模型 |
| [视觉触觉智能综述](./paper-vision-based-tactile-intelligence.md) | 同为视觉触觉方向综述，可对照其分类方式 |

## 局限与风险

- **综述不是实验证据：** 结论依赖被引文献的质量，本页不做重复验证。
- **条目链接未逐条核实：** 配套站点的条目级外部链接在抽取中不可见，使用前需要逐条打开。
- **时效性：** 2026-10 发布，领域变化快，基础模型部分尤其需要跟进。
- **作者与机构未核实：** 作者列表来自 arXiv 摘要页，机构未在入库时核实。

## 关联页面

- [触觉感知](../concepts/tactile-sensing.md) — 传感原理背景
- [视触觉融合](../concepts/visuo-tactile-fusion.md) — 融合子问题
- [TF-ART 触觉力综述](./paper-tf-art-tactile-force-survey.md) — 同类综述对照
- [视觉触觉智能综述](./paper-vision-based-tactile-intelligence.md) — 同类综述对照
- [Foundation Policy](../concepts/foundation-policy.md) — 基础模型方向的背景

## 参考来源

- [tactile_guided_survey_arxiv_2610_20134.md](../../sources/papers/tactile_guided_survey_arxiv_2610_20134.md) — 综述摘录与开源核查
- [arXiv:2610.20134](https://arxiv.org/abs/2610.20134) — 原文（v1，2026-10-14）

## 推荐继续阅读

- [配套文献库](https://sites.google.com/view/touch-in-robot-learning-review/home) — 按条目索引的论文与资源
- [arXiv 原文](https://arxiv.org/abs/2610.20134) — 综述全文入口
