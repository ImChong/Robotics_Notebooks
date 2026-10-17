---
type: entity
project_id: physical-rsi-research-map
papers: ["2610.18842"]
tags: [physical-ai, recursive-self-improvement, embodied-ai, research-map, literature-review, evaluation, safety, harness]
status: complete
updated: 2026-10-17
related:
  - ./physical-rsi.md
  - ./robodojo.md
  - ./paper-harnesspai.md
  - ./paper-harness-vla.md
  - ../queries/rsi-four-tier-five-pushes.md
sources:
  - ../../sources/sites/sail-physical-rsi-research-map.md
summary: "SAIL Research Lab 的 Physical RSI 是一份具身系统递归自我改进研究地图与文献目录，提供 L0–L5 分类及 harness、训练、安全、验证和评测等主题索引；不是单一机器人策略论文或 Physical RSI 1.0 系统。"
---

# Physical RSI Research Map（SAIL Research Lab）

**Physical RSI — Physical Recursive Self-Improvement** 是 SAIL-Research-Lab 维护的具身递归改进研究地图。项目将自己定位为遵循配套论文证据边界的阅读索引，而非排行榜；README 的目录徽章标示 301 项 works/resources。其核心问题是：物理交互能否不仅改变机器人的当前行为，还能因果性地改变其产生、评估或筛选后续改进的机制。

| 项目 | 内容 |
|------|------|
| 主入口 | <https://github.com/SAIL-Research-Lab/Physical-RSI> |
| 维护方 | SAIL-Research-Lab |
| 文献索引 | [paper-reference-index.md](https://github.com/SAIL-Research-Lab/Physical-RSI/blob/main/docs/paper-reference-index.md) |
| 专题材料 | [model_training.md](https://github.com/SAIL-Research-Lab/Physical-RSI/blob/main/docs/model_training.md) |
| 许可 | MIT（仓库）；引用论文与第三方材料各循其许可 |
| 核查日期 | 2026-10-14 |

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RSI | Recursive Self-Improvement | 递归自我改进：改进会因果性地改变后续改进机制 |
| SAIL | SAIL Research Lab | 本研究地图的维护方 |
| MIT | MIT License | 仓库采用的开源许可证 |

## 核心边界

项目强调，不应把所有部署后适应都称作递归自我改进：收集更多数据、修复一次失败或微调策略，若没有保留的变化进入未来改进机制，仍只是适应。其框架把执行系统 \(M=(\Theta,H)\)（模型与 harness）与改进过程 \(\Phi=(Adapt,Train,Ver)\) 分开，并要求区分“任务表现变好”与“产生未来更新的过程变好”。

## L0–L5 能力分类

README 按更新对象与收益迁移范围组织能力层级：

| 层级 | 含义（项目 README 概述） |
|------|-------------------------|
| L0 | 没有自主且被保留的改进闭环 |
| L1 / L3 | 继承的执行系统更新分别改善同一任务 / 迁移到其他任务 |
| L2 / L4 | 继承的改进程序更新分别支持同任务 / 跨任务改进 |
| L5 | 连续的程序修订在匹配预算和受保护的跨任务评测下，使后续改进能力持续提升 |

L5 的表述强调验证和控制变量；仅仅换了程序或得到一次更高任务分数，并不足以证明元层级改进。

## 研究地图覆盖范围

仓库将文献和资源归到以下主题：

- Agent harness 与执行：记忆、知识管理、技能库、工具构建、规划工作流、失败诊断与恢复。
- 模型训练与适应：数据准备、模仿学习、强化学习、自监督表征、持续适应、知识迁移与训练程序优化。
- 安全与保障：安全探索、安全迁移、运行时控制、监督完整性与风险分类。
- 验证与评测：候选更新验证、评估器可靠性、改进结果评估。
- 基准、数据集与平台：共享机器人数据、任务泛化、真实世界鲁棒性、反馈可靠性与机器人研发基准。

## 读法与限制

- 这是一份持续维护的二级索引；目录所列研究不必然都符合 Physical RSI 的严格定义，也不构成统一协议下的性能比较。
- README 称其与一篇 accompanying paper 对齐，但当前入口没有直接给出可辨识的论文标题或 PDF；此处仅收录可核查的仓库与目录，不另外虚构论文实体。
- 需要判断单项方法是否真正实现递归改进时，应回到原论文检查物理反馈、更新保留与继承、验证器独立性、跨任务评估和资源预算等证据。

## 关联页面

- [Physical RSI 1.0（HKU MMLab × Kinetix AI）](./physical-rsi.md) — 同名但不同项目的具体系统。
- [RoboDojo](./robodojo.md) — 项目地图中的机器人操作评测基准。
- [HarnessPAI](./paper-harnesspai.md) 与 [Harness VLA](./paper-harness-vla.md) — 相关 harness 研究。
- [RSI 四 tier 五 pushes](../queries/rsi-four-tier-five-pushes.md) — RSI 分类与选型框架。
- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 研究地图中的验证与评测、基准与平台主题可按该链的四层评测选型归位

## 同名项目消歧

本页对应 **SAIL Research Lab 的研究地图仓库**。香港大学 MMLab × Kinetix AI 的 [Physical RSI 1.0](./physical-rsi.md) 是另一具体系统与项目页，名称相似但主体、材料和主张均不同；不可将两者的论文、榜单成绩或开源状态混为一谈。

## 参考来源

- [SAIL Physical RSI 项目归档](../../sources/sites/sail-physical-rsi-research-map.md)
- [项目 README](https://github.com/SAIL-Research-Lab/Physical-RSI#readme)
- [参考文献索引](https://github.com/SAIL-Research-Lab/Physical-RSI/blob/main/docs/paper-reference-index.md)
