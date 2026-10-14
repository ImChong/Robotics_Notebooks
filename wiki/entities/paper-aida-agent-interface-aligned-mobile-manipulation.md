---
type: entity
project_id: aida-agent-interface-aligned-mobile-manipulation
tags:
  - paper
  - mobile-manipulation
  - active-vision
  - demonstration
  - sim2real
status: complete
updated: 2026-10-14
arxiv: "2610.15572"
project: https://connerqiu.github.io/look-before-you-move/
related:
  - ../tasks/loco-manipulation.md
  - ../methods/imitation-learning.md
  - ../tasks/zero-shot-object-navigation.md
  - ../overview/embodied-station-2026-10-14-41-papers-technology-map.md
sources:
  - ../../sources/papers/aida_agent_interface_aligned_mobile_manipulation_arxiv_2610_15572.md
  - ../../sources/sites/aida-agent-interface-aligned-mobile-manipulation.md
  - ../../sources/blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md
summary: "AIDA（arXiv:2610.15572）：让教师通过部署时相同的技能接口采集示范，显式覆盖「保持目标可见」与「丢失后恢复」，仿真训练后真机迁移。"
---

# AIDA（arXiv:2610.15572）

**AIDA**（*Look Before You Move: Agent-Interface-Aligned Supervision for Open-World Mobile Manipulation*，[arXiv:2610.15572](https://arxiv.org/abs/2610.15572)，[项目页](https://connerqiu.github.io/look-before-you-move/)）来自 [具身智能小站 2026-10-14 盘点](../../sources/blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)，属「人形与移动操作」组。

## 一句话定义

**让教师通过部署时相同的技能接口采集示范，显式覆盖「保持目标可见」与「丢失后恢复」，仿真训练后真机迁移。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AIDA | Agent-Interface-Aligned Demonstration/supervision | 接口对齐监督（论文方法） |
| MM | Mobile Manipulation | 移动操作 |
| IL | Imitation Learning | 模仿学习 |

## 为什么重要

- 移动操作数据质量不只取决于轨迹，还取决于示范是否包含主动看的行为。
- 开源结论：**项目页未列 GitHub**（步骤 2.5，2026-10-14）。项目页入库日未列 GitHub。
- 同期同组工作见 [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../overview/embodied-station-2026-10-14-41-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.15572](https://arxiv.org/abs/2610.15572) |
| **开源** | **项目页未列 GitHub** |
| **要点** | 接口对齐示范 + 主动视觉监督。 |
| **文内指标** | 仿真训练、真机迁移；以原文为准。 |

## 源码运行时序图

**不适用（项目页未列 GitHub）** — 项目页入库日未列 GitHub。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 仿真训练、真机迁移；以原文为准。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../overview/embodied-station-2026-10-14-41-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**采示范前先对齐接口：教师与策略用同一套技能接口，数据才可直接复用。**

1. 开源边界：**项目页未列 GitHub** — 项目页入库日未列 GitHub。
2. 核心机制：接口对齐示范 + 主动视觉监督。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [loco-manipulation](../tasks/loco-manipulation.md)
- [imitation-learning](../methods/imitation-learning.md)
- [zero-shot-object-navigation](../tasks/zero-shot-object-navigation.md)
- [embodied-station-2026-10-14-41-papers-technology-map](../overview/embodied-station-2026-10-14-41-papers-technology-map.md)

## 参考来源

- [aida_agent_interface_aligned_mobile_manipulation_arxiv_2610_15572.md](../../sources/papers/aida_agent_interface_aligned_mobile_manipulation_arxiv_2610_15572.md)
- [wechat_embodied_station_41_papers_streamingwam_2026-10-14.md](../../sources/blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)
- [arXiv:2610.15572](https://arxiv.org/abs/2610.15572)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.15572)
- [项目页](https://connerqiu.github.io/look-before-you-move/)
