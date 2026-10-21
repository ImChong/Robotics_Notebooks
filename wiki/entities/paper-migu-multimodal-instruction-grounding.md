---
type: entity
project_id: migu-multimodal-instruction-grounding
tags:
  - paper
  - instruction-grounding
  - pointing-gesture
  - uncertainty
  - mobile-manipulation
status: complete
updated: 2026-10-18
arxiv: "2609.24995"
project: https://multimodal-instruction.github.io/
related:
  - ../tasks/manipulation.md
  - ../tasks/loco-manipulation.md
  - ../methods/vla.md
  - ../overview/embodied-station-2026-10-18-26-papers-technology-map.md
sources:
  - ../../sources/papers/migu_multimodal_instruction_grounding_arxiv_2609_24995.md
  - ../../sources/sites/migu-multimodal-instruction-grounding.md
  - ../../sources/blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md
summary: "MIGU（arXiv:2609.24995）：融合语言语义与指向手势的三维几何不确定性，先形成目标信念，再决定执行或澄清。"
---

# MIGU（arXiv:2609.24995）

**MIGU**（*MIGU: Multimodal Instruction Grounding under Uncertainty for Manipulation Planning*，[arXiv:2609.24995](https://arxiv.org/abs/2609.24995)，[项目页](https://multimodal-instruction.github.io/)）来自 [具身智能小站 2026-10-18 盘点](../../sources/blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)，属「策略与规划」组。

## 一句话定义

**融合语言语义与指向手势的三维几何不确定性，先形成目标信念，再决定执行或澄清。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MIGU | Multimodal Instruction Grounding under Uncertainty | 不确定性下的多模态指令落地 |

## 为什么重要

- 语言加手势也需要知道何时追问；不确定性显式建模连接移动操作与桌面规划。
- 开源结论：**待发布**（步骤 2.5，2026-10-18）。项目页 HTTP 200，含 coming soon 字样，未列 GitHub。
- 同期同组工作见 [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../overview/embodied-station-2026-10-18-26-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.24995](https://arxiv.org/abs/2609.24995) |
| **开源** | **待发布** |
| **要点** | 语言语义 + 手势三维几何不确定性 → 目标信念 → 执行/澄清决策。 |
| **文内指标** | 公众号提示：当前决策只允许单轮追问；未给具体数字。 |

## 源码运行时序图

**不适用（待发布）** — 项目页 HTTP 200，含 coming soon 字样，未列 GitHub。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号提示：当前决策只允许单轮追问；未给具体数字。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../overview/embodied-station-2026-10-18-26-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**显式的目标信念让“执行还是追问”成为可评估的决策，当前仅支持单轮澄清。**

1. 开源边界：**待发布** — 项目页 HTTP 200，含 coming soon 字样，未列 GitHub。
2. 核心机制：语言语义 + 手势三维几何不确定性 → 目标信念 → 执行/澄清决策。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [loco-manipulation](../tasks/loco-manipulation.md)
- [vla](../methods/vla.md)
- [embodied-station-2026-10-18-26-papers-technology-map](../overview/embodied-station-2026-10-18-26-papers-technology-map.md)

## 参考来源

- [migu_multimodal_instruction_grounding_arxiv_2609_24995.md](../../sources/papers/migu_multimodal_instruction_grounding_arxiv_2609_24995.md)
- [wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md](../../sources/blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)
- [arXiv:2609.24995](https://arxiv.org/abs/2609.24995)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.24995)
- [项目页](https://multimodal-instruction.github.io/)
