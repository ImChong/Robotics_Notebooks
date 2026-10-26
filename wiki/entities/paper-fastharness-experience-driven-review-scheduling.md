---
type: entity
tags:
  - paper
  - long-horizon
  - manipulation
status: complete
updated: 2026-10-26
arxiv: "2610.31427"
project_id: fastharness-experience-driven-review-scheduling
code: https://github.com/FastHarness/fast-harness
related:
  - ../overview/senlanke-weekly-2026-10-19-23-manipulation-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/fastharness-experience-driven-review-scheduling_arxiv_2610_31427.md
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md
summary: "FastHarness：面向视觉语言动作策略的经验驱动复查调度（arXiv:2610.31427）：FastHarness 将历史执行经验引入复查调度，研究如何根据经验决定何时重新评估 VLA 的执行状态。"
---

# FastHarness：面向视觉语言动作策略的经验驱动复查调度

**FastHarness**（*FastHarness: Experience-Driven Review Scheduling for Vision-Language-Action Policies*，[arXiv:2610.31427](https://arxiv.org/abs/2610.31427)）收录于 senlanke「具身运控lab」10-19–10-23 前沿论文动态（Manipulation）。本页依据**周报条目 + arXiv 摘要页**整理，未逐页核对正文 PDF；指标与结论以论文为准。

## 一句话定义

**FastHarness：面向视觉语言动作策略的经验驱动复查调度——FastHarness 将历史执行经验引入复查调度，研究如何根据经验决定何时重新评估 VLA 的执行状态。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |
| IL | Imitation Learning | 模仿学习 |

## 为什么重要

- 周报所述待解决的问题：长程机器人执行过程中，需要适时检查任务状态。频繁检查会增加计算开销，检查过少又可能导致错误积累。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.31427](https://arxiv.org/abs/2610.31427) · [PDF](https://arxiv.org/pdf/2610.31427) |
| **作者（arXiv）** | Tao Lin, Yuxin Du, Haoxiao Wang, Zile Zhong, Bo Zhao |
| **机构（周报所述，未逐条核实）** | 待核实（作者：Tao Lin、Yuxin Du、Haoxiao Wang、Bo Zhao 等）。 |
| **arXiv 评论栏** | Github:this https URL |
| **开源** | arXiv 摘要页/评论给出 GitHub 链接（https://github.com/FastHarness/fast-harness；仅核对到链接存在，未核验仓库内容）（2026-10-26） |
| **周报所述创新点** | FastHarness 将历史执行经验引入复查调度，研究如何根据经验决定何时重新评估 VLA 的执行状态。 |

### arXiv 摘要节选（英文原文）

> Agentic robot systems combine language-model reasoning with robot policies to monitor execution and recover from failures. However, reviewing every action chunk incurs substantial overhead, even in familiar situations.

## 源码运行时序图

**不适用**：arXiv 摘要页/评论给出 GitHub 链接（https://github.com/FastHarness/fast-harness；仅核对到链接存在，未核验仓库内容），无可对齐的官方代码流程。

## 实验与评测

- 周报与摘要给出的数字/场景已写入上表「创新点」与摘要节选；其余实验细节、基线与成功率定义请回到论文正文。
- 读法：先对齐任务设定、传感器与成功定义（仿真/真机），再解读 headline 数字，勿跨论文直接比较。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见 [Manipulation周报技术地图](../overview/senlanke-weekly-2026-10-19-23-manipulation-technology-map.md)，按方向浏览同期论文 |
| **开源状态** | arXiv 摘要页/评论给出 GitHub 链接（https://github.com/FastHarness/fast-harness；仅核对到链接存在，未核验仓库内容） |

## 结论

**FastHarness：面向视觉语言动作策略的经验驱动复查调度。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-19-23-manipulation-technology-map](../overview/senlanke-weekly-2026-10-19-23-manipulation-technology-map.md)
- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)

## 参考来源

- [fastharness-experience-driven-review-scheduling_arxiv_2610_31427.md](../../sources/papers/fastharness-experience-driven-review-scheduling_arxiv_2610_31427.md)
- [wechat_senlanke_weekly_manipulation_2026-10-19_23.md](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md)
- [arXiv:2610.31427](https://arxiv.org/abs/2610.31427)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.31427)
