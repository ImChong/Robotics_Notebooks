---
type: entity
tags: ['paper', 'vla', 'tamp', 'demonstration', 'manipulation']
status: complete
updated: 2026-09-24
arxiv: "2609.28314"
related:
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../concepts/foundation-policy.md
  - ../overview/embodied-13-papers-technology-map.md
sources:
  - ../../sources/papers/tandem_arxiv_2609_28314.md
  - ../../sources/blogs/wechat_embodied_13_papers_forgetmimic_2026-09-24.md
summary: "TANDEM（arXiv:2609.28314）：规划器能走的步骤交给 TAMP，只在能力缺口处按需遥操作，并把各阶段拼成完整 VLA 微调示范。"
---

# TANDEM（arXiv:2609.28314）

**TANDEM: Task and Motion Planning with As-Needed Demonstrations for Efficient Vision-Language-Action Model Fine-tuning**（[项目页](https://prpl-group.com/tandem/)，[arXiv:2609.28314](https://arxiv.org/abs/2609.28314)）来自 [具身智能小站 · 13 篇盘点](../../sources/blogs/wechat_embodied_13_papers_forgetmimic_2026-09-24.md)（2026-09-24）。

## 一句话定义

**规划器能走的步骤交给 TAMP，只在能力缺口处按需遥操作，并把各阶段拼成完整 VLA 微调示范。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| RL | Reinforcement Learning | 强化学习 |
| TO | Trajectory Optimization | 轨迹优化 |
| HITL | Human-in-the-Loop | 人在回路 |

## 为什么重要

- 长时操作数据采集瓶颈常是「人做了机器人本就能做的事」；TANDEM 把人类协助建模为按需 magic operator。
- 纳入 [13 篇技术地图](../overview/embodied-13-papers-technology-map.md) 阅读坐标。
- 开源结论（步骤 2.5，2026-09-24）：**待发布**。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.28314](https://arxiv.org/abs/2609.28314) |
| **开源** | **待发布** |
| **要点** | VLM 扩展 TAMP 域：发明缺失 predicate + 人类执行的 magic operator；每段人工后重感知并验证 effect 再续规划；DATAFARM 对齐 TAMP 段与 VLA 预训练分布。 |
| **文内指标** | 代表任务同等人工时间下示范量约为全程遥操作 **2.9×**；五任务各 **20** 条示范微调 π0.5-DROID，平均成功率 **0%→60%**。 |


## 源码运行时序图

**不适用**（截至 2026-09-24 项目页/论文未提供可运行官方代码；开源状态：**待发布**）。


## 实验与评测

- 代表任务同等人工时间下示范量约为全程遥操作 **2.9×**；五任务各 **20** 条示范微调 π0.5-DROID，平均成功率 **0%→60%**。
- **读法：** 公众号归纳；逐项 baseline 与协议以 arXiv PDF 为准。

## 与其他工作对比

- 横向索引见 [13 篇技术地图](../overview/embodied-13-papers-technology-map.md)。

## 结论

**示范预算应投在规划器覆盖不了的阶段** — 推理期无 planner，需单独验证 15 Hz 端到端 rollout。

1. 开源边界：**待发布** — 以项目页/仓库实际链接为准（入库日 2026-09-24）。
2. 核心机制：VLM 扩展 TAMP 域：发明缺失 predicate + 人类执行的 magic operator；每段人工后重感知并验证 effect 再续规划；DATAFARM 对齐 TAMP 段与 VLA 预训练分布。…
3. 部署前核对任务协议与硬件条件，勿直接横比公众号摘录数字。

## 关联页面

- [Vla](../methods/vla.md)
- [Manipulation](../tasks/manipulation.md)
- [Foundation Policy](../concepts/foundation-policy.md)
- [Embodied 13 Papers Technology Map](../overview/embodied-13-papers-technology-map.md)

## 参考来源

- [13 篇盘点（公众号）](../../sources/blogs/wechat_embodied_13_papers_forgetmimic_2026-09-24.md)
- [TANDEM: Task and Motion Planning with As-Needed Demonstrations for Efficient Vision-Language-Action Model Fine-tuning](../../sources/papers/tandem_arxiv_2609_28314.md)

## 推荐继续阅读

- [arXiv:2609.28314](https://arxiv.org/abs/2609.28314) — 原文
