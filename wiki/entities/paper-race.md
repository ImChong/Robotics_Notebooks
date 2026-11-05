---
type: entity
project_id: race
tags:
  - paper
  - benchmark
  - evaluation
status: complete
updated: 2026-11-05
arxiv: "2611.05838"
project: https://changyeon.site/race/
related:
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../overview/embodied-station-2026-11-05-43-papers-technology-map.md
sources:
  - ../../sources/papers/race_arxiv_2611_05838.md
  - ../../sources/sites/race.md
  - ../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md
summary: "RACE（arXiv:2611.05838）：夹爪靠近物体时，几毫米偏差可能决定成败。"
---

# RACE（arXiv:2611.05838）

**RACE**（*RACE: Calibrated Cumulative Scoring for Rollout-Free Robot Policy Checkpoint Selection*，[arXiv:2611.05838](https://arxiv.org/abs/2611.05838)，[项目页](https://changyeon.site/race/)，[资源](https://changyeon.site/race/data.json)）来自 [具身智能小站 2026-11-05 盘点](../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)，属「策略泛化与评测」组。

## 一句话定义

**夹爪靠近物体时，几毫米偏差可能决定成败。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RACE | RACE: Calibrated Cumulative Scoring for Rollout-Free Robot Policy Checkpoint Selection | 本文方法 / 基准的简称 |

## 为什么重要

- 离线挑策略，先看它从哪一步开始偏离。RACE 用成功示范估计各状态可容忍的动作偏差，一旦偏得太远，就调低后续步骤的权重，据此给训练检查点排序。
- 开源结论：**待发布**（步骤 2.5，2026-11-05）。项目页 HTTP 200，标注 Code (coming soon)；公众号资源入口为 data.json（HTTP 200）。
- 同期同组工作见 [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.05838](https://arxiv.org/abs/2611.05838) |
| **开源** | **待发布** |
| **要点** | 夹爪靠近物体时，几毫米偏差可能决定成败。 |
| **文内指标** | 公众号转述：RACE 用成功示范估计各状态可容忍的动作偏差，一旦偏得太远，就调低后续步骤的权重，据此给训练检查点排序。 |

## 源码运行时序图

**不适用（待发布）** — 项目页 HTTP 200，标注 Code (coming soon)；公众号资源入口为 data.json（HTTP 200）。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号转述：RACE 用成功示范估计各状态可容忍的动作偏差，一旦偏得太远，就调低后续步骤的权重，据此给训练检查点排序。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**离线挑策略，先看它从哪一步开始偏离：RACE 用成功示范估计各状态可容忍的动作偏差，一旦偏得太远，就调低后续步骤的权重，据此给训练检查点排序。**

1. 开源边界：**待发布** — 项目页 HTTP 200，标注 Code (coming soon)；公众号资源入口为 data.json（HTTP 200）。
2. 核心机制：夹爪靠近物体时，几毫米偏差可能决定成败。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [simulation-evaluation-infrastructure](../concepts/simulation-evaluation-infrastructure.md)
- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [embodied-station-2026-11-05-43-papers-technology-map](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)

## 参考来源

- [race_arxiv_2611_05838.md](../../sources/papers/race_arxiv_2611_05838.md)
- [wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md](../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)
- [arXiv:2611.05838](https://arxiv.org/abs/2611.05838)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.05838)
- [项目页](https://changyeon.site/race/)
- [https://changyeon.site/race/data.json](https://changyeon.site/race/data.json)
