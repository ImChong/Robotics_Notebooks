---
type: entity
project_id: rollout-then-recycle
tags:
  - paper
  - benchmark
  - evaluation
status: complete
updated: 2026-11-04
arxiv: "2611.03168"
related:
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../overview/embodied-station-2026-11-04-57-papers-technology-map.md
  - ../overview/hub-embodied-eval-benchmark.md
sources:
  - ../../sources/papers/rollout_then_recycle_arxiv_2611_03168.md
  - ../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md
summary: "Rollout then Recycle（arXiv:2611.03168）：汇集同一批任务上多种策略的演示与执行轨迹，保留成功、次优和失败标签；适合奖励模型与失效分析，使用时需留意低成本平台和任务范围。"
---

# Rollout then Recycle（arXiv:2611.03168）

**Rollout then Recycle**（*Rollout then Recycle: 3,718 Labelled Real-Robot Episodes from Benchmarking 7 Policies on 12 Tasks*，[arXiv:2611.03168](https://arxiv.org/abs/2611.03168)，[资源](https://huggingface.co/collections/armnet/armnetbench-v01)）来自 [具身智能小站 2026-11-04 盘点](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)，属「评测、数据与基准」组。

## 一句话定义

**汇集同一批任务上多种策略的演示与执行轨迹，保留成功、次优和失败标签；适合奖励模型与失效分析，使用时需留意低成本平台和任务范围。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Rollout then Recycle | Rollout then Recycle: 3,718 Labelled Real-Robot Episodes from Benchmarking 7 Policies on 12 Tasks | 本文方法 / 基准的简称 |

## 为什么重要

- 把策略评测失败变成可复用数据。
- 开源结论：**项目页未列 GitHub**（步骤 2.5，2026-11-04）。公众号资源入口为 Hugging Face 合集页（HTTP 200），未见代码仓库。
- 同期同组工作见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.03168](https://arxiv.org/abs/2611.03168) |
| **开源** | **项目页未列 GitHub** |
| **要点** | 汇集同一批任务上多种策略的演示与执行轨迹，保留成功、次优和失败标签；适合奖励模型与失效分析，使用时需留意低成本平台和任务范围。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

**不适用（项目页未列 GitHub）** — 公众号资源入口为 Hugging Face 合集页（HTTP 200），未见代码仓库。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号未给具体数字，本页不引用。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**把策略评测失败变成可复用数据：汇集同一批任务上多种策略的演示与执行轨迹，保留成功、次优和失败标签；适合奖励模型与失效分析，使用时需留意低成本平台和任务范围。**

1. 开源边界：**项目页未列 GitHub** — 公众号资源入口为 Hugging Face 合集页（HTTP 200），未见代码仓库。
2. 核心机制：汇集同一批任务上多种策略的演示与执行轨迹，保留成功、次优和失败标签；适合奖励模型与失效分析，使用时需留意低成本平台和任务范围。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [simulation-evaluation-infrastructure](../concepts/simulation-evaluation-infrastructure.md)
- [vla](../methods/vla.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-station-2026-11-04-57-papers-technology-map](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)
- [具身评测基准选型闭环（知识链）](../overview/hub-embodied-eval-benchmark.md)

## 参考来源

- [rollout_then_recycle_arxiv_2611_03168.md](../../sources/papers/rollout_then_recycle_arxiv_2611_03168.md)
- [wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)
- [arXiv:2611.03168](https://arxiv.org/abs/2611.03168)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.03168)
- [https://huggingface.co/collections/armnet/armnetbench-v01](https://huggingface.co/collections/armnet/armnetbench-v01)
