---
type: entity
project_id: inertia-mismatch
tags:
  - paper
  - benchmark
  - evaluation
status: complete
updated: 2026-11-04
arxiv: "2611.03501"
project: https://younghyopark.me/inertia-mismatch/
related:
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../overview/embodied-station-2026-11-04-57-papers-technology-map.md
sources:
  - ../../sources/papers/inertia_mismatch_arxiv_2611_03501.md
  - ../../sources/sites/inertia-mismatch.md
  - ../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md
summary: "When Robots Outweigh Objects（arXiv:2611.03501）：通过受控仿真改变机器人、物体质量与传动反射惯量，发现接触惯量失配会影响行为克隆；也讨论在既有硬件上选择更低惯量的示教姿态。"
---

# When Robots Outweigh Objects（arXiv:2611.03501）

**When Robots Outweigh Objects**（*When Robots Outweigh Objects: How Inertia Mismatch Shapes Robot Policy Learning*，[arXiv:2611.03501](https://arxiv.org/abs/2611.03501)，[项目页](https://younghyopark.me/inertia-mismatch/)）来自 [具身智能小站 2026-11-04 盘点](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)，属「评测、数据与基准」组。

## 一句话定义

**通过受控仿真改变机器人、物体质量与传动反射惯量，发现接触惯量失配会影响行为克隆；也讨论在既有硬件上选择更低惯量的示教姿态。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| When Robots Outweigh Objects | When Robots Outweigh Objects: How Inertia Mismatch Shapes Robot Policy Learning | 本文方法 / 基准的简称 |

## 为什么重要

- 机器人太重，会增加模仿学习难度吗。
- 开源结论：**项目页未列 GitHub**（步骤 2.5，2026-11-04）。项目页 HTTP 200，仅见 robosuite、MuJoCo 等依赖链接，未见论文代码。
- 同期同组工作见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.03501](https://arxiv.org/abs/2611.03501) |
| **开源** | **项目页未列 GitHub** |
| **要点** | 通过受控仿真改变机器人、物体质量与传动反射惯量，发现接触惯量失配会影响行为克隆；也讨论在既有硬件上选择更低惯量的示教姿态。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

**不适用（项目页未列 GitHub）** — 项目页 HTTP 200，仅见 robosuite、MuJoCo 等依赖链接，未见论文代码。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号未给具体数字，本页不引用。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**机器人太重，会增加模仿学习难度吗：通过受控仿真改变机器人、物体质量与传动反射惯量，发现接触惯量失配会影响行为克隆；也讨论在既有硬件上选择更低惯量的示教姿态。**

1. 开源边界：**项目页未列 GitHub** — 项目页 HTTP 200，仅见 robosuite、MuJoCo 等依赖链接，未见论文代码。
2. 核心机制：通过受控仿真改变机器人、物体质量与传动反射惯量，发现接触惯量失配会影响行为克隆；也讨论在既有硬件上选择更低惯量的示教姿态。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [simulation-evaluation-infrastructure](../concepts/simulation-evaluation-infrastructure.md)
- [vla](../methods/vla.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-station-2026-11-04-57-papers-technology-map](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)

## 参考来源

- [inertia_mismatch_arxiv_2611_03501.md](../../sources/papers/inertia_mismatch_arxiv_2611_03501.md)
- [wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)
- [arXiv:2611.03501](https://arxiv.org/abs/2611.03501)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.03501)
- [项目页](https://younghyopark.me/inertia-mismatch/)
