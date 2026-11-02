---
type: entity
project_id: cnp-flow-conditional-source-distribution
tags:
  - paper
  - flow-matching
  - source-distribution
  - video-generation
  - planning
  - ntu
status: complete
updated: 2026-10-14
arxiv: "2610.05349"
project: https://embodiedai-ntu.github.io/cnpflow/index.html
related:
  - ../concepts/flow-matching-embodied-policy.md
  - ./paper-delta-flow-matching-robot-foundation-models.md
  - ../overview/embodied-station-2026-10-13-34-papers-technology-map.md
sources:
  - ../../sources/papers/cnp_flow_conditional_source_distribution_arxiv_2610_05349.md
  - ../../sources/sites/cnp-flow-conditional-source-distribution.md
  - ../../sources/blogs/wechat_embodied_station_34_papers_mulligan_2026-10-13.md
summary: "CNP-Flow（arXiv:2610.05349）：通过流反演监督条件噪声预测器再微调原流模型；视频生成与七自由度机械臂规划表明改进噪声起点可降采样开销。"
---

# CNP-Flow（arXiv:2610.05349）

**CNP-Flow**（*Learning Conditional Source Distribution via Flow Reversal for Temporal Flow Matching*，[arXiv:2610.05349](https://arxiv.org/abs/2610.05349)，[项目页](https://embodiedai-ntu.github.io/cnpflow/index.html)）来自 [具身智能小站 2026-10-13 盘点](../../sources/blogs/wechat_embodied_station_34_papers_mulligan_2026-10-13.md)，属「VLA 与动作生成」组。

## 一句话定义

**通过流反演监督条件噪声预测器再微调原流模型；视频生成与七自由度机械臂规划表明改进噪声起点可降采样开销。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CNP | Conditional Noise Predictor | 条件噪声预测器 |
| FM | Flow Matching | 流匹配 |
| 7-DoF | 7 Degrees of Freedom | 七自由度机械臂 |

## 为什么重要

- 改噪声起点是降低采样成本的另一杠杆。
- 开源结论：**项目页未列 GitHub**（步骤 2.5，2026-10-14）。项目页入库日未列 GitHub。
- 同期同组工作见 [失败导向采集与可组合世界动作模型：具身智能小站 10-13 期 34 篇阅读坐标](../overview/embodied-station-2026-10-13-34-papers-technology-map.md)。

## 流程总览

```mermaid
flowchart TD
  A["阶段 1：用标准高斯 N(0,I) 作源分布<br/>预训练条件 FM，得到 v_θ(x_t, t | C_T)"] --> B["对目标样本 x1 做流反演（Eq.4）<br/>得到源样本 x̃0，按时间条件聚簇"]
  B --> C["阶段 2：训练条件噪声预测器 CNP<br/>输出各向同性高斯 μ_φ(C_T), σ_φ(C_T)<br/>损失 = NLL + λ·KL(对 N(0,I))"]
  C --> D["阶段 3：从 CNP 预测的源分布采样 x0<br/>替换高斯后微调 FM（Eq.7）"]
  D --> E["推理：x0 ~ p_φ(·|C_T)<br/>FM 对向量场积分得到目标样本"]
  E --> F["应用：视频预测 / 视频插值 / 运动规划<br/>区别仅在时间条件 C_T 的构造"]
```

图依据：论文 §4.1 / Fig.3（三阶段训练流程）、§4.2、Algo.1–4

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.05349](https://arxiv.org/abs/2610.05349) |
| **开源** | **项目页未列 GitHub** |
| **要点** | 与 Δ-Flow 同属「改起点」路线。 |
| **文内指标** | 以原文为准。 |

## 源码运行时序图

**不适用（项目页未列 GitHub）** — 项目页入库日未列 GitHub。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 以原文为准。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [失败导向采集与可组合世界动作模型：具身智能小站 10-13 期 34 篇阅读坐标](../overview/embodied-station-2026-10-13-34-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**条件起点分布可学，且与 Δ-Flow 的历史起点互补。**

1. 开源边界：**项目页未列 GitHub** — 项目页入库日未列 GitHub。
2. 核心机制：与 Δ-Flow 同属「改起点」路线。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [flow-matching-embodied-policy](../concepts/flow-matching-embodied-policy.md)
- [paper-delta-flow-matching-robot-foundation-models](./paper-delta-flow-matching-robot-foundation-models.md)
- [embodied-station-2026-10-13-34-papers-technology-map](../overview/embodied-station-2026-10-13-34-papers-technology-map.md)

## 参考来源

- [cnp_flow_conditional_source_distribution_arxiv_2610_05349.md](../../sources/papers/cnp_flow_conditional_source_distribution_arxiv_2610_05349.md)
- [wechat_embodied_station_34_papers_mulligan_2026-10-13.md](../../sources/blogs/wechat_embodied_station_34_papers_mulligan_2026-10-13.md)
- [arXiv:2610.05349](https://arxiv.org/abs/2610.05349)
- [arXiv PDF](https://arxiv.org/pdf/2610.05349)：已通读方法部分（实验与附录以略读为准）：§3 预备、§4 方法（4.1、4.2）、§5.1 实验设定、附录 A 算法

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.05349)
- [项目页](https://embodiedai-ntu.github.io/cnpflow/index.html)
