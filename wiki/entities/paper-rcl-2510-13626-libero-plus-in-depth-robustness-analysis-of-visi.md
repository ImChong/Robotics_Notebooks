---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2510.13626"
venue: "2025"
summary: "RCL Awesome WAM 清单收录（Benchmarks & simulators）；细节以原文 PDF / 项目页为准。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2510_13626_libero-plus-in-depth-robustness-analysis.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# LIBERO-Plus

**LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 030/564** 篇，分组 **Benchmarks & simulators**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

RCL Awesome WAM 清单收录（Benchmarks & simulators）；细节以原文 PDF / 项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- RCL Awesome WAM 清单收录（Benchmarks & simulators）；细节以原文 PDF / 项目页为准。
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    L["LIBERO 的 40 个评测任务（Spatial / Object / Goal / Long）"] --> P["沿 7 个扰动维度自动生成：每个子任务 500 个实例，共 14,000 候选任务"]
    P --> E["用若干主流 VLA 基线评测成功率分布"]
    E --> F["过滤：移除被全部或大多数模型解出的任务（避免天花板效应）"]
    F --> B["按增强子维度均衡任务"]
    B --> BM["LIBERO-Plus：10,030 个纯测试任务，7 个扰动因子 / 21 个底层组件"]
    E --> LV["按 4 个代表模型的准确率分布划分 Level-1 至 Level-5 难度"]
    LV --> BM
    P -.-> TD["同一自动化流水线调整参数生成训练集：回放 LIBERO 的 (state, action) 并只保留成功轨迹"]
    TD --> FT["混合微调 OpenVLA-OFT，在 LIBERO-Plus 上评测"]
    FT --> BM
```

图依据：论文 §6.1 Benchmark Construction（Fig. 6）、§6.2 与附录 C.1 Data Generation and Filtering、D.2–D.3 训练集生成。本页是基准/分析类论文，图画的是基准构建与训练集生成流程，而非新策略架构；难度分级的四个模型以正文 Fig. 5 为准。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 030/564 |
| 分组 | Benchmarks & simulators |
| 出处 | 2025 |
| 论文 | <https://arxiv.org/abs/2510.13626> |
| 子类 / 象限 | 机器人操作基准 · 鲁棒性与泛化评测 · 不适用 |

## 核心机制（归纳）

### 策展导读要点

RCL Awesome WAM 清单收录（Benchmarks & simulators）；细节以原文 PDF / 项目页为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2510.13626) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **Benchmarks & simulators** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「LIBERO-Plus」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Benchmarks & simulators**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Contribution 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/rcl_awesome_wam_2510_13626_libero-plus-in-depth-robustness-analysis.md`](../../sources/papers/rcl_awesome_wam_2510_13626_libero-plus-in-depth-robustness-analysis.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2510.13626>
- [arXiv PDF](https://arxiv.org/pdf/2510.13626)：已通读方法部分（实验与附录以略读为准）：§6 LIBERO-Plus（6.1 基准构建、6.2 泛化训练集）、附录 C（扰动与基准构建）、附录 D（训练集构建）

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2510.13626)
