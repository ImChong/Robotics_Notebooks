---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2506.04363"
venue: "arXiv 2025"
summary: "A video‑based benchmark for evaluating AI models’ world modeling and procedural planning abilities."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2506_04363_worldprediction-a-benchmark-for-high-lev.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# WorldPrediction

**WorldPrediction: A Benchmark for High-level World Modeling and Long-horizon Procedural Planning** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 051/571** 篇，分组 **115 Benchmarks for World Model Simulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

A video‑based benchmark for evaluating AI models’ world modeling and procedural planning abilities.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- A video‑based benchmark for evaluating AI models’ world modeling and procedural planning abilities.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    FORM["理论形式化 §3.1：⟨S, A, T, O⟩（POSMDP 启发）：隐状态 s、高层动作 a、隐转移模型 T、观测模型 O 把 s 与 a 映为图像 O(s) 与视频段 O(a)"]
    FORM --> TASK["任务 §3.2–§3.3：给定初末状态图像，从含 3 个干扰项的 4 个候选中选择正确动作或动作序列"]
    TASK --> WM["WorldPrediction-WM：选出哪个动作片段导致 O(st) 到 O(st+1)"]
    TASK --> PP["WorldPrediction-PP：选出哪个动作序列（T 步）排序正确，连接 O(s_init) 与 O(s_final)"]
    SRC["数据来源 §3.4：COIN / CrossTask / EgoExo4D / EPIC-KITCHENS-100 / IKEA-ASM（官方划分的 test 或 validation）"] --> CAND["采样候选：WM 干扰项取同一视频中不相容的动作；PP 干扰项为打乱后的真值序列"]
    CAND --> EQV["Action Equivalents：把真值动作换成不同背景或视角下的同类动作，避免背景连续性捷径"]
    EQV --> FILT["Observability Filtering：用 DINOv2 特征距离阈值剔除剧烈变化；EgoExo4D 另用 VLM 提示剔除背对镜头与遮挡；剔除状态差异过小的样本"]
    FILT --> HUM["人工验证 §4.3：初始各 1500 个样本，两名标注者独立作答，仅保留两人都答对者"]
    HUM --> BENCH["WorldPrediction-WM 825 条 / WorldPrediction-PP 570 条"]
    WM --> BENCH
    PP --> BENCH
    BENCH --> EVAL["被测对象 §4：VLM（视频或图像 token 直接输入）/ Socratic LLM（先 change captioning 与 action captioning 再文本推理）/ 视频扩散模型（按 caption 生成末帧，与目标状态比相似度）/ OEPP 规划器"]
```

图依据：基准构建与评测协议图（任务形式化 → 数据与过滤 → 人工验证 → 评测对象）：论文 §3.1–§3.4、Fig. 1–5、§4.3、Fig. 4。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 051/571 |
| 分组 | 115 Benchmarks for World Model Simulation |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2506.04363> |

## 核心机制（归纳）

### 策展导读要点

A video‑based benchmark for evaluating AI models’ world modeling and procedural planning abilities.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.04363) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **115 Benchmarks for World Model Simulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「WorldPrediction」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **115 Benchmarks for World Model Simulation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome World Models](../entities/awesome-world-models.md)
- 技术地图：[Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_wm_2506_04363_worldprediction-a-benchmark-for-high-lev.md`](../../sources/papers/sun_awesome_wm_2506_04363_worldprediction-a-benchmark-for-high-lev.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2506.04363>
- [arXiv PDF](https://arxiv.org/pdf/2506.04363)：已通读方法部分（实验与附录以略读为准）：摘要、§1–§3（含 Fig. 1–5、Table 1）、§4.3 与附录 A 的人工评测设置、Fig. 4 基线示意；§4 各基线的详细结果表以略读为准

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2506.04363)
