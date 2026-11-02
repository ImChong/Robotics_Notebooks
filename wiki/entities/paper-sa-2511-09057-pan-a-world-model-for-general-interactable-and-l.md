---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2511.09057"
venue: "arXiv 2025"
summary: "General, interactable, and long-horizon world simulation model."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2511_09057_pan-a-world-model-for-general-interactab.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# PAN

**PAN: A World Model for General, Interactable, and Long-Horizon World Simulation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 149/571** 篇，分组 **47 General Foundation World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

General, interactable, and long-horizon world simulation model.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- General, interactable, and long-horizon world simulation model.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    OBS["观测 o_t（视频）"] --> ENC["Vision Encoder h<br/>Qwen2.5-VL-7B 视觉塔 → 潜表示 ŝ_t"]
    ENC --> BB
    ACT["自然语言动作 a_t<br/>（视频块级）"] --> BB
    HIST["历史 (ŝ_1, a_1, …, ŝ_t, a_t)<br/>多轮对话格式，窗口内最近 10 轮"] --> BB
    BB["自回归世界模型骨干 f<br/>Qwen2.5-VL-7B 语言模型 + 256 个可学习 query<br/>→ 下一潜世界状态 ŝ_t+1（256 个连续 token）"]
    BB --> DEC
    ACT --> T5["umT5 编码动作文本"] --> DEC
    PREV["上一块的部分去噪上下文 ô_t"] --> DEC
    DEC["视频扩散解码器 g<br/>Wan2.1-T2V-14B 改造，Causal Swin-DPM<br/>滑窗内两块不同噪声级，块级因果注意力<br/>潜状态经新增 cross-attention 流（零初始化）注入"]
    DEC --> NEXT["预测下一段视频 ô_t+1"]
    NEXT -.->|"自回归 rollout：作为下一步观测上下文"| ENC
    subgraph TR["训练（两阶段）"]
        S1["阶段 1 模块级：冻结 VLM（视觉编码器+骨干），<br/>把解码器适配为 Causal Swin-DPM"] --> S2["阶段 2 联合训练：GLP 生成式目标 (flow matching)<br/>VLM 冻结，只训 query 嵌入与解码器"]
    end
```

图依据：论文 §2.1–2.3 / Fig. 1 与补充材料 §1（模型架构）、§2（训练）；rollout 回灌箭头依据式 (4)–(6)。树状规划（Fig. 3）与 WR-Arena 评测未在图中展开。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 149/571 |
| 分组 | 47 General Foundation World Models |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2511.09057> |
| 项目页 | <https://panworld.ai/> |

## 核心机制（归纳）

### 策展导读要点

General, interactable, and long-horizon world simulation model.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2511.09057) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **47 General Foundation World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「PAN」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **47 General Foundation World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2511_09057_pan-a-world-model-for-general-interactab.md`](../../sources/papers/sun_awesome_wm_2511_09057_pan-a-world-model-for-general-interactab.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2511.09057>
- [arXiv PDF](https://arxiv.org/pdf/2511.09057)：已通读方法部分（实验与附录以略读为准）：§2 GLP 框架与设计选择、§5.3 训练、补充材料 §1 模型架构与 §2 训练（评测与其余附录略读）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2511.09057)
