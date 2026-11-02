---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2605.18556"
summary: "In this paper, we introduce Key-Gram, a conditional-memory framework that separates language-derived world knowledge from visual-state reasoning for embodied control."
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2605_18556_key-gram-extensible-world-knowledge-for.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# Key-Gram

**Key-Gram: Extensible World Knowledge for Embodied Manipulation** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 242/564** 篇，分组 **VLA**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

In this paper, we introduce Key-Gram, a conditional-memory framework that separates language-derived world knowledge from visual-state reasoning for embodied control.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- In this paper, we introduce Key-Gram, a conditional-memory framework that separates language-derived world knowledge from visual-state reasoning for embodied control.
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TB
    INS["语言指令 I"] --> PARSE["轻量 LM / API 解析器抽取固定数量的短 key-gram<br/>(例: put mug in microwave)"]
    PARSE --> HASH["补零定长 → 乘法-XOR 哈希, 映射到 H 个 hash head<br/>确定性 O(1) 查表, 取静态记忆嵌入并拼接"]
    TABLE["外部记忆: 由 slot x head 物理子表组成的逻辑表<br/>可加 slot / 扩容子表, 可放主机内存"] -.-> HASH
    HASH --> FUSE
    V0["初始视觉观测 V0"] --> BB["视觉骨干 (π0 / π0.5)"]
    subgraph FUSE["Context-Adaptive Key-Gram Fusion (选定 Transformer 层, 注意力前残差)"]
        direction TB
        KV["记忆 → 投影为 K_m, V_m"] --> GATE["视觉 token 对 K_m 计算逐 token 门控 A"]
        GATE --> CONV["A ⊙ V_m → 长跨度卷积 → 残差加回隐状态"]
    end
    BB --> FUSE
    FUSE --> BB
    BB -->|"最终层隐状态"| VQ["可学习查询 + cross-attention + 两级上采样 → 未来视觉 latent (VAE 空间)"]
    BB -->|"KV cache"| AE["较小的动作专家: 流匹配去噪 → 动作轨迹"]
```

图依据：论文 §3.1–3.5 / Fig.1、Fig.2

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 242/564 |
| 分组 | VLA |
| 出处 | 见清单 / 原文 |
| 论文 | <https://arxiv.org/abs/2605.18556> |
| 子类 / 象限 | VLA后训练与数据增强 · 分层与双系统VLA · 不适用 |

## 核心机制（归纳）

### 策展导读要点

In this paper, we introduce Key-Gram, a conditional-memory framework that separates language-derived world knowledge from visual-state reasoning for embodied control.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2605.18556) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **VLA** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Key-Gram」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **VLA**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/rcl_awesome_wam_2605_18556_key-gram-extensible-world-knowledge-for.md`](../../sources/papers/rcl_awesome_wam_2605_18556_key-gram-extensible-world-knowledge-for.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2605.18556>
- [arXiv PDF](https://arxiv.org/pdf/2605.18556)：已通读方法部分（实验与附录以略读为准）：§3 方法（3.1 概览、3.2 稀疏 key-gram 检索、3.3 上下文自适应融合、3.4 视觉与动作预测、3.5 可扩展记忆分配）；§4 实验以略读为准

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2605.18556)
