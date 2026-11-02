---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2606.02436"
venue: "arXiv 2026"
summary: "Proposes a geometry-aware implicit memory framework that compresses variable-length histories into fixed-size memory tokens via a lightweight transformer."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2606_02436_geometry-aware-implicit-memory-for-video.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Geometry-Aware Implicit Memory for Video World Models

**Geometry-Aware Implicit Memory for Video World Models** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 078/571** 篇，分组 **42 Visual / Video World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Proposes a geometry-aware implicit memory framework that compresses variable-length histories into fixed-size memory tokens via a lightweight transformer.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Proposes a geometry-aware implicit memory framework that compresses variable-length histories into fixed-size memory tokens via a lightweight transformer.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    HIST["历史帧 latent<br/>VAE + patch embedding"] --> PRUNE["信息引导剪枝 §3.4<br/>互信息 + 高斯过程（位姿-时间核）<br/>贪心保留至多 K 帧"]
    PRUNE --> CAMADD["历史 token 加相机嵌入 Ec(ci)"]
    Q0["Nm 个可学习记忆查询 Q0<br/>不含位姿"] --> ENC
    CAMADD --> ENC["记忆编码器 §3.2（2 个 block）<br/>紧凑 self-attention（Compact/Expand）+ FFN"]
    ENC --> MEM["固定大小记忆 token m"]
    MEM --> DIT["扩散 DiT 骨干<br/>沿时间轴与目标 latent 拼接"]
    ACT["动作序列 a / 动作嵌入"] --> DIT
    DIT --> NEXT["生成下一段 latent 帧<br/>（自回归 rollout）"]
    NEXT -->|"追加到历史"| HIST
    DIT --> LFM["流匹配损失 LFM"]
    subgraph TRAIN["仅训练期：相机可查询的几何监督 §3.3"]
        SAMP["随机采样一个历史帧 i 的相机 ci"] --> RAY["射线图 → MLP + 2D 位置嵌入 → 查询 qi"]
        RAY --> GH["几何头：CrossAttn(qi, m) → SelfAttn → FFN"]
        GH --> LGEO["与冻结 VGGT 特征做逐 patch 余弦损失 Lgeo"]
    end
    MEM --> GH
    LFM --> LOSS["总损失 L = LFM + λ·Lgeo<br/>记忆编码器、几何头、骨干端到端联合训练"]
    LGEO --> LOSS
```

图依据：论文 §3.2–§3.5 / Fig.2（推理时丢弃 VGGT 与几何头）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 078/571 |
| 分组 | 42 Visual / Video World Models |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.02436> |

## 核心机制（归纳）

### 策展导读要点

Proposes a geometry-aware implicit memory framework that compresses variable-length histories into fixed-size memory tokens via a lightweight transformer.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.02436) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **42 Visual / Video World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Geometry-Aware Implicit Memory for Video World Models」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **42 Visual / Video World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2606_02436_geometry-aware-implicit-memory-for-video.md`](../../sources/papers/sun_awesome_wm_2606_02436_geometry-aware-implicit-memory-for-video.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2606.02436>
- [arXiv PDF](https://arxiv.org/pdf/2606.02436)：已通读方法部分（实验与附录以略读为准）：§1 引言、§3.2 记忆编码器、§3.3 几何监督、§3.4 信息引导剪枝、§3.5 训练与推理、§4.1 实验设定；§2 相关工作与实验结果略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2606.02436)
