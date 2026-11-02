---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2605.30263"
venue: "arXiv 2026"
code: https://github.com/shengshu-ai/minWM
summary: "Full-stack open-source framework for real-time interactive video world models."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2605_30263_minwm-a-full-stack-open-source-framework.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# minWM

**minWM: A Full-Stack Open-Source Framework for Real-Time Interactive Video World Models** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 571/571** 篇，分组 **Tutorials & Starter Resources**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Full-stack open-source framework for real-time interactive video world models.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Full-stack open-source framework for real-time interactive video world models.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    B["双向多步 T2V / TI2V 基础模型<br/>Wan2.1-T2V-1.3B、HY1.5-TI2V-8B"] --> P1["阶段一：相机可控微调<br/>PRoPE 将相机内外参注入自注意力"]
    D["相机标注数据<br/>DL3DV 重建后沿给定轨迹重渲染<br/>或 WorldPlay 按轨迹生成视频"] --> P1
    P1 --> BI["相机可控的双向多步扩散模型"]
    BI --> S1["蒸馏 Stage 1：AR 扩散训练<br/>teacher forcing 得到自回归多步模型"]
    S1 --> S2["Stage 2：初始化少步 AR 模型<br/>(a) 因果 ODE 初始化 或 (b) 因果 CD 初始化"]
    S2 --> S3["Stage 3：非对称 DMD<br/>学生自回放 rollout，双向模型提供 s_real"]
    BI -.->|"s_real / s_fake 初始化"| S3
    S3 --> M["相机可控的少步自回归世界模型"]
    M --> INF["低延迟流式推理<br/>流式 DiT 去噪 + 流式 VAE 解码"]
```

图依据：论文 §2.1–§2.2、§3.3 / Fig. 1（推理环节取自 Fig. 1 的框内文字）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 571/571 |
| 分组 | Tutorials & Starter Resources |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2605.30263> |
| 代码/项目 | <https://github.com/shengshu-ai/minWM> |

## 核心机制（归纳）

### 策展导读要点

Full-stack open-source framework for real-time interactive video world models.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2605.30263) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **Tutorials & Starter Resources** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「minWM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Tutorials & Starter Resources**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2605_30263_minwm-a-full-stack-open-source-framework.md`](../../sources/papers/sun_awesome_wm_2605_30263_minwm-a-full-stack-open-source-framework.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2605.30263>
- [arXiv PDF](https://arxiv.org/pdf/2605.30263)：已通读方法部分（实验与附录以略读为准）：摘要、§1、§2.1–§2.2、§3.1–§3.3、Fig. 1

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2605.30263)
