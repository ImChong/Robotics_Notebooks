---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2607.02517"
venue: "arXiv 2026"
summary: "Decouples semantic motion orchestration from visual generation."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2607_02517_worlddirector-building-controllable-worl.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# WorldDirector

**WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 169/571** 篇，分组 **51 General Interactive Frameworks**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Decouples semantic motion orchestration from visual generation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Decouples semantic motion orchestration from visual generation.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TB
  subgraph INF["推理：LLM 世界规划"]
    IMG["初始图像"] --> EST["SAM + DepthAnything v2<br/>估计目标物体 3D 框与初始朝向"]
    EST --> LLM["LLM (Gemini)<br/>规划物体 3D 框轨迹 + 相机路径"]
    LLM --> PROJ["3D 轨迹投影到 2D 图像平面<br/>→ 位置条件 B (彩色 2D 框视频)"]
  end
  PROJ --> CH["按时间切分为连续 chunk，因果逐 chunk 生成"]
  subgraph GEN["每个 chunk 的视频生成（基于 LingBot-World-Base DiT）"]
    A["外观条件 A：参考裁剪映射到框位置<br/>(Temporal Drop：入镜后前 16 帧稠密，之后每 6 帧留 1 帧)"]
    VAE["3D VAE 编码 B 与 A<br/>与带噪 latent 通道拼接"]
    M["上下文记忆 M：静态(FoV 重叠 top-K)+动态(含活跃物体的帧)<br/>检索，序列拼接，非对称注意力使 M 不受噪声污染"]
    CAM["相机位姿 → Plücker 坐标 → 卷积嵌入<br/>经自适应归一化注入每个 DiT block"]
    TXT["全局提示 + 实体级提示<br/>Spatial-Aware 加权 cross-attention"]
    A --> VAE --> DIT["DiT flow matching 去噪 → 当前 chunk"]
    M --> DIT
    CAM --> DIT
    TXT --> DIT
  end
  CH --> GEN
  DIT --> POOL["已生成 chunk 池"]
  POOL -->|"更新 A、检索 M，进入下一 chunk"| GEN
```

图依据：论文 §3.2–3.4 / Fig.2、附录 C（首个 chunk 仅用初始参考帧作 A，M 为空；训练数据管线 §3.1 未入图）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 169/571 |
| 分组 | 51 General Interactive Frameworks |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2607.02517> |

## 核心机制（归纳）

### 策展导读要点

Decouples semantic motion orchestration from visual generation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2607.02517) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **51 General Interactive Frameworks** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「WorldDirector」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **51 General Interactive Frameworks**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2607_02517_worlddirector-building-controllable-worl.md`](../../sources/papers/sun_awesome_wm_2607_02517_worlddirector-building-controllable-worl.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2607.02517>
- [arXiv PDF](https://arxiv.org/pdf/2607.02517)：已通读方法部分（实验与附录以略读为准）：§3 Method（3.1–3.4）、附录 B/C 的推理与检索细节

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2607.02517)
