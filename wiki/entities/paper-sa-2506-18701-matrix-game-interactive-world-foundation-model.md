---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2506.18701"
venue: "arXiv 2025"
summary: "An interactive world foundation model for controllable game‑world generation, trained with a two‑stage pipeline."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2506_18701_matrix-game-interactive-world-foundation.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Matrix-Game

**Matrix-Game: Interactive World Foundation Model** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 035/571** 篇，分组 **113 Interactive World Models & Game Engines**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

An interactive world foundation model for controllable game‑world generation, trained with a two‑stage pipeline.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- An interactive world foundation model for controllable game‑world generation, trained with a two‑stage pipeline.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph DATA["Matrix-Game-MC 数据"]
        U["未标注 Minecraft 视频<br/>MineDojo 约 6000 h"] --> F["分层过滤：质量/美学 → 菜单/字幕/人脸 → 运动与相机"]
        L["带动作标注数据<br/>MineRL 探索 agent + Unreal 程序化仿真"]
    end
    F --> S1["阶段 1：无标注预训练<br/>2700 h，再精选 870 h；无动作模块"]
    S1 --> S2["阶段 2：带动作训练<br/>加入控制模块，约 1200 h"]
    L --> S2
    subgraph INF["推理：图像到世界"]
        R["单张参考图像"] --> VE["视觉编码器"]
        N["高斯噪声 + 运动帧（末 k=5 帧）<br/>通道拼接 + 二值掩码"] --> PE["Patch Embedding"]
        A["用户动作<br/>键盘离散 / 鼠标连续"] --> CM["控制模块<br/>键盘 cross-attention / 鼠标 self-attention"]
        VE --> DiT["MMDiT 扩散 Transformer<br/>（双流 + 单流块）"]
        PE --> DiT
        CM --> DiT
        DiT --> DEC["3D Causal VAE 解码器"] --> V["视频片段"]
        V -- "末几帧作下一段运动条件" --> N
    end
    S2 -. "训练得到" .-> DiT
```

图依据：论文 §3.1–3.2 / Fig.3、§4.1 / Fig.4–6、§4.2（两阶段训练）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 035/571 |
| 分组 | 113 Interactive World Models & Game Engines |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2506.18701> |

## 核心机制（归纳）

### 策展导读要点

An interactive world foundation model for controllable game‑world generation, trained with a two‑stage pipeline.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.18701) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **113 Interactive World Models & Game Engines** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Matrix-Game」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **113 Interactive World Models & Game Engines**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2506_18701_matrix-game-interactive-world-foundation.md`](../../sources/papers/sun_awesome_wm_2506_18701_matrix-game-interactive-world-foundation.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2506.18701>
- [arXiv PDF](https://arxiv.org/pdf/2506.18701)：已通读方法部分（实验与附录以略读为准）：§3 数据集构建、§4.1 模型架构、§4.2 训练。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2506.18701)
