---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2605.18431"
venue: "arXiv 2026"
summary: "Multi-robot cooperative egocentric spatial reasoning using multimodal large language models"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2605_18431_seeing-together-multi-robot-cooperative.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# Seeing Together

**Seeing Together: Multi-Robot Cooperative Egocentric Spatial Reasoning with Multimodal Large Language Models** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 098/249** 篇，分组 **27 Embodied AI & Vision-Language-Action Ego-VLA**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Multi-robot cooperative egocentric spatial reasoning using multimodal large language models

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Multi-robot cooperative egocentric spatial reasoning using multimodal large language models
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TB
    VID["N 个机器人的同步第一视角视频 V_r + 问题 q"] --> SAMP
    subgraph SAMP["SE-MR2FS: 免训练频谱能量感知帧采样 (每个机器人独立)"]
        direction TB
        C1["冻结 CLIP 编码帧与问题, 余弦相似度取 top-M 候选"] --> C2["候选内对 CLIP 特征做时间维 FFT, 得频率能量 e"]
        C2 --> C3["能量 + 语义项综合打分, 保留 K 帧"]
    end
    SAMP --> TOK["MLLM 视觉编码器给每个机器人 K 个视觉 token<br/>低频幅度谱经 MLP → 谱 token F_r"]
    TOK --> FUSION
    POSE["位姿/运动指令嵌入 P_r (训练时由仿真或动捕位姿给出)"] --> FUSION
    subgraph FUSION["SGI-MRF: 谱与几何感知多机器人融合"]
        direction TB
        A1["辅助 token A_r = W_p P_r + W_f F_r + 机器人角色嵌入 E_r"] --> A2["视觉状态 μ_r 对 A_r 做 cross-attention"]
        A2 --> A3["由视觉特征估计不确定性 σ_r → softmax 可靠性权重 w_r"]
        A3 --> A4["加权聚合为全局团队信念 z"]
    end
    FUSION --> STU["学生提示 P_S = g_S(z, C), 推理时只用学生"]
    TEACH["GMIPSD 训练期教师提示 P_T: 由真值位姿嵌入生成"] -.->|"余弦距离蒸馏 L_distill"| STU
    STU --> LM["与 W_z z 及问题嵌入拼接成前缀 → 语言模型 LM → 答案 y"]
```

图依据：论文 §4.1–4.2 / Fig.2

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 098/249 |
| 分组 | 27 Embodied AI & Vision-Language-Action Ego-VLA |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2605.18431> |

## 核心机制（归纳）

### 策展导读要点

Multi-robot cooperative egocentric spatial reasoning using multimodal large language models

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2605.18431) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **27 Embodied AI & Vision-Language-Action Ego-VLA** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Seeing Together」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **27 Embodied AI & Vision-Language-Action Ego-VLA**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md)
- 技术地图：[Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md)
- 方法/任务：[vla.md](../methods/vla.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_ego_2605_18431_seeing-together-multi-robot-cooperative.md`](../../sources/papers/sun_awesome_ego_2605_18431_seeing-together-multi-robot-cooperative.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2605.18431>
- [arXiv PDF](https://arxiv.org/pdf/2605.18431)：已通读方法部分（实验与附录以略读为准）：§4 方法（4.1 问题形式化、4.2 SG-CoR：SE-MR2FS、SGI-MRF、GMIPSD、训练目标）；§3 基准构建与 §5 实验以略读为准

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2605.18431)
