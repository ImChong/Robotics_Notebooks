---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2510.22129"
venue: "NeurIPS 2025"
summary: "Multimodal dataset collected via Meta Project Aria glasses, combining video, eye tracking and biometric signals for real-life emotion analysis"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2510_22129_egoemotion-egocentric-vision-and-physiol.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# egoEMOTION

**egoEMOTION: Egocentric Vision and Physiological Signals for Emotion and Personality Recognition in Real-World Tasks** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 066/249** 篇，分组 **25 Gaze Attention & Intention Prediction**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Multimodal dataset collected via Meta Project Aria glasses, combining video, eye tracking and biometric signals for real-life emotion analysis

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Multimodal dataset collected via Meta Project Aria glasses, combining video, eye tracking and biometric signals for real-life emotion analysis
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    REC["§3.2 招募：43 名参与者，每人单次录制约 105 分钟；到场后讲解协议、签署知情同意、佩戴传感器"]
    REC --> CAL["开始 Session A 前做 eye-tracking 校准"]
    CAL --> A["Session A（§3.1.1）：9 段诱发视频（8 种 Mikels' Wheel 情绪 + 1 个中性），每段前先看 5 秒注视十字"]
    CAL --> B["Session B：7 项自然活动（Flappy Bird、Jelly Bean、Jenga、Painting、Sad Letter、Slenderman、Try to Laugh）"]
    A --> SR["每个任务后的自评（§3.1.2）：emoti-SAM（Arousal / Valence / Dominance，7 点）+ 加权情绪标签（9 类，100% 以 10% 为步长分配）；随后看中性云视频以缓解残留效应"]
    B --> SR
    BFI["实验前在线填写 BFI-2 人格问卷（Big Five）"]
    SENS["§3.1.3 同步采集：Project Aria 眼镜（eye-tracking 视频、POV 相机、双 IMU）、鼻托 PPG；Shimmer3（耳部 PPG、手指 EDA）、ECG 胸带、呼吸带（RSP）；网络摄像头记录面部"]
    A --> SENS
    B --> SENS
    SENS --> COMP["§3.3 数据集构成：以 IMU 摇动在实验起止同步各传感器；按参与者组织；人工标注每个任务的起止"]
    SR --> COMP
    BFI --> COMP
    COMP --> FE["§5.1 特征提取：612 个特征（ECG/PPG、EDA、呼吸、瞳孔、眼动、眨眼、微表情、加速度等）"]
    FE --> T1["任务 1 连续情感（V-A-D，按训练集中位数二值化）：SVM(RBF)"]
    FE --> T2["任务 2 离散情绪（9 类，取自评最强情绪）：Random Forest + SelectKBest(top 10)"]
    FE --> T3["任务 3 人格特质（Big Five，按训练集中位数二值化）：Random Forest + SelectKBest(top 10)"]
    T1 --> EVAL["评测协议：leave-one-subject-out 交叉验证，报告 F1；另对比 CNN 与 transformer（WER）深度学习模型（§5.5）"]
    T2 --> EVAL
    T3 --> EVAL
```

图依据：数据采集与基准评测流程图，论文 Fig. 1、Fig. 2、§3.1（协议、标注、传感器）、§3.2–§3.3（招募与数据集构成）、§5.1–§5.5（特征提取与三个基准任务）；不含结果数字。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 066/249 |
| 分组 | 25 Gaze Attention & Intention Prediction |
| 出处 | NeurIPS 2025 |
| 论文 | <https://arxiv.org/abs/2510.22129> |

## 核心机制（归纳）

### 策展导读要点

Multimodal dataset collected via Meta Project Aria glasses, combining video, eye tracking and biometric signals for real-life emotion analysis

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2510.22129) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **25 Gaze Attention & Intention Prediction** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「egoEMOTION」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **25 Gaze Attention & Intention Prediction**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2510_22129_egoemotion-egocentric-vision-and-physiol.md`](../../sources/papers/sun_awesome_ego_2510_22129_egoemotion-egocentric-vision-and-physiol.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2510.22129>
- [arXiv PDF](https://arxiv.org/pdf/2510.22129)：已通读摘要、§1–§3（数据集设计、招募、构成）、§4 描述性分析、§5 基线（5.1–5.6）与 §6 局限；附录未读

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2510.22129)
