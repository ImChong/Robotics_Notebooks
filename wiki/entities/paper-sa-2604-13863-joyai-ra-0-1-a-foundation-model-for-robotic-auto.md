---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2604.13863"
venue: "arXiv 2026"
summary: "VLA embodied foundation model with multi-source multi-level pretraining integrating web data, large-scale egocentric human manipulation videos, simulation trajectories, and real-robot data"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2604_13863_joyai-ra-0-1-a-foundation-model-for-robo.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# JoyAI-RA 0.1

**JoyAI-RA 0.1: A Foundation Model for Robotic Autonomy** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 092/249** 篇，分组 **27 Embodied AI & Vision-Language-Action Ego-VLA**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

VLA embodied foundation model with multi-source multi-level pretraining integrating web data, large-scale egocentric human manipulation videos, simulation trajectories, and real-robot data

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- VLA embodied foundation model with multi-source multi-level pretraining integrating web data, large-scale egocentric human manipulation videos, simulation trajectories, and real-robot data
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph SRC["多源数据"]
        WEB["网页多模态数据"]
        EGO["第一人称人类操作视频"]
        SIM["仿真轨迹"]
        REAL["真实机器人数据"]
    end
    UAS["统一动作空间：相机系末端位姿（平移 + 轴角）+ 固定维度动作向量，缺失自由度做 mask"]
    EGO -->|"重定向到目标本体"| UAS
    SIM --> UAS
    REAL --> UAS
    subgraph TRAIN["三阶段训练"]
        T1["阶段 1：VLM Co-Pretraining（自回归损失，含 VQA、FAST 离散动作 token）"]
        T2["阶段 2：VLA Co-Pretraining（自回归损失 + flow matching 损失）"]
        T3["阶段 3：目标机器人 Post-Training（仅 flow matching）"]
        T1 --> T2 --> T3
    end
    WEB --> T1
    UAS --> T1
    UAS --> T2
    subgraph MODEL["模型"]
        IN["多视角观测 + 语言指令 + 本体状态"]
        VLM["VLM：空间落地的多模态表征，先生成子任务文本"]
        EXP["感知-动作专家：Perceiver 注意力块，flow matching 预测速度场"]
        IN --> VLM --> EXP
    end
    T3 -.->|"训练得到"| MODEL
    EXP --> OUT["连续动作块 a_t:t+H"]
```

图依据：论文 §3.1–3.5 / Fig.1、Fig.3（注意：清单给出的 arXiv 号 2604.13863 实为另一篇论文，本图依据 JoyAI-RA 0.1 的正确论文 arXiv:2604.20100）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 092/249 |
| 分组 | 27 Embodied AI & Vision-Language-Action Ego-VLA |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2604.13863> |

## 核心机制（归纳）

### 策展导读要点

VLA embodied foundation model with multi-source multi-level pretraining integrating web data, large-scale egocentric human manipulation videos, simulation trajectories, and real-robot data

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2604.13863) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **27 Embodied AI & Vision-Language-Action Ego-VLA** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「JoyAI-RA 0.1」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_ego_2604_13863_joyai-ra-0-1-a-foundation-model-for-robo.md`](../../sources/papers/sun_awesome_ego_2604_13863_joyai-ra-0-1-a-foundation-model-for-robo.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2604.13863>
- [arXiv PDF](https://arxiv.org/pdf/2604.20100)：已通读方法部分（实验与附录以略读为准）：§3 Method（3.1 概览、3.2 数据源、3.3 统一动作空间、3.4 模型架构、3.5 训练配方）；注：本页 front matter 与上方链接的 2604.13863 与该论文标题不符，JoyAI-RA 0.1 的 arXiv 号为 2604.20100

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2604.13863)
