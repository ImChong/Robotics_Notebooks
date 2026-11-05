---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2503.22405"
venue: "CVPR 2025"
summary: "Proposes the Adaptive Multiple Normal Action Representation (AMNAR) framework, which predicts all valid next actions and reconstructs their corresponding normal representations to detect fine-grained execution deviations"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2503_22405_modeling-multiple-normal-action-represen.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# Modeling Multiple Normal Action Representations for Error Detection in Proced...

**Modeling Multiple Normal Action Representations for Error Detection in Procedural Tasks** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 121/249** 篇，分组 **291 Procedural Error Detection**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Proposes the Adaptive Multiple Normal Action Representation (AMNAR) framework, which predicts all valid next actions and reconstructs their corresponding normal representations to detect fine-grained execution deviations in procedural tasks

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Proposes the Adaptive Multiple Normal Action Representation (AMNAR) framework, which predicts all valid next actions and reconstructs their corresponding normal representations to detect fine-grained execution deviations in procedural tasks
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    VID["过程性任务视频"] --> FE["视觉特征提取器（如 I3D）"]
    FE --> ASM["动作分割模型 ASM<br/>输出动作段 (标签, 起止帧) 与帧级特征 F"]
    ASM --> SEQ["已执行动作序列 s_t"]
    ASM --> FACT["当前动作特征 f_t^action<br/>段内帧特征取平均"]
    TG["任务图 G（预定义）"] --> PAPB
    SEQ --> PAPB["潜在动作预测块 PAPB<br/>动态规划求最长公共子序列并合并 → s_t*<br/>取子节点并去掉已执行动作 → 有效下一动作 C_t"]
    PAPB --> RRB["表示重建块 RRB<br/>帧特征经扩张卷积；C_t 作 query 做局部交叉注意力<br/>预测残差 r，与类簇中心 c 相加得 f_normal"]
    ASM --> RRB
    RRB --> RMB["表示匹配块 RMB<br/>计算 f_action 与各 f_normal 的欧氏距离，取最小 d_min"]
    FACT --> RMB
    RMB --> TH{"d_min 是否大于该动作类阈值 θ(y_t)"}
    TH -->|是| ERR["判定为错误动作"]
    TH -->|否| OK["判定为正常动作"]
    TRN["训练阶段：仅用正常视频；省略 PAPB，RRB query 数为 1<br/>最小化 f_normal 与 f_action 的距离"] -.-> RRB
```

图依据：论文 §3.2 Overview / Fig.2、§3.3–3.7（PAPB 见 Fig.3 与 Fig.5）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 121/249 |
| 分组 | 291 Procedural Error Detection |
| 出处 | CVPR 2025 |
| 论文 | <https://arxiv.org/abs/2503.22405> |

## 核心机制（归纳）

### 策展导读要点

Proposes the Adaptive Multiple Normal Action Representation (AMNAR) framework, which predicts all valid next actions and reconstructs their corresponding normal representations to detect fine-grained execution deviations in procedural tasks

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2503.22405) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **291 Procedural Error Detection** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Modeling Multiple Normal Action Representations for Error Detection in Proced...」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **291 Procedural Error Detection**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2503_22405_modeling-multiple-normal-action-represen.md`](../../sources/papers/sun_awesome_ego_2503_22405_modeling-multiple-normal-action-represen.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2503.22405>
- [arXiv PDF](https://arxiv.org/pdf/2503.22405)：已通读方法部分（实验与附录以略读为准）：§3.1–3.7

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2503.22405)
