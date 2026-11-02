---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2501.16443"
venue: "arXiv 2025"
summary: "Object-centric world models improve RL in visually complex environments."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2501_16443_oc-storm-objects-matter-object-centric-w.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# OC-STORM

**OC-STORM: Objects matter – object-centric world models improve RL in visually complex environments** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 131/571** 篇，分组 **45 Object-Centric World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Object-centric world models improve RL in visually complex environments.（arXiv 当前版本题为《Object-Centric World Models from Few-Shot Annotations for Sample-Efficient Reinforcement Learning》；本页沿用清单所列题名）

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Object-centric world models improve RL in visually complex environments.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    ANN["人工标注关键物体<br/>仅少量帧（如 6–12 帧）"] --> SEG
    OBS["环境观测 o_t"] --> SEG["冻结的预训练视频分割网络<br/>Cutie 或 SAM2，对象注意力输出<br/>每集开始重置内部状态"]
    SEG --> OBJ["物体特征 s_obj：K × obj_dim<br/>K 个物体，用户按环境设定"]
    OBS --> VIS["缩放为 64×64 的视觉输入 s_vis"]
    OBJ --> ENC1["MLP 编码器 + Categorical VAE<br/>物体离散潜变量 16×16"]
    VIS --> ENC2["CNN 编码器 + Categorical VAE<br/>视觉离散潜变量 32×32"]
    ENC1 --> DYN
    ENC2 --> DYN
    ACT["动作 a_t"] --> DYN["时空物体中心动力学<br/>STORM：空间注意力与时间注意力交替<br/>或 DreamerV3：RNN + 空间注意力"]
    DYN --> HEADS["预测头<br/>动力学预测器预测下一潜状态<br/>[CLS] 式查询 token 聚合后预测奖励与终止"]
    HEADS --> WM["世界模型<br/>以重建与预测损失端到端训练"]
    WM -->|"想象轨迹"| AC["Actor-Critic（沿用 DreamerV3）<br/>策略梯度 + 价值基线"]
    AC -->|"策略与环境交互，采集新数据"| OBS
```

图依据：论文 §3.1–§3.5 与 Fig. 1c；“策略与环境交互采集新数据”一支由 §2 所述标准 MBRL 范式（先学世界模型、再用模型生成轨迹训练策略）推断。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 131/571 |
| 分组 | 45 Object-Centric World Models |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2501.16443> |

## 核心机制（归纳）

### 策展导读要点

Object-centric world models improve RL in visually complex environments.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2501.16443) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **45 Object-Centric World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「OC-STORM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **45 Object-Centric World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2501_16443_oc-storm-objects-matter-object-centric-w.md`](../../sources/papers/sun_awesome_wm_2501_16443_oc-storm-objects-matter-object-centric-w.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2501.16443>
- [arXiv PDF](https://arxiv.org/pdf/2501.16443)：已通读方法部分（实验与附录以略读为准）：§1 引言末段、§2 Preliminaries、§3 Method（3.1–3.5）。注：当前 arXiv 版标题为 Object-Centric World Models from Few-Shot Annotations（ICLR 2026），与页面标题措辞不同，编号一致
- 核对（2026-11-02）：查 arXiv 摘要页版本史：v1（2025-01-27）题 Objects matter: object-centric world models improve reinforcement learning in visually complex environments，v2（2026-02-25，现行）改题；本页题名对应 v1；已补题名变化说明；本次只核对版本史与题名，流程图及方法细节未重读，沿用此前核对。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2501.16443)
