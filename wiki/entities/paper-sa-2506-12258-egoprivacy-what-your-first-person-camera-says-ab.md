---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2506.12258"
venue: "ICML 2025"
summary: "The first large‑scale benchmark for egocentric privacy risk assessment, systematically answering 'what can be inferred about the wearer from first‑person video?'. Covers three privacy categories (demographic, individual,"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2506_12258_egoprivacy-what-your-first-person-camera.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# EgoPrivacy

**EgoPrivacy: What Your First-Person Camera Says About You?** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 003/249** 篇，分组 **2101 Privacy Risk Assessment & Benchmarking**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

The first large‑scale benchmark for egocentric privacy risk assessment, systematically answering "what can be inferred about the wearer from first‑person video?". Covers three privacy categories (demographic, individual, contextual) and defines seven granularity‑varying privacy recovery tasks. Proposes a **Retrieval...

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- The first large‑scale benchmark for egocentric privacy risk assessment, systematically answering "what can be inferred about the wearer from first‑person video?". Covers three privacy categories (demographic, individual, contextual) and defines seven granularity‑varying privacy recovery tasks. Proposes a **Retrieval...
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph BM["EgoPrivacy 基准（§3）"]
        DS["Ego-Exo4D（训练/ID 测试）+ Charades-Ego（OOD 测试）<br/>同步 ego-exo 视频；人工标注性别/年龄/种族"] --> TK["七项隐私任务：<br/>人口统计（性别、年龄、种族）<br/>个体（ego-to-ego / ego-to-exo 身份检索）<br/>情境（场景检索、时刻检索）"]
    end
    subgraph CAP["攻击者能力（§4）"]
        C1["能力 1：零样本基础模型"]
        C2["能力 2：在 Dtrain 上微调"]
        C3["能力 3：检索增强（RAA）"]
    end
    TK --> CAP
    subgraph RAA["检索增强攻击 RAA（§5，Fig.2）"]
        Q["第一人称查询视频 x(E)"] --> EMB["ego-exo 联合嵌入 g, g'<br/>（监督对比学习 SupCon 训练）"]
        POOL["外部未标注第三人称视频池 D(X)"] --> EMB
        EMB --> TOPM["按相似度取 top-M 第三人称视频"]
        Q --> FE["直接第一人称攻击 f(x(E))"]
        TOPM --> FX["对检索到的视频做第三人称攻击 f'(x(X)_i)"]
        FE --> AGG["聚合函数 A：多数投票 / 加权池化"]
        FX --> AGG
        AGG --> PRED["最终隐私预测"]
    end
    C3 --> RAA
```

图依据：论文 §3.2、§4.1–4.2、§5.1–5.2 / Fig.2（RAA 公式 f_RAA = A(f(x_E), f'(x_1), …, f'(x_M))）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 003/249 |
| 分组 | 2101 Privacy Risk Assessment & Benchmarking |
| 出处 | ICML 2025 |
| 论文 | <https://arxiv.org/abs/2506.12258> |

## 核心机制（归纳）

### 策展导读要点

The first large‑scale benchmark for egocentric privacy risk assessment, systematically answering "what can be inferred about the wearer from first‑person video?". Covers three privacy categories (demographic, individual, contextual) and defines seven granularity‑varying privacy recovery tasks. Proposes a **Retrieval...

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.12258) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **2101 Privacy Risk Assessment & Benchmarking** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EgoPrivacy」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **2101 Privacy Risk Assessment & Benchmarking**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2506_12258_egoprivacy-what-your-first-person-camera.md`](../../sources/papers/sun_awesome_ego_2506_12258_egoprivacy-what-your-first-person-camera.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2506.12258>
- [arXiv PDF](https://arxiv.org/pdf/2506.12258)：已通读方法部分（实验与附录以略读为准）：§3 基准设计、§4 攻击者能力与实现、§5 检索增强攻击

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2506.12258)
