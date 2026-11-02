---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2606.27677"
venue: "arXiv 2026"
summary: "Augments WAM with Diverse Historical Event Memory (DHEM)."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2606_27677_dim-wam-world-action-modeling-with-diver.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# DiM-WAM

**DiM-WAM: World Action Modeling with Diverse Historical Event Memory** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 190/571** 篇，分组 **62 Joint WAM & Unified VLA-World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Augments WAM with Diverse Historical Event Memory (DHEM).

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Augments WAM with Diverse Historical Event Memory (DHEM).
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    OBS["决策段 i 结束后的当前观测：视觉特征 Xi"] --> W
    subgraph DHEM["DHEM：K 个并行记忆库，每库 N 个有界槽位（成本 O(KN)）"]
        direction TB
        W["写入：各库用可学习 summary query 对同一观测做注意力汇总，得候选事件特征"] --> MG["维护：槽 1 固定为初始状态锚点，槽 N 为最新事件；库满时按冗余优先级（余弦相似 + 时间衰减）合并最冗余的相邻中间历史对，或丢弃冗余的新候选"]
    end
    MG --> RD["协作读取：有效事件 token 按槽优先、库次之展平，加 bank 身份嵌入与 RoPE 槽位时间顺序，得可读记忆 token 序列 Ri"]
    LOC["滑动窗口 WAM 的局部 KV 上下文 Ci"] --> WAM
    LANG["语言指令 c"] --> WAM
    RD --> WAM["WAM：记忆 token 同时条件化视频与动作两个去噪分支"]
    WAM --> PRED["预测未来 H 步视觉潜变量 ẑ 与动作 â"]
    RD -.->|"仅训练"| PH["进度头：预测粗粒度任务进度区间（辅助损失 Lprog，不参与推理决策）"]
    MG -.->|"仅训练"| DV["多样性损失 Ldiv：抑制各记忆库塌缩"]
    PRED -.->|"执行后产生新观测"| OBS
```

图依据：论文 §III-A 至 III-E / Fig. 2–3、式 (2)–(19)

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 190/571 |
| 分组 | 62 Joint WAM & Unified VLA-World Models |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.27677> |

## 核心机制（归纳）

### 策展导读要点

Augments WAM with Diverse Historical Event Memory (DHEM).

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.27677) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **62 Joint WAM & Unified VLA-World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「DiM-WAM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **62 Joint WAM & Unified VLA-World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2606_27677_dim-wam-world-action-modeling-with-diver.md`](../../sources/papers/sun_awesome_wm_2606_27677_dim-wam-world-action-modeling-with-diver.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2606.27677>
- [arXiv PDF](https://arxiv.org/pdf/2606.27677)：已通读方法部分（实验与附录以略读为准）：§III-A 概览、§III-B 多样历史事件记忆库、§III-C 写入与维护、§III-D 协作读取、§III-E 任务进度感知训练

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2606.27677)
