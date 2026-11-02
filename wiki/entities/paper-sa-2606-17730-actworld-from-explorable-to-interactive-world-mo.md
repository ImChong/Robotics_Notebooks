---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2606.17730"
venue: "arXiv 2026"
summary: "Extends navigation-centric generators to support object interactions."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2606_17730_actworld-from-explorable-to-interactive.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# ActWorld

**ActWorld: From Explorable to Interactive World Model via Action-Aware Memory** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 155/571** 篇，分组 **51 General Interactive Frameworks**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Extends navigation-centric generators to support object interactions.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Extends navigation-centric generators to support object interactions.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    V["训练视频切成 33 帧 chunk<br/>每个 chunk 取 5 个关键帧"] --> CoT["GPT-5.4 做 CoT 逐 chunk 标注<br/>交互标志、6 类交互阶段、1-2 句描述"]
    CoT --> TXT["冻结 UMT5 编码 chunk 描述<br/>作为 cross-attention 条件"]
    CoT --> LAB["结构化标签（交互标志、阶段）<br/>+ 视频级动作类别 a"]
    KM["键盘 / 鼠标控制"] --> PL["相机轨迹 → Plücker 射线<br/>PlückerFiLM 逐 token 缩放与平移（仅当前 chunk）"]
    KM --> SYM["命令映射为文本模板<br/>UMT5 编码，与 chunk 描述拼接"]
    SYM --> TXT
    HIST["过往 chunk 历史"] --> EAFR["EAFR：按阶段先验 + 近因打分<br/>重新分配 fine / mid / coarse 三档历史桶"]
    LAB --> EAFR
    LAB --> ACHA["ACHA：动作类别 embedding<br/>放大自注意力中的历史 key"]
    LAB --> BANK["持久动作记忆库（至多 16 token，FIFO + 阶段固定）<br/>事件 token + DINOv3 物体 token"]
    HIST --> BANK
    EAFR --> DIT["自回归 DiT 逐 chunk 生成"]
    ACHA --> DIT
    BANK --> DIT
    PL --> DIT
    TXT --> DIT
    DIT --> OUT["新 chunk"]
    OUT -. "进入历史；交互 chunk 写入记忆库" .-> HIST
    DIT --> DST["少步蒸馏：多分辨率 flow matching + DMD 式蒸馏<br/>50 步教师 → 3 步生成器"]
```

图依据：论文 §3.1–3.4 / Fig.2

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 155/571 |
| 分组 | 51 General Interactive Frameworks |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.17730> |

## 核心机制（归纳）

### 策展导读要点

Extends navigation-centric generators to support object interactions.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.17730) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **51 General Interactive Frameworks** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「ActWorld」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2606_17730_actworld-from-explorable-to-interactive.md`](../../sources/papers/sun_awesome_wm_2606_17730_actworld-from-explorable-to-interactive.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2606.17730>
- [arXiv PDF](https://arxiv.org/pdf/2606.17730)：已通读方法部分（实验与附录以略读为准）：§3 方法（3.1–3.4）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2606.17730)
