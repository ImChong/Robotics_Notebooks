---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2605.10347"
venue: "arXiv 2026"
summary: "Investigates how mobile world models guide GUI agents."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2605_10347_how-mobile-world-model-guides-gui-agents.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# How Mobile World Model Guides GUI Agents?

**How Mobile World Model Guides GUI Agents?** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 504/571** 篇，分组 **83 GUI / Computer-Use Agents**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Investigates how mobile world models guide GUI agents.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Investigates how mobile world models guide GUI agents.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    D["移动端 GUI 轨迹数据集（AITW / AMEX / AITZ / AndroidControl / GUI-Odyssey）"] --> T["转为状态转移三元组 (s_t, a_t, s_t+1)"]
    T --> F["过滤：图级去重 → 渲染保真度过滤 → 动作逻辑过滤 → 页面噪声移除"]
    F --> L["用 VLM 重标注：delta 描述 / 完整描述 / 可渲染 HTML 代码"]
    L --> W["训练 Mobile World Model：输入 (s_t, a_t)，四种预测格式"]
    W --> W1["Delta Text：只预测变化"]
    W --> W2["Full Text：完整描述下一页面"]
    W --> W3["Code2Image：预测可渲染代码再渲染成图"]
    W --> W4["Diffusion Image：直接生成下一截图"]
    W1 --> U
    W2 --> U
    W3 --> U
    W4 --> U
    U["三种用法"] --> U1["先验感知 prior：决策前查询世界模型"]
    U --> U2["后验 posterior：评判 / 反思已采样动作（可由熵门控触发）"]
    U --> U3["想象训练：Code2Image 作虚拟环境采轨迹，过滤后微调 action model"]
```

图依据：论文 §2.1–2.2（数据构建与四种预测格式）、§4.2（先验 / 后验交互）、§5（想象训练）；Fig.2、Fig.4

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 504/571 |
| 分组 | 83 GUI / Computer-Use Agents |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2605.10347> |

## 核心机制（归纳）

### 策展导读要点

Investigates how mobile world models guide GUI agents.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2605.10347) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **83 GUI / Computer-Use Agents** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「How Mobile World Model Guides GUI Agents?」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **83 GUI / Computer-Use Agents**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2605_10347_how-mobile-world-model-guides-gui-agents.md`](../../sources/papers/sun_awesome_wm_2605_10347_how-mobile-world-model-guides-gui-agents.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2605.10347>
- [arXiv PDF](https://arxiv.org/pdf/2605.10347)：已通读方法部分（实验与附录以略读为准）：§1 引言、§2 构建 Mobile World Model（预测格式、转移级数据构建、数据统计）、§4.2 交互模式、§5 想象训练；§3 与 §4.1、§4.3 以略读为准

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2605.10347)
