---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2605.06337"
venue: "arXiv 2026"
summary: "Grid-free, observation-native atmospheric world model matching the fidelity of established physical frameworks."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2605_06337_earth-o1-a-grid-free-observation-native.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Earth-o1

**Earth-o1: A Grid-free Observation-native Atmospheric World Model** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 519/571** 篇，分组 **84 Scientific Discovery**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Grid-free, observation-native atmospheric world model matching the fidelity of established physical frameworks.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Grid-free, observation-native atmospheric world model matching the fidelity of established physical frameworks.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    OBS["多源异构观测<br/>地球静止/极轨卫星 + 站点/船舶/飞机等 in-situ"] --> TS["卫星 tokenizer：滑窗子图 + 缺测 mask 拼为通道<br/>masked attention 编码，形态学腐蚀筛 token"]
    OBS --> TI["in-situ tokenizer：观测聚合到空间锚点 (KNN + point attention)<br/>再下采样到与卫星 token 对齐"]
    TS --> PR["MMAE 模态投影器 + 模态 norm<br/>缺失/被 mask 位置用各模态可学习 domain token 填充"]
    TI --> PR
    PR --> LAT["MMAE 融合得到统一 Latent Observation Space (LatentOS)<br/>损失：MAE 重建 + in-situ 联合站点损失"]
    LAT --> REC["重建：解码到任意位置的观测量"]
    LAT --> FC["Forecast：由前 T 步各模态表示预测后 T 步<br/>统一 patch embedding + attention 块 + 各模态线性解码器"]
    FC --> INV["Inversion：把预测观测反演为下游产品<br/>如海冰、CO2、降水"]
```

图依据：论文 §1 Main（三模块：reconstruction / prediction / inversion）与 §4 Method（4.1 观测 tokenization、4.2 MMAE 融合、4.3 Forecast 与 Inversion）/ Fig. 7。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 519/571 |
| 分组 | 84 Scientific Discovery |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2605.06337> |

## 核心机制（归纳）

### 策展导读要点

Grid-free, observation-native atmospheric world model matching the fidelity of established physical frameworks.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2605.06337) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **84 Scientific Discovery** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Earth-o1」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **84 Scientific Discovery**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2605_06337_earth-o1-a-grid-free-observation-native.md`](../../sources/papers/sun_awesome_wm_2605_06337_earth-o1-a-grid-free-observation-native.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2605.06337>
- [arXiv PDF](https://arxiv.org/pdf/2605.06337)：已通读方法部分（实验与附录以略读为准）：§1 Main、§3 Discussion、§4 Method（4.1–4.3）、§5 Dataset 开头

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2605.06337)
