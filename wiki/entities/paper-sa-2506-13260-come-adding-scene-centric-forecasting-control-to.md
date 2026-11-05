---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2506.13260"
venue: "arXiv 2025"
code: https://github.com/synsin0/COME
summary: "Adds scene-centric forecasting control to occupancy world models."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2506_13260_come-adding-scene-centric-forecasting-co.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# COME

**COME: Adding Scene-Centric Forecasting Control to Occupancy World Model** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 401/571** 篇，分组 **82 Autonomous Driving**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Adds scene-centric forecasting control to occupancy world models.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Adds scene-centric forecasting control to occupancy world models.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    IN["历史占据 x(1:t) + 自车位姿 p(1:t)<br/>（可选 BEV layout、未来轨迹）"] --> SC
    subgraph SC["场景中心预测模块（UNet）"]
        T1["刚体变换：各帧占据对齐到当前自车系 x'(1:t)<br/>消除自车运动影响"] --> U["UNet（2D 卷积，时间维并入通道）→ 预测 y'(t+1:t+τ)"]
        U --> T2["变换回各未来时刻 ŷ(i) → Occ-VAE 编码 → 场景条件 z(i)"]
    end
    subgraph CN["COME ControlNet"]
        TC["前 N/2 块：世界模型时空块的可训练副本"] --> ZC["后 N/2 块：零卷积层 → 控制特征 c(i)"]
        VM["可见性感知掩码 M(i)：<br/>历史未观测区域特征置零（阈值 ε=0.5）"] --> ZC
    end
    T2 --> TC
    ZC --> INJ["残差相加注入世界模型后半部分块（与跳连特征相加）"]
    IN --> OWM
    subgraph OWM["占据世界模型（DOME 式 DiT）"]
        VAE["Occ-VAE 编码历史占据 → latent"] --> DIT["时空交替 Diffusion Transformer<br/>（轨迹条件注入每个块，早晚层跳连）"]
    end
    INJ --> DIT
    DIT --> OUT["Occ-VAE 解码 → 未来占据 x̂(t+1:t+τ)"]
```

图依据：论文 §3.1–3.4 / Fig.2；三阶段训练见 §3.4：先训世界模型，再用交叉熵训预测模块，最后冻结其余只训 ControlNet。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 401/571 |
| 分组 | 82 Autonomous Driving |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2506.13260> |
| 代码/项目 | <https://github.com/synsin0/COME> |

## 核心机制（归纳）

### 策展导读要点

Adds scene-centric forecasting control to occupancy world models.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.13260) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **82 Autonomous Driving** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「COME」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **82 Autonomous Driving**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2506_13260_come-adding-scene-centric-forecasting-co.md`](../../sources/papers/sun_awesome_wm_2506_13260_come-adding-scene-centric-forecasting-co.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2506.13260>
- [arXiv PDF](https://arxiv.org/pdf/2506.13260)：已通读方法部分（实验与附录以略读为准）：§3.1–3.4 方法

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2506.13260)
