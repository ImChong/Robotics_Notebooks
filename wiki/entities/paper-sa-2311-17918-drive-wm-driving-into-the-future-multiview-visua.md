---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2311.17918"
venue: "arXiv 2023"
code: https://github.com/BraveGroup/Drive-WM
summary: "Multiview visual forecasting and planning with world models."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2311_17918_drive-wm-driving-into-the-future-multivi.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Drive-WM

**Drive-WM: Driving into the Future – Multiview Visual Forecasting and Planning with World Model** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 416/571** 篇，分组 **82 Autonomous Driving**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Multiview visual forecasting and planning with world models.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Multiview visual forecasting and planning with world models.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    IMG["单视角图像数据"] --> SD["预训练图像扩散模型 θ（SD 初始化）"]
    SD --> FT["冻结 θ，视频数据微调 temporal layers φ 与 multiview layers ψ"]
    COND["条件：初始上下文帧 / 布局（3D box、HD map、BEV）/ 文本 / 自车 action (Δx,Δy)"] --> ENC["统一条件接口：各条件编码为 d 维嵌入并拼接 c_t"]
    FT --> GEN["3D UNet 多视角视频生成（c_t 经逐帧 cross attention 注入）"]
    ENC --> GEN
    GEN --> FAC["因子分解生成：p(x1..K) 逐视角以邻接视角为条件，增强重叠区域一致性"]
    FAC --> VID["多视角未来视频"]
    PLAN["预训练 planner：真实多视角图像 → 采样候选轨迹"] --> ACT["候选 action 作为条件"]
    ACT --> GEN
    VID --> PER["感知模型：3D 检测器 + 在线 HDMap 预测"]
    PER --> REW["object reward × map reward（总奖励为乘积）"]
    REW --> SEL["选最大奖励轨迹，扩展 planning tree"]
    SEL -->|"下一时刻迭代"| PLAN
```

图依据：论文 §3.1–3.3（多视角视频生成）、§4.1–4.2 / Fig.5（基于世界模型的规划与图像奖励）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 416/571 |
| 分组 | 82 Autonomous Driving |
| 出处 | arXiv 2023 |
| 论文 | <https://arxiv.org/abs/2311.17918> |
| 代码/项目 | <https://github.com/BraveGroup/Drive-WM> |

## 核心机制（归纳）

### 策展导读要点

Multiview visual forecasting and planning with world models.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2311.17918) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **82 Autonomous Driving** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Drive-WM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2311_17918_drive-wm-driving-into-the-future-multivi.md`](../../sources/papers/sun_awesome_wm_2311_17918_drive-wm-driving-into-the-future-multivi.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2311.17918>
- [arXiv PDF](https://arxiv.org/pdf/2311.17918)：已通读方法部分（实验与附录以略读为准）：§3 多视角视频生成、§4 端到端规划、§5.1 实验设置

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2311.17918)
