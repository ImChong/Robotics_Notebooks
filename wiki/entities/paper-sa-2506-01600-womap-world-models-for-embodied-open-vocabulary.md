---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2506.01600"
venue: "arXiv 2025"
summary: "World models for embodied open-vocabulary object localization."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2506_01600_womap-world-models-for-embodied-open-voc.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# WoMAP

**WoMAP: World Models For Embodied Open-Vocabulary Object Localization** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 342/571** 篇，分组 **812 Manipulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

World models for embodied open-vocabulary object localization.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- World models for embodied open-vocabulary object localization.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph DATA["数据生成：real-to-sim-to-real"]
        A["少量真实场景视频"] --> B["训练 Gaussian Splat，蒸馏 CLIP 语义得到语义 Gaussian Splat（目标位置与尺寸）"]
        B --> C["渲染任意位姿观测，得 (图像 o, 奖励 r, 位姿 P) 数据集 D"]
        C --> D["GroundingDINO 检测置信度 × 框大小 = 稠密奖励标签"]
    end
    subgraph WM["世界模型训练"]
        E["DINOv2 观测编码器 → 隐状态 z_t"] --> F["ViT 动态预测器 q_ψ(z_t+1 | z_t, a_t)，KL 损失"]
        E --> G["奖励预测器 v_ϕ(r | z, e_g)，蒸馏检测奖励（无需重建损失）"]
    end
    D --> E
    subgraph PLAN["推理规划"]
        H["语言目标 l → 语言嵌入 e_g"] --> I["VLM 链式思维，多选题给出 3 个高层动作提案"]
        I --> J["用世界模型 + MPC 优化各提案：最大化预测奖励并加入动作平滑项"]
        J --> K["选预测奖励最高的动作序列执行"]
    end
    F --> J
    G --> J
```

图依据：论文 §3.2 Scalable Data Generation / §3.3 World Models（Fig. 4）/ §3.4 Planning with WoMAP / Fig. 2–3

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 342/571 |
| 分组 | 812 Manipulation |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2506.01600> |
| 项目页 | <https://robot-womap.github.io> |

## 核心机制（归纳）

### 策展导读要点

World models for embodied open-vocabulary object localization.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.01600) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **812 Manipulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「WoMAP」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **812 Manipulation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2506_01600_womap-world-models-for-embodied-open-voc.md`](../../sources/papers/sun_awesome_wm_2506_01600_womap-world-models-for-embodied-open-voc.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2506.01600>
- [arXiv PDF](https://arxiv.org/pdf/2506.01600)：已通读方法部分（实验与附录以略读为准）：摘要、§3.1–3.4 方法；§4 实验仅略读，附录未读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2506.01600)
