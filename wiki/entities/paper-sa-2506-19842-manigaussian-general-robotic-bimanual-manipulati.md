---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2506.19842"
venue: "arXiv 2025"
code: https://github.com/April-Yz/ManiGaussian_Bimanual
summary: "General robotic bimanual manipulation with hierarchical Gaussian world model."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2506_19842_manigaussian-general-robotic-bimanual-ma.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# ManiGaussian++

**ManiGaussian++: General Robotic Bimanual Manipulation with Hierarchical Gaussian World Model** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 308/571** 篇，分组 **812 Manipulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

General robotic bimanual manipulation with hierarchical Gaussian world model.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- General robotic bimanual manipulation with hierarchical Gaussian world model.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    O["观测 RGB-D 图像 + 本体感知 + 语言指令"] --> VX["按标定相机参数转为体素\n稀疏卷积网络编码 → 视觉表示 v"]
    VX --> GR["Gaussian 回归器（前馈）\n重建任务导向 Gaussian 辐射场\n每个高斯带实例 logit"]
    VLM["预训练 VLM（GroundedSAM）\n按指令关键词生成实例真值"] -.监督.-> GR
    GR --> LD["Leader 形变模型\n由稳定臂动作预测高斯偏移"]
    LD --> FL["Follower 形变模型\n结合执行臂与稳定臂动作预测后果"]
    FL --> RD["Gaussian 渲染器\n输出未来多视角图像与实例图"]
    RD --> LP["光度监督 L_Pred（与真实未来图像对齐）"]
    GR --> LR["当前场景重建 L_Recon + 实例分割 L_Task"]
    VX --> PIO["PerceiverIO 多模态 Transformer\n结合增强后的体素表示与语言"]
    PIO --> ACT["左右臂离散动作分类（交叉熵 L_BC）"]
    LP --> OBJ["总目标 L = L_BC + λ_Recon·L_Recon + λ_Task·L_Task + λ_Pred·L_Pred"]
    LR --> OBJ
    ACT --> OBJ
```

图依据：论文 §III-B–E / Fig.2、式 (5)–(10)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 308/571 |
| 分组 | 812 Manipulation |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2506.19842> |
| 代码/项目 | <https://github.com/April-Yz/ManiGaussian_Bimanual> |

## 核心机制（归纳）

### 策展导读要点

General robotic bimanual manipulation with hierarchical Gaussian world model.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.19842) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **812 Manipulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「ManiGaussian++」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2506_19842_manigaussian-general-robotic-bimanual-ma.md`](../../sources/papers/sun_awesome_wm_2506_19842_manigaussian-general-robotic-bimanual-ma.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2506.19842>
- [arXiv PDF](https://arxiv.org/pdf/2506.19842)：已通读方法部分（实验与附录以略读为准）：§III 方法 A–E（含 Fig.2、式 1–10）、§IV-A 实验设置

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2506.19842)
