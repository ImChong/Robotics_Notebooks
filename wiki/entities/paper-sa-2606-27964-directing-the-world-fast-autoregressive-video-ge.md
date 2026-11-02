---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2606.27964"
venue: "arXiv 2026"
summary: "Fast autoregressive video generation via compositional human-camera control."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2606_27964_directing-the-world-fast-autoregressive.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Directing the World

**Directing the World: Fast Autoregressive Video Generation with Compositional Human-Camera Control** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 076/571** 篇，分组 **42 Visual / Video World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Fast autoregressive video generation via compositional human-camera control.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Fast autoregressive video generation via compositional human-camera control.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    BB["MMPL 块式自回归骨干（Plan-then-Populate）：先预测含终止帧 x_end 的规划帧，再填充中间帧；x_end 作下一段初始条件"]
    subgraph S1["阶段一：人体运动控制（运动中心数据，低相机运动）"]
        direction TB
        SM["SMPL 序列渲染 → 空间对齐（缩放 + 包围盒重叠）→ 同一 VAE 编码得 c_smpl"] --> PJ["t-guided Dynamic Projection：按去噪时间步 t 投影 c_smpl 后注入噪声潜变量"]
        PJ --> MC["Motion-CFG：固定文本提示，以 SMPL 条件做 CFG；无条件分支用黑视频的 VAE 潜变量"]
    end
    subgraph S2["阶段二：相机轨迹控制（相机中心数据，在已习得的运动先验上）"]
        direction TB
        CT["相机轨迹（Plücker 坐标）→ 相机编码器（全局自注意）得上下文化特征 H"] --> CW["仅注入与当前生成块帧窗口 Wi 时间对齐的相机特征"]
    end
    BB --> S1
    S1 --> S2
    S2 --> OUT["组合人体运动与相机运动的可控长视频"]
    FS["Fast-Slow Memory 训练：原自注意 / 文本交叉注意为慢参数（小学习率 ηs），新增控制模块为快参数（大学习率 ηf）"] -.-> S1
    FS -.-> S2
```

图依据：论文 §3.1–3.5 / Fig. 2、§3.6 数据划分（Stage-I / Stage-II）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 076/571 |
| 分组 | 42 Visual / Video World Models |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.27964> |

## 核心机制（归纳）

### 策展导读要点

Fast autoregressive video generation via compositional human-camera control.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.27964) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **42 Visual / Video World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Directing the World」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **42 Visual / Video World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2606_27964_directing-the-world-fast-autoregressive.md`](../../sources/papers/sun_awesome_wm_2606_27964_directing-the-world-fast-autoregressive.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2606.27964>
- [arXiv PDF](https://arxiv.org/pdf/2606.27964)：已通读方法部分（实验与附录以略读为准）：§3.1 Overview、§3.2 MMPL 骨干、§3.3 人体运动控制、§3.4 相机控制、§3.5 Fast-Slow 训练、§3.6 数据构建

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2606.27964)
