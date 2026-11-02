---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2606.16993"
venue: "arXiv 2026"
code: https://github.com/AMAP-ML/DreamX-World
summary: "General-purpose interactive world model."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2606_16993_dreamx-world-1-0-a-general-purpose-inter.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# DreamX-World 1.0

**DreamX-World 1.0: A General-Purpose Interactive World Model** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 144/571** 篇，分组 **47 General Foundation World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

General-purpose interactive world model.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- General-purpose interactive world model.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    D["数据系统：UE 渲染 + 游戏录像 + 真实视频<br/>基础过滤 → 相机位姿几何清洗 → 字幕与属性标注"] --> S0["初始化：Wan2.2-TI2V 双向视频生成器"]
    S0 --> S1["3.1 相机感知训练<br/>E-PRoPE：空间降采样 token 上算投影相机注意力<br/>冻结 DiT 骨干，仅训 PRoPE 模块，输出上采样后加回"]
    S1 --> S2["3.2 记忆条件场景持久化<br/>几何检索（相机位姿 + 视角重叠）选记忆帧<br/>与近期历史、目标帧打包 [记忆 | 历史 | 目标]<br/>残差回收：仅对条件 token 注入误差"]
    S2 --> S3["3.3 Event Instruction Tuning<br/>结构化事件指令渲染成文本提示，微调整个 DiT"]
    S3 --> S4["3.4 自回归蒸馏<br/>causal forcing → 长序列 rollout → DMD 蒸馏<br/>双向 E-PRoPE 教师 → 少步自回归学生"]
    S4 --> S5["3.5 强化学习后训练（DiffusionNFT 软更新）<br/>相机控制奖励 + 视频质量奖励融合，KL 约束"]
    S5 --> INF["§4 流式推理：逐 chunk 从噪声去噪<br/>相机控制 + 滚动 KV cache，chunk 相对位姿<br/>混合精度 / 残差复用 / 剪枝 VAE / 异步流水线并行"]
```

图依据：论文 §2–§4 / Fig.2、Fig.5–Fig.9（训练阶段顺序依 §3 与 Fig.2 的 Progressive Training 描述）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 144/571 |
| 分组 | 47 General Foundation World Models |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.16993> |
| 代码/项目 | <https://github.com/AMAP-ML/DreamX-World> |

## 核心机制（归纳）

### 策展导读要点

General-purpose interactive world model.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.16993) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **47 General Foundation World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「DreamX-World 1.0」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **47 General Foundation World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2606_16993_dreamx-world-1-0-a-general-purpose-inter.md`](../../sources/papers/sun_awesome_wm_2606_16993_dreamx-world-1-0-a-general-purpose-inter.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2606.16993>
- [arXiv PDF](https://arxiv.org/pdf/2606.16993)：已通读方法部分（实验与附录以略读为准）：§1 引言、§2.1–2.2 数据、§3 渐进训练（3.1–3.5）、§4 推理加速；§2.3 事件数据与 §5 评测略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2606.16993)
