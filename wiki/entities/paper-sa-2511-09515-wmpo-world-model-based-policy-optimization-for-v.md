---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-11-04
arxiv: "2511.09515"
venue: "arXiv 2025"
summary: "World model-based policy optimization for VLAs."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
  - ../overview/embodied-algorithm-stack-2026-vla-diffusion-rl-world-model.md
sources:
  - ../../sources/papers/sun_awesome_wm_2511_09515_wmpo-world-model-based-policy-optimizati.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
  - ../../sources/blogs/wechat_xinjiqishijue_embodied_algorithm_stack_2026-11-04.md
---

# WMPO

**WMPO: World Model-based Policy Optimization for Vision-Language-Action Models** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 231/571** 篇，分组 **63 World Models for VLA Training & Evaluation Combining Paradigms**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

World model-based policy optimization for VLAs.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- World model-based policy optimization for VLAs.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph WMTRAIN["世界模型与奖励模型准备"]
        OXE["Open X-Embodiment 预训练<br/>(OpenSora 视频扩散骨干，换 SDXL 2D VAE；<br/>noisy-frame conditioning + 帧级动作控制 AdaLN)"] --> FT["用策略自身的真实 rollout 微调<br/>(Policy Behavior Alignment，覆盖失败模式)"]
        FT --> WM["世界模型 p_φ"]
        REALT["真实轨迹：终端片段为正样本，<br/>其余片段/失败轨迹为负样本"] --> RM["奖励模型 R_ψ：VideoMAE 编码器 + 线性头<br/>滑窗判定，任一片段超阈值即成功 (二值)"]
    end
    subgraph LOOP["WMPO 在线策略优化循环（Fig. 2）"]
        S0["真实环境初始帧 I_0:c + 语言指令 g"] --> GEN["(1) 想象轨迹生成：<br/>VLA 策略 π_θold 输出动作块 → 世界模型生成后续 K 帧，交替至最大长度 N"]
        WM --> GEN
        GEN --> SAMP["(2) 轨迹采样：每个初始帧采样一组 G 条轨迹"]
        SAMP --> RM
        RM --> FILT["二值奖励；若全成功或全失败则丢弃该组<br/>(Dynamic Sampling) 并补采样"]
        FILT --> UPD["(3) 策略更新：GRPO 组内归一化优势<br/>(去除 KL 项，clip 上下界不同)"]
        UPD -->|"更新 π_θ"| GEN
    end
```

图依据：论文 §3.2–3.4 / Fig. 2、式 (2)–(5)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 231/571 |
| 分组 | 63 World Models for VLA Training & Evaluation Combining Paradigms |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2511.09515> |
| 项目页 | <https://wm-po.github.io> |

## 核心机制（归纳）

### 策展导读要点

World model-based policy optimization for VLAs.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2511.09515) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **63 World Models for VLA Training & Evaluation Combining Paradigms** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「WMPO」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **63 World Models for VLA Training & Evaluation Combining Paradigms**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2511_09515_wmpo-world-model-based-policy-optimizati.md`](../../sources/papers/sun_awesome_wm_2511_09515_wmpo-world-model-based-policy-optimizati.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2511.09515>
- [arXiv PDF](https://arxiv.org/pdf/2511.09515)：已通读方法部分（实验与附录以略读为准）：§1 引言、§3.1–3.4 方法、§4 实验设置（略读）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2511.09515)
