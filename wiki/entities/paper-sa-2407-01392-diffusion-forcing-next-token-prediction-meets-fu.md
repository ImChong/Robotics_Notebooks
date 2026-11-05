---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2407.01392"
venue: "arXiv 2024"
summary: "Unifies next-token prediction and full-sequence diffusion."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2407_01392_diffusion-forcing-next-token-prediction.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Diffusion Forcing

**Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 075/571** 篇，分组 **42 Visual / Video World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Unifies next-token prediction and full-sequence diffusion.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Unifies next-token prediction and full-sequence diffusion.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart LR
    subgraph TR["训练 Algorithm 1"]
        A["轨迹 x1..xT"] --> B["每个 token 独立采样噪声级 kt"]
        B --> C["ForwardDiffuse 得到带噪 token"]
        C --> D["因果 RNN 更新潜状态 zt<br/>输入 zt-1 / 带噪 xt / kt"]
        D --> E["预测噪声 ε<br/>MSE 损失 反传更新 θ"]
    end
    subgraph SA["采样 Algorithm 2"]
        F["token 初始化为白噪声<br/>噪声调度矩阵 K 的行 m 列 t"] --> G["按行逐步去噪<br/>每行从左到右 到 Km,t 指定噪声级"]
        G --> H["可选 AddGuidance<br/>代价函数梯度引导"]
        H --> I["最后一行 m=0<br/>得到干净序列"]
    end
    subgraph DM["序贯决策 §3.4"]
        J["token xt = at, rt, ot+1"] --> K["过去 token 汇总为潜状态 zt-1"]
        K --> L["采样未来 H 步计划<br/>含动作 奖励 观测"]
        L --> M["执行首个动作 得到 rt 与 ot+1"]
        M --> N["后验更新 zt 噪声级 k=0"]
        N --> K
    end
    E -.训练好的模型.-> G
    I -.-> L
```

图依据：论文 §3.1–§3.4 / Fig.2、Algorithm 1、Algorithm 2

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 075/571 |
| 分组 | 42 Visual / Video World Models |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2407.01392> |
| 项目页 | <https://boyuan.space/diffusion-forcing> |

## 核心机制（归纳）

### 策展导读要点

Unifies next-token prediction and full-sequence diffusion.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2407.01392) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **42 Visual / Video World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Diffusion Forcing」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2407_01392_diffusion-forcing-next-token-prediction.md`](../../sources/papers/sun_awesome_wm_2407_01392_diffusion-forcing-next-token-prediction.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2407.01392>
- [arXiv PDF](https://arxiv.org/pdf/2407.01392)：已通读方法部分（实验与附录以略读为准）：§2 预备知识、§3.1 将加噪视为部分掩码、§3.2 训练与采样、§3.3 新能力、§3.4 序贯决策；§4 实验与附录以略读/未读为准


## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2407.01392)
