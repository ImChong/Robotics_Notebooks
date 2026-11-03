---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2502.21142"
venue: "arXiv 2025"
summary: "Global workspace approach to world model-based RL."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2502_21142_multimodal-dreaming-a-global-workspace-a.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Multimodal Dreaming

**Multimodal Dreaming: A Global Workspace Approach to World Model-Based RL** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 262/571** 篇，分组 **73 Offline & Generalization**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Global workspace approach to world model-based RL.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Global workspace approach to world model-based RL.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph S1["阶段一：Global Workspace 预训练"]
        DS["随机或专家策略收集的观测数据集"] --> VAE["预训练并冻结的 VAE（图像、属性各一）"]
        VAE --> UV["单模态隐向量 u_v、u_attr"]
        UV --> EN["编码器 e_v、e_attr，得到 z_v、z_attr"]
        EN --> FU["加权求和（α_v + α_attr = 1）后 Tanh，得到 GW 表示 z"]
        FU --> DE["解码器 d_v、d_attr 还原单模态隐向量"]
        FU --> LS["损失：对比损失 L_cont + 广播损失 L_broad"]
    end
    subgraph S2["阶段二：World Model 与 Actor-Critic"]
        FZ["冻结 GW，固定 α_v = α_attr = 0.5，得到 z_t"] --> WM["World Model：GRU，输入 z_t 与 a_t"]
        WM --> HD["预测头（MLP）：下一步 GW 表示、奖励、终止信号"]
        HD --> WL["L_WM：MSE（表示、奖励）+ BCE（终止）"]
        WM --> AC["Actor-Critic：以 h_t 为输入，输出动作与状态价值"]
        AC --> DR["Dreaming：仅首步用真实观测，之后由 WM 自回归模拟"]
        DR --> ACL["仅用模拟轨迹计算 AC 损失；梯度不传入 WM"]
    end
    S1 --> S2
```

图依据：论文 §II-A（Global Workspace，Fig. 1）、§II-B/C（World Model 与 Actor-Critic，Fig. 3）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 262/571 |
| 分组 | 73 Offline & Generalization |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2502.21142> |

## 核心机制（归纳）

### 策展导读要点

Global workspace approach to world model-based RL.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2502.21142) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **73 Offline & Generalization** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Multimodal Dreaming」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **73 Offline & Generalization**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2502_21142_multimodal-dreaming-a-global-workspace-a.md`](../../sources/papers/sun_awesome_wm_2502_21142_multimodal-dreaming-a-global-workspace-a.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2502.21142>
- [arXiv PDF](https://arxiv.org/pdf/2502.21142)：已通读方法部分（实验与附录以略读为准）：§II 模型（A–C，含 Fig. 1/3）与 §III 环境设置；§IV 结果以略读为准

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2502.21142)
