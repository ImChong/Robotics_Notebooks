---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2410.17491"
venue: "arXiv 2024"
summary: "End-to-end generalizable navigation via world modeling."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2410_17491_x-mobility-end-to-end-generalizable-navi.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# X-MOBILITY

**X-MOBILITY: End-To-End Generalizable Navigation via World Modeling** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 359/571** 篇，分组 **813 Navigation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

End-to-end generalizable navigation via world modeling.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- End-to-end generalizable navigation via world modeling.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TB
    subgraph OBS["观测编码 (§II-C.2)"]
        I["前视 RGB 图像"] --> DN["DINOv2 编码器<br/>class token + 平均池化 patch token → u_t"]
        V["机器人线速度"] --> FC["全连接层 → m_t"]
        DN --> OE["观测嵌入 o_t = [u_t, m_t]"]
        FC --> OE
    end
    subgraph WM["世界模型：学习到的状态空间"]
        SE["状态估计网络<br/>GRU 历史 h_t + Normal 分布采样信念状态 s_t"]
        Z["潜状态 z_t = [h_t-1, s_t]"]
        SP["状态预测网络<br/>(无观测嵌入，用 h_t-1 与 a_t-1 预测 s_t)"]
        DEC["多任务解码头<br/>RGB 重建 (LDM) + 语义分割 (StyleGAN 式)"]
        SE --> Z
        Z --> DEC
        SE -.->|"KL 散度对齐"| SP
    end
    subgraph POL["动作策略网络 (§II-E)"]
        R["全局路线 → 机器人坐标系 → 截取局部路线"] --> VN["VectorNet 路线编码 g_t"]
        SA["Self-Attention 融合 z_t 与 g_t"]
        MLP["MLP 解码：动作指令 a_t (6 维速度)<br/>+ 可选 5 点导航路径"]
        VN --> SA --> MLP
    end
    OE --> SE
    ACT["动作 a_t-1：来自策略或数据集"] --> SE
    Z --> SA
    T["教师策略动作"] -->|"L1 模仿损失"| MLP
    MLP -->|"策略动作"| ACT
```

图依据：论文 §II-B～F（Fig. 2 模型架构 (a)–(e)；§II-F 多阶段训练：第一阶段关闭策略网络用离线数据训练世界模型，第二阶段联合训练策略）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 359/571 |
| 分组 | 813 Navigation |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2410.17491> |

## 核心机制（归纳）

### 策展导读要点

End-to-end generalizable navigation via world modeling.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2410.17491) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **813 Navigation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「X-MOBILITY」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **813 Navigation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2410_17491_x-mobility-end-to-end-generalizable-navi.md`](../../sources/papers/sun_awesome_wm_2410_17491_x-mobility-end-to-end-generalizable-navi.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2410.17491>
- [arXiv PDF](https://arxiv.org/pdf/2410.17491)：已通读方法部分（实验与附录以略读为准）：§II-A～F。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2410.17491)
