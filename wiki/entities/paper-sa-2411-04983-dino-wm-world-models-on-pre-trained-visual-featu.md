---
type: entity
project_id: dino-wm-dinov2
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-11-08
arxiv: "2411.04983"
code: https://github.com/gaoyuezhou/dino_wm
venue: "arXiv 2024"
summary: "World models on pre-trained visual features enable zero-shot planning."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
original:
  - ../../sources/papers/sun_awesome_wm_2411_04983_dino-wm-world-models-on-pre-trained-visu.md
sources:
  - ../../sources/papers/sun_awesome_wm_2411_04983_dino-wm-world-models-on-pre-trained-visu.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# DINO-WM

**DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 110/571** 篇，分组 **44 Latent Space World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

World models on pre-trained visual features enable zero-shot planning.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- World models on pre-trained visual features enable zero-shot planning.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  subgraph TRAIN["离线训练：仅用轨迹数据，不需奖励"]
    A["观测 o_t"] --> B["冻结的 DINOv2<br/>编码为 patch 特征 z_t"]
    B --> C["ViT 转移模型 p_θ（因果注意力，帧级自回归）<br/>输入历史 z_(t-H:t-1)，并拼接动作 / 本体感知"]
    C --> D["预测下一帧潜在状态<br/>隐空间一致性损失 L_pred（teacher forcing）"]
    B -.-> E["可选解码器 q_θ（转置卷积）<br/>重建损失 L_rec，仅用于可视化"]
  end
  subgraph PLAN["测试期规划：隐空间 MPC"]
    F["当前观测 o_0 与目标观测 o_g"] --> G["同一 DINOv2 编码 → ẑ_0, z_g"]
    G --> H["CEM 采样动作序列<br/>用 p_θ 在隐空间展开 ẑ_T"]
    H --> I["代价 C = ‖ẑ_T − z_g‖²<br/>取代价最低的 top-K 更新采样分布并迭代"]
    I --> J["执行前若干个动作 → 取新观测，重新规划"]
    J --> F
  end
  D --> H
```

图依据：论文 §3 / Fig.2、附录 A.5.1（CEM 的 MPC 规划）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 110/571 |
| 分组 | 44 Latent Space World Models |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2411.04983> |
| 项目页 | <https://dino-wm.github.io/> |

## 核心机制（归纳）

### 策展导读要点

World models on pre-trained visual features enable zero-shot planning.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2411.04983) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **44 Latent Space World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「DINO-WM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **44 Latent Space World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 源码运行时序图

```mermaid
sequenceDiagram
    autonumber
    actor User as 用户
    participant Train as train.py
    participant Data as datasets/
    participant WM as models/
    participant Plan as plan.py
    participant CEM as planning/
    User->>Train: --config-name train.yaml env=point_maze
    Train->>Data: 加载离线视觉轨迹
    Train->>WM: 训练 DINOv2 特征预测世界模型
    User->>Plan: model_name=point_maze planner=cem
    Plan->>WM: 加载模型并预测未来特征
    Plan->>CEM: 优化动作序列与视觉目标距离
    CEM-->>Plan: 规划动作
```

图只表示 `README.md` 提供的训练/规划调用接口；未执行。


## 工程实践

官方 `gaoyuezhou/dino_wm` 提供 `train.py` 离线训练及 `plan.py` + CEM 目标图像规划；论文与仓库使用 DINOv2。


## 局限与风险

- 尚未完成主论文与附录逐段通读，不作为完整复现指南；数值比较须保持同任务、同评测协议。
- 代码可见不代表依赖、权重、训练实现全部公开，也不代表已执行。


## DINOv3 世界模型一手资料复核（2026-11-08）

DINO-WM（2024/2025）采用预训练 DINOv2 特征构建无奖励零样本视觉目标规划的世界模型，属于 DINOv3 时代的前作/基线，不应标成 DINOv3 世界模型。

官方 `gaoyuezhou/dino_wm` 提供 `train.py` 离线训练及 `plan.py` + CEM 目标图像规划；论文与仓库使用 DINOv2。

- [作者论文](https://arxiv.org/abs/2411.04983) · [原文归档](../../sources/papers/sun_awesome_wm_2411_04983_dino-wm-world-models-on-pre-trained-visu.md) · [代码核验记录](../../sources/repos/dinov3-wm-dino-wm.md)
- 原文主文及必要附录已核查；代码入口以官方仓库为准，尚未亲自运行训练和评测。

## 2026-11-08 原文与源码完整复核

**方法：** 使用 **DINOv2**（不是 DINOv3）的冻结 patch 表征，学习 action-conditioned causal Transformer 预测器，使用视觉目标嵌入差异指导 CEM 零样本规划。

**核对结果：** 在 PointMaze、Wall、PushT、Reach、Rope、Granular 任务测试；CEM 动作序列搜索开销显著，不等于单次 WM forward 时延。

**限制：** 需要覆盖足够的离线状态-动作轨迹，训练仍依赖真实动作标签；原项目只能作为 DINOv3 世界模型的历史基线。

全文版本：[arXiv:2411.04983](https://arxiv.org/abs/2411.04983)，未在本环境实际执行训练或真机评测。

## 关联页面

- 列表实体：[Awesome World Models](../entities/awesome-world-models.md)
- 技术地图：[Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_wm_2411_04983_dino-wm-world-models-on-pre-trained-visu.md`](../../sources/papers/sun_awesome_wm_2411_04983_dino-wm-world-models-on-pre-trained-visu.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2411.04983>
- [arXiv PDF](https://arxiv.org/pdf/2411.04983)：已通读方法部分（实验与附录以略读为准）：§3 DINO World Models（3.1 观测/转移/解码器、3.2 视觉规划）、附录 A.5 规划优化；§4 实验略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2411.04983)
