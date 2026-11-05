---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2410.18084"
venue: "arXiv 2024"
code: https://github.com/3DTopia/DynamicCity
summary: "Large-scale 4D occupancy generation from dynamic scenes."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2410_18084_dynamiccity-large-scale-4d-occupancy-gen.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# DynamicCity

**DynamicCity: Large-Scale 4D Occupancy Generation from Dynamic Scenes** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 426/571** 篇，分组 **82 Autonomous Driving**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Large-scale 4D occupancy generation from dynamic scenes.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Large-scale 4D occupancy generation from dynamic scenes.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TB
    subgraph VAE["(a) VAE：编码与解码 HexPlane (§4.1)"]
        Q["4D 占据序列 Q (T×X×Y×Z×C)"] --> F["共享 3D 卷积特征提取 f_θ<br/>得到特征体序列 X_txyz"]
        F --> PM["Projection Module (transformer 投影网络 h)<br/>先得 XYZ 特征体，再沿各轴压缩"]
        PM --> HP["HexPlane H<br/>空间三平面 Pxy, Pxz, Pyz<br/>+ 时空三平面 Ptx, Pty, Ptz"]
        HP --> ESS["Expansion & Squeeze Strategy<br/>平面扩展 + Hadamard 积"]
        ESS --> G["卷积上采样 g_φ<br/>(拼接 4D 位置编码)"]
        G --> QR["重建语义占据 Q'"]
        QR -.->|"L_CE + L_Lovász + L_KL"| Q
    end
    subgraph DIT["(b) HexPlane DiT 生成 (§4.2)"]
        PRO["Padded Rollout<br/>六平面拼成方形特征图 (角落补零)"] --> PT["Patchify + 频率位置编码<br/>(补零 token 不参与扩散)"]
        PT --> DT["DiT 去噪生成新 HexPlane<br/>adaLN-Zero 注入条件 + 图像条件 cross-attention"]
    end
    COND["可选条件：HexPlane / Layout / Command / Trajectory"] --> DT
    HP -->|"训练后的 VAE 潜变量"| PRO
    DT --> NH["生成的 HexPlane"]
    NH -->|"VAE 解码器"| ESS
    DT -.->|"采样时遮罩 HexPlane 区域 → Inpaint"| DT
```

图依据：论文 §4.1 VAE for 4D Occupancy（Fig. 3）、§4.2 Diffusion Transformer for HexPlane（Fig. 4 Padded Rollout、Fig. 5 条件注入）、§4.3 下游应用（Fig. 2 总流程）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 426/571 |
| 分组 | 82 Autonomous Driving |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2410.18084> |
| 代码/项目 | <https://github.com/3DTopia/DynamicCity> |

## 核心机制（归纳）

### 策展导读要点

Large-scale 4D occupancy generation from dynamic scenes.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2410.18084) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **82 Autonomous Driving** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「DynamicCity」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2410_18084_dynamiccity-large-scale-4d-occupancy-gen.md`](../../sources/papers/sun_awesome_wm_2410_18084_dynamiccity-large-scale-4d-occupancy-gen.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2410.18084>
- [arXiv PDF](https://arxiv.org/pdf/2410.18084)：已通读方法部分（实验与附录以略读为准）：§3 预备知识（HexPlane/DiT）、§4.1～4.3。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2410.18084)
