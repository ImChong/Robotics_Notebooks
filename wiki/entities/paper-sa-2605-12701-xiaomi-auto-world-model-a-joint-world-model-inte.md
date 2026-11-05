---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2605.18137"
venue: "arXiv 2026"
summary: "Xiaomi's joint world model integrating reconstruction and generation."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2605_12701_xiaomi-auto-world-model-a-joint-world-mo.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Xiaomi Auto World Model

**Xiaomi Auto World Model: A Joint World Model Integrating Reconstruction and Generation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 498/571** 篇，分组 **82 Autonomous Driving**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Xiaomi's joint world model integrating reconstruction and generation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Xiaomi's joint world model integrating reconstruction and generation.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    IMG["多相机多时刻图像 I_t^c"] --> WR
    subgraph WR["WorldRec 前馈重建 (§3.1)"]
        R1["共享权重骨干提取多尺度特征"]
        R2["世界空间初始化 N 个稀疏 3D query<br/>按标定参数投影到各视图并双线性采样"]
        R3["可见性感知加权聚合 (跨视角 + 相邻帧对齐) → 场景 token q_i"]
        R4["MLP 解码 3D Gaussian 属性<br/>(位置偏移/颜色/不透明度/尺度/旋转)"]
        R5["可微高斯光栅化渲染, 像素 + 感知损失监督"]
        R1 --> R2 --> R3 --> R4 --> R5
    end
    subgraph WG["WorldGen 因果视频生成 (§3.2)"]
        G1["阶段1 双向 DiT 预训练 (rectified flow)"]
        G2["阶段2 因果微调: Teacher Forcing"]
        G3["ODE 蒸馏 (50 步 → 4 步)"]
        G4["DMD (以自身生成帧作历史上下文)"]
        G1 --> G2 --> G3 --> G4
    end
    subgraph JW["Joint World Model (§3.3)"]
        J1["scene fusion: 新观测 token 与缓存场景 token 做交叉注意力融合<br/>增量更新 Gaussian 场景"]
        J2["场景 token 光栅化到目标相机视角, 得 Prior Image 渲染先验"]
        J3["WorldGen 在 ego 轨迹/相机参数之外额外以渲染先验为条件<br/>补全未观测区域, 抑制几何漂移"]
        J1 --> J2 --> J3
    end
    R4 --> J1
    G4 --> J3
    J3 --> OUT["因果多视角驾驶视频"]
```

图依据：论文 §3.1–3.3 / Fig.2、Fig.3、Fig.4、Fig.5。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 498/571 |
| 分组 | 82 Autonomous Driving |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2605.18137> |

## 核心机制（归纳）

### 策展导读要点

Xiaomi's joint world model integrating reconstruction and generation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2605.18137) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **82 Autonomous Driving** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Xiaomi Auto World Model」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2605_12701_xiaomi-auto-world-model-a-joint-world-mo.md`](../../sources/papers/sun_awesome_wm_2605_12701_xiaomi-auto-world-model-a-joint-world-mo.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2605.18137>
- 编号核对（2026-11-02）：上游清单登记的 2605.12701 经 arXiv 官方记录核对为《Do Fair Models Reason Fairly? Counterfactual Explanation Consistency for Procedural Fairness in Credit Decisions》，与本页论文不符；已更正为 2605.18137《Xiaomi Auto World Model: A Joint World Model Integrating Reconstruction and Generation for Autonomous Driving》。
- [arXiv PDF](https://arxiv.org/pdf/2605.18137)：已通读方法部分（实验与附录以略读为准）：§3 Method（§3.1–3.3）、Fig.3/5 图注；图 2、图 4 为图像，仅依据正文描述

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2605.18137)
