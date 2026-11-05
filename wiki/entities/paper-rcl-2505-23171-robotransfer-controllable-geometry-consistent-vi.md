---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2505.23171"
venue: "arXiv preprint"
summary: "RoboTransfer augments robot demonstrations by changing their visual appearance while conditioning video diffusion on the demonstrated geometry. Jointly encoded camera views, metric depth, normals and separate background/"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2505_23171_robotransfer-controllable-geometry-consi.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# RoboTransfer

**RoboTransfer: Controllable Geometry-Consistent Video Diffusion for Manipulation Policy Transfer** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 119/564** 篇，分组 **Datasets**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

RoboTransfer augments robot demonstrations by changing their visual appearance while conditioning video diffusion on the demonstrated geometry. Jointly encoded camera views, metric depth, normals and separate background/object references support multi-view synthesis. A separately trained ACT policy benefits from the...

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- RoboTransfer augments robot demonstrations by changing their visual appearance while conditioning video diffusion on the demonstrated geometry. Jointly encoded camera views, metric depth, normals and separate background/object references support multi-view synthesis. A separately trained ACT policy benefits from the...
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph DATA["数据构建 §3.3: 真实演示拆成三元组"]
        R["真实机器人演示<br/>头部相机 + 双腕相机"] --> G1["几何条件<br/>深度估计器 + RGB-D 尺度对齐(鲁棒最小二乘)得 metric depth<br/>单目法线估计得 surface normal"]
        R --> K["多视角 RGB 关键帧"]
        K --> V["VLM 描述器生成场景/物体描述<br/>Grounding-SAM 分割物体"]
        V --> BG["物体修补(inpaint)得无物体背景参考图 C_b"]
        V --> OB["物体 mask 经 CLIP 得语义嵌入 C_o"]
    end
    subgraph GEN["RoboTransfer 生成 §3.2: 多视角视频扩散"]
        MV["N 路同步视频沿宽度拼接<br/>VAE 联合编码, 无需改动扩散骨干"]
        G1 --> GE["几何 VAE(卷积)编码 depth + normal 视频<br/>与噪声潜变量按通道拼接"]
        BG --> BE["VAE 编码背景参考图<br/>与潜变量拼接, 控制全局纹理"]
        OB --> OE["CLIP 全局嵌入经 cross attention<br/>作为非结构化物体外观条件"]
        MV --> D["视频扩散去噪"]
        GE --> D
        BE --> D
        OE --> D
    end
    D --> S["几何一致、外观可控的多视角合成视频"]
    S --> P["与真实数据混合训练 ACT 策略<br/>论文实验 50/50 混合最佳(§5.2)"]
```

图依据：论文 §3.2 RoboTransfer Framework、§3.3 Dataset Construction / Fig.2–3、§5 Real Robot Experiments。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 119/564 |
| 分组 | Datasets |
| 出处 | arXiv preprint |
| 论文 | <https://arxiv.org/abs/2505.23171> |
| 子类 / 象限 | 合成数据与数据生成 · 不适用 |

## 核心机制（归纳）

### 策展导读要点

RoboTransfer augments robot demonstrations by changing their visual appearance while conditioning video diffusion on the demonstrated geometry. Jointly encoded camera views, metric depth, normals and separate background/object references support multi-view synthesis. A separately trained ACT policy benefits from the...

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2505.23171) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **Datasets** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「RoboTransfer」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Datasets**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Contribution 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/rcl_awesome_wam_2505_23171_robotransfer-controllable-geometry-consi.md`](../../sources/papers/rcl_awesome_wam_2505_23171_robotransfer-controllable-geometry-consi.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2505.23171>
- [arXiv PDF](https://arxiv.org/pdf/2505.23171)：已通读方法部分（实验与附录以略读为准）：§3.2 框架、§3.3 数据构建、§5 真机实验设定与合成数据比例（略读）

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2505.23171)
