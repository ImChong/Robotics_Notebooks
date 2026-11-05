---
type: entity
tags: [paper, curated-index, awesome-real2sim2real, sun254667-r2s2r]
status: complete
updated: 2026-10-06
arxiv: "2409.20291"
venue: "arXiv 2024"
summary: "A novel real-to-sim-to-real framework incorporating 3D Gaussian Splatting into conventional RL simulation pipelines, enabling zero-shot sim-to-real transfer for vision-based deep reinforcement learning"
related:
  - ../entities/awesome-real2sim2real.md
  - ../overview/sun-awesome-r2s2r-technology-map.md
  - ../methods/reinforcement-learning.md
  - ../methods/crisp-real2sim.md
  - ../tasks/locomotion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/sun_awesome_r2s2r_2409_20291_rl-gsbridge-3d-gaussian-splatting-based.md
  - ../../sources/papers/sun_awesome_r2s2r_catalog.md
  - ../../sources/repos/awesome-real2sim2real.md
---

# RL-GSBridge

**RL-GSBridge: 3D Gaussian Splatting Based Real2Sim2Real Method for Robotic Manipulation Learning** 收录于 [Awesome-Real2Sim2Real](https://github.com/sun254667/Awesome-Real2Sim2Real) **第 037/063** 篇，分组 **41 Gaussian Splatting-based Frameworks**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

A novel real-to-sim-to-real framework incorporating 3D Gaussian Splatting into conventional RL simulation pipelines, enabling zero-shot sim-to-real transfer for vision-based deep reinforcement learning

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移到真机 |
| Real2Sim | Real to Simulation | 真机数据重建/校准仿真 |
| R2S2R | Real2Sim2Real | 真机→仿真→真机闭环 |
| DR | Domain Randomization | 域随机化 |

## 为什么重要

- A novel real-to-sim-to-real framework incorporating 3D Gaussian Splatting into conventional RL simulation pipelines, enabling zero-shot sim-to-real transfer for vision-based deep reinforcement learning
- 想横向对照同一分组的其他工作，可以从 [Awesome-Real2Sim2Real 技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome-Real2Sim2Real](../entities/awesome-real2sim2real.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph R2S["Real2Sim：构建带 GS 渲染的仿真器"]
        A["手机/单目相机拍摄 1-2 分钟视频，选约 200 关键帧"] --> B["COLMAP 估计相机内外参；SAM-track 分割目标"]
        B --> C["openMVS 重建网格模型 Mk（几何）"]
        B --> D["软网格绑定 GS：沿法向加可学习权重 αn ∈ [-1,1]（纹理）"]
        C --> D
        D --> E["GS 模型 Gk + 网格 Mk"]
        E --> F["RANSAC 平面回归 + 人工对齐，设定 GS 初始位姿"]
    end
    subgraph S2R["Sim2Real：仿真中训练并零样本迁移"]
        G["PyBullet 物理仿真器：读取各物体位姿 Tk、四元数 qk"] --> H["物理动力学 GS 编辑：μ = Tk·μ，r = qk × r"]
        F --> H
        H --> I["拼接各物体高斯得到整景 Gscene，光栅化渲染"]
        I --> J["仿真第一人称（eye-in-hand）图像"]
        J --> K["Actor-Critic（Conv + MLP）SACwB：基线控制器以概率 λ 引导"]
        K -->|"动作"| G
        K --> L["零样本迁移至真实机械臂（RealSense 第一人称 RGB）"]
    end
```

图依据：论文 §III-A / §III-B、Fig.1–3（箭头 K→G 的动作回路为按 Fig.3 与仿真训练设定的推断）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 037/063 |
| 分组 | 41 Gaussian Splatting-based Frameworks |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2409.20291> |

## 核心机制（归纳）

### 策展导读要点

A novel real-to-sim-to-real framework incorporating 3D Gaussian Splatting into conventional RL simulation pipelines, enabling zero-shot sim-to-real transfer for vision-based deep reinforcement learning

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2409.20291) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 的 **41 Gaussian Splatting-based Frameworks** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「RL-GSBridge」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **41 Gaussian Splatting-based Frameworks**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome-Real2Sim2Real](../entities/awesome-real2sim2real.md)
- 技术地图：[Awesome-Real2Sim2Real 技术地图](../overview/sun-awesome-r2s2r-technology-map.md)
- 方法/任务：[reinforcement-learning.md](../methods/reinforcement-learning.md)、[locomotion.md](../tasks/locomotion.md)
- 渲染 fidelity：[RAGA](./paper-raga-real-time-ray-traced-gaussian-shadow-casting.md) — 3DGS 场景中 avatar 实时光线追踪阴影（ECCV 2026）

## 参考来源

- [`sources/papers/sun_awesome_r2s2r_2409_20291_rl-gsbridge-3d-gaussian-splatting-based.md`](../../sources/papers/sun_awesome_r2s2r_2409_20291_rl-gsbridge-3d-gaussian-splatting-based.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_r2s2r_catalog.md`](../../sources/papers/sun_awesome_r2s2r_catalog.md) — 列表总表
- [`sources/repos/awesome-real2sim2real.md`](../../sources/repos/awesome-real2sim2real.md)
- 论文：<https://arxiv.org/abs/2409.20291>
- [arXiv PDF](https://arxiv.org/pdf/2409.20291)：已通读方法部分（实验与附录以略读为准）：§I 引言、§II 相关工作、§III 方法（Real2Sim 与 Sim2Real）

## 推荐继续阅读

- [Awesome-Real2Sim2Real 仓库](https://github.com/sun254667/Awesome-Real2Sim2Real)
- [原文](https://arxiv.org/abs/2409.20291)
