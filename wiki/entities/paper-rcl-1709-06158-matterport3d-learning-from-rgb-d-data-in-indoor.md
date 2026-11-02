---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "1709.06158"
venue: "3DV 2017"
code: https://github.com/niessner/Matterport
summary: "RCL Awesome WAM 清单收录（Datasets）；细节以原文 PDF / 项目页为准。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_1709_06158_matterport3d-learning-from-rgb-d-data-in.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# Matterport3D

**Matterport3D: Learning from RGB-D Data in Indoor Environments** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 113/564** 篇，分组 **Datasets**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

RCL Awesome WAM 清单收录（Datasets）；细节以原文 PDF / 项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- RCL Awesome WAM 清单收录（Datasets）；细节以原文 PDF / 项目页为准。
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["3.1 数据采集: 三脚架 Matterport 相机 (3 个 color + 3 个 depth)<br/>绕重力方向转 6 个朝向, 每个 panorama 得 18 张 RGB-D<br/>沿可步行平面约每 2.5m 一个 panorama; 用户在 iPad app 标记 windows 与 mirrors"] --> B["Matterport 处理<br/>1) panorama 拼接成 skybox<br/>2) 全局 bundle adjustment 估计每张图 6 DoF 位姿<br/>3) 重建单个 textured mesh"]
    B --> C["原始数据: 90 栋建筑, 194,400 张 RGB-D, 10,800 个 panorama"]
    C --> D["3.2 语义标注<br/>先由标注者画 floor plan, 抽取 room-like regions (含类别)<br/>再对每个 region 抽 mesh (screened Poisson)<br/>用 ScanNet 众包界面 paint 三角形标注 object instances"]
    D --> E["AMT 初始标签 → 十名专家标注者补全/修正/验证<br/>自由文本标签规整为 40 个 object categories (WordNet synsets)"]
    E --> F["发布: RGB-D 图像 + 位姿 + skybox + textured mesh + region 与 object 语义标注<br/>划分 61 场景 train / 11 validation / 18 test"]
    F --> G["Sec. 4 五个任务及 baseline<br/>Keypoint Matching / View Overlap Prediction / Surface Normal Estimation / Region-Type Classification / Semantic Voxel Labeling"]
```

图依据：数据采集与标注流程图，论文 §3.1 Data Acquisition Process、§3.2 Semantic Annotation（Fig. 2–4）、§4 开头的 train/val/test 划分与五个任务小节。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 113/564 |
| 分组 | Datasets |
| 出处 | 3DV 2017 |
| 论文 | <https://ieeexplore.ieee.org/document/8374622/> |
| 代码/项目 | <https://github.com/niessner/Matterport> |
| 子类 / 象限 | 三维场景与RGB-D数据 · 多传感器与空间标注 · 不适用 |

## 核心机制（归纳）

### 策展导读要点

RCL Awesome WAM 清单收录（Datasets）；细节以原文 PDF / 项目页为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://ieeexplore.ieee.org/document/8374622/) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **Datasets** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Matterport3D」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/rcl_awesome_wam_1709_06158_matterport3d-learning-from-rgb-d-data-in.md`](../../sources/papers/rcl_awesome_wam_1709_06158_matterport3d-learning-from-rgb-d-data-in.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://ieeexplore.ieee.org/document/8374622/>
- [arXiv PDF](https://arxiv.org/pdf/1709.06158)：已通读 §1 Introduction、§3 Matterport3D Dataset（§3.1–3.3）、§4 Learning from the Data 的任务设定；各任务实验数字与补充材料以略读为准

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://ieeexplore.ieee.org/document/8374622/)
