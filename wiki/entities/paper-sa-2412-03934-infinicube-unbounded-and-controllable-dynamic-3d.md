---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2412.03934"
venue: "arXiv 2024"
summary: "Unbounded and controllable dynamic 3D driving scene generation."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2412_03934_infinicube-unbounded-and-controllable-dy.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# InfiniCube

**InfiniCube: Unbounded and Controllable Dynamic 3D Driving Scene Generation with World-Guided Video Models** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 447/571** 篇，分组 **82 Autonomous Driving**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Unbounded and controllable dynamic 3D driving scene generation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Unbounded and controllable dynamic 3D driving scene generation.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    IN["输入: HD 地图 + 车辆 3D 框 + 文本提示 + 车辆轨迹"] --> S1
    subgraph S1["§4.1 无界体素世界生成"]
        CV["条件体素: 地图折线 (道路边缘/车道线) + 路面 + 车辆框 (含 sin/cos 朝向)"]
        CV --> LDM["稀疏体素 LDM (基于 XCube): 3D U-Net 去噪 -> 语义体素块"]
        LDM --> OUTP["Repaint 式 outpainting: 与已有块重叠区固定, 逐块扩展"]
    end
    OUTP --> VW["大范围语义体素世界"]
    VW --> BUF["沿轨迹渲染引导缓冲 (fVDB): 语义缓冲 + 坐标缓冲"]
    IN --> FLUX["文本 + 语义缓冲 -> FLUX ControlNet 生成首帧"]
    FLUX --> VID
    BUF --> VID["§4.2 世界引导视频生成: SVD-XT 条件为首帧 + 引导缓冲, 复用末帧潜变量自回归续生成长视频"]
    VID --> REC
    VW --> REC
    subgraph REC["§4.3 前馈动态 3DGS 重建"]
        VB["体素分支: 静态背景, 图像特征反投影到体素 + 3D 稀疏卷积 U-Net -> 每体素 Gaussians"]
        PB["像素分支: 2D UNet 逐像素 Gaussians, 输入含掩码体素深度与 Depth Anything V2 特征, 只保留中景与动态物体"]
        SKY["隐式天空模型 (编码器 + AdaLN MLP)"]
    end
    REC --> OUT["动态 3DGS 场景 (物体按轨迹/框可控)"]
```

图依据：论文 §4 / Fig.2、Fig.4

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 447/571 |
| 分组 | 82 Autonomous Driving |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2412.03934> |
| 项目页 | <https://research.nvidia.com/labs/toronto-ai/infinicube/> |

## 核心机制（归纳）

### 策展导读要点

Unbounded and controllable dynamic 3D driving scene generation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2412.03934) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **82 Autonomous Driving** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「InfiniCube」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2412_03934_infinicube-unbounded-and-controllable-dy.md`](../../sources/papers/sun_awesome_wm_2412_03934_infinicube-unbounded-and-controllable-dy.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2412.03934>
- [arXiv PDF](https://arxiv.org/pdf/2412.03934)：已通读方法部分（实验与附录以略读为准）：§4 方法（4.1-4.3）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2412.03934)
