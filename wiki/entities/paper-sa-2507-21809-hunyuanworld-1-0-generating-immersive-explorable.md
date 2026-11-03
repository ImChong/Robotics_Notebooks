---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2507.21809"
venue: "arXiv 2025"
code: https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
summary: "Generates immersive, explorable, and interactive 3D worlds from words or pixels."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2507_21809_hunyuanworld-1-0-generating-immersive-ex.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# HunyuanWorld 1.0

**HunyuanWorld 1.0: Generating Immersive, Explorable, and Interactive 3D Worlds from Words or Pixels** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 174/571** 篇，分组 **52 Game & Open-World Exploration Application**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Generates immersive, explorable, and interactive 3D worlds from words or pixels.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Generates immersive, explorable, and interactive 3D worlds from words or pixels.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TB
    subgraph IN["输入条件（§2.1）"]
        direction LR
        TXT["文本：LLM 翻译并增强提示词"]
        IMGI["单张图像：ERP 投影到全景空间<br/>相机内参由 MoGe / UniK3D 估计"]
        SAP["Scene-aware prompt：VLM 提取显著物体作负提示<br/>并补全 360° 场景描述"]
        IMGI --> SAP
    end
    IN --> PDIT["Panorama-DiT（基于 DiT）生成全景图<br/>elevation-aware 增强 + circular denoising"]
    PDIT --> PANO["全景图 = 世界代理 world proxy"]
    PANO --> AG["Agentic world layering（§2.3）<br/>VLM 实例识别"]
    AG --> DET["圆形 padding 后检测（Grounding DINO）<br/>+ 分割（ZIM）+ 面积 NMS"]
    DET --> COMP["迭代移除物体并 inpainting<br/>（微调 Panorama-DiT 补全）"]
    COMP --> LAYERS["语义分层：前景物体层 / 背景层 / 天空层"]
    LAYERS --> DEPTH["Layer-aligned depth estimation<br/>基础全景深度 + 跨层深度对齐"]
    DEPTH --> MESH["Layer-wise 3D 重建：sheet warping 网格<br/>前景可选 3D 生成（Hunyuan3D）并摆放<br/>天空可用 HDRI"]
    MESH --> WORLD["分层 3D 世界 mesh（也支持分层 3DGS）"]
    WORLD --> VOY["Voyager：world cache 点云 + 世界一致视频扩散<br/>沿用户相机轨迹自回归扩展（§2.5）"]
    WORLD --> OPT["网格压缩（decimation / Draco）与 TensorRT 推理加速（§2.6）"]
```

图依据：论文 §2 Technical Details / Fig.2、§2.1–§2.6。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 174/571 |
| 分组 | 52 Game & Open-World Exploration Application |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2507.21809> |
| 代码/项目 | <https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0> |

## 核心机制（归纳）

### 策展导读要点

Generates immersive, explorable, and interactive 3D worlds from words or pixels.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2507.21809) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **52 Game & Open-World Exploration Application** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「HunyuanWorld 1.0」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **52 Game & Open-World Exploration Application**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2507_21809_hunyuanworld-1-0-generating-immersive-ex.md`](../../sources/papers/sun_awesome_wm_2507_21809_hunyuanworld-1-0-generating-immersive-ex.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2507.21809>
- [arXiv PDF](https://arxiv.org/pdf/2507.21809)：已通读方法部分（实验与附录以略读为准）：§2 Technical Details（§2.1–§2.6，含 Fig.2），§3 评测略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2507.21809)
