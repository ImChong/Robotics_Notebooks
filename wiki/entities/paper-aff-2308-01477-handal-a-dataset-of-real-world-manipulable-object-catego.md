---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-handal-a-dataset-of-real-world-manipulable-object-catego
arxiv: "2308.01477"
venue: "IROS 2023"
summary: "IROS 2023 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Visual · Spatial） ｜ Datasets / Benchmarks（Data Modality: RGB-D/6-DoF pose/3D mesh）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_handal-a-dataset-of-real-world-manipulable-object-catego.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# HANDAL

**HANDAL: A Dataset of Real-World Manipulable Object Categories with Pose Annotations, Affordances, and Reconstructions**（IROS 2023）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

HANDAL: A Dataset of Real-World Manipulable Object Categories with Pose Annotations, Affordances, and Reconstructions——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Visual · Spatial） ｜ Datasets / Benchmarks（Data Modality: RGB-D/6-DoF pose/3D mesh）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Affordance | Action possibility (Gibson) | 环境/物体对 agent 提供的可交互机会 |
| HOI | Human-Object Interaction | 人–物交互，常作 affordance 监督来源 |
| VLM | Vision-Language Model | 视觉–语言模型，用于 affordance 推理与接地 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |

## 为什么重要

- 综述把 affordance 学习拆成感知、推理、引导动作三段（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），本条目是其中一段的代表文献之一。
- 可顺着 [技术地图](../overview/affordance-awesome-technology-map.md) 的同分组条目横向阅读同方向工作。

## 流程总览

```mermaid
flowchart TD
    A["静态场景：移动设备相机拍摄 HD 视频（ARKit/ARCore 粗位姿）"] --> B["按 Laplacian 方差选取最清晰帧（约 1000 帧 → 120 帧）"]
    B --> C["COLMAP 估计无尺度相机位姿"]
    C --> D["XMem 分割前景物体"]
    D --> E["Instant NGP 重建 + Marching Cubes 提取网格"]
    E --> F["用 ARKit/ARCore 位姿恢复尺度并对齐重力方向"]
    F --> G["与类别规范坐标系下的 canonical mesh 半自动配准"]
    G --> H["逐帧物体 6-DoF 位姿 + 尺度（人工剔除重投影差的帧）"]
    E --> I["在 3D 网格上人工标注把手区域 affordance"]
    J["动态场景：iPad TrueDepth 视频，物体旋转 + 人手功能性抓取"] --> K["XMem 分割 + BundleSDF 重建与 6-DoF 跟踪，与 ICP 结果取平均"]
    K --> L["同样配准到 canonical mesh，得逐帧位姿"]
    H --> M["标准格式导出：COCO 2D 框/实例掩码，BOP 6-DoF 位姿"]
    I --> M
    L --> M
```

图依据：论文 §IV（A–D）数据采集与标注流程，§V-A 标注格式

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | IROS 2023 |
| 方法名（清单标注） | HANDAL |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Visual · Spatial |
| 分组 2 | Datasets / Benchmarks · Data Modality: RGB-D/6-DoF pose/3D mesh |
| 论文 | <https://arxiv.org/abs/2308.01477> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要。数据采集与标注流程见上方「流程总览」（依据论文方法部分）；需要机制与实验请读 [原文](https://arxiv.org/abs/2308.01477)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2308.01477) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「HANDAL」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的有标题、出处与分类标签，以及「流程总览」所述方法流程（依据论文方法部分）；实验结果未经本页核实。
- 开源状态清单未标注，入库时未打开项目页核查，复现前请自行确认。
- 要深读：从原文入手，再回综述实体页对照 perception → reasoning → action 的位置。

## 常见误区

1. 不要把分类标签当作论文结论；标签来自清单维护者的归类。
2. 不要把本页当作方法证明；它只是清单入口。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [manipulation.md](../tasks/manipulation.md)、[vla.md](../methods/vla.md)

## 参考来源

- [`sources/papers/affordance_awesome_handal-a-dataset-of-real-world-manipulable-object-catego.md`](../../sources/papers/affordance_awesome_handal-a-dataset-of-real-world-manipulable-object-catego.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2308.01477>
- [arXiv PDF](https://arxiv.org/pdf/2308.01477)：已通读方法部分（实验与附录以略读为准）：§III 数据集概览、§IV 方法（数据采集与标注 A–D）、§V-A 数据集统计

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2308.01477)
