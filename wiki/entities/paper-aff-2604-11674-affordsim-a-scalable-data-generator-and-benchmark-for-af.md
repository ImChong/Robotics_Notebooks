---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-affordsim-a-scalable-data-generator-and-benchmark-for-af
arxiv: "2604.11674"
venue: "arXiv 2026"
summary: "arXiv 2026 的 affordance 论文，清单分组：Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action · Simulation benchmark） ｜ Datasets / Benchmarks（Data Modality: simu"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_affordsim-a-scalable-data-generator-and-benchmark-for-af.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# AffordSim

**AffordSim: A Scalable Data Generator and Benchmark for Affordance-Aware Robotic Manipulation**（arXiv 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

AffordSim: A Scalable Data Generator and Benchmark for Affordance-Aware Robotic Manipulation——在清单中归入 **Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action · Simulation benchmark） ｜ Datasets / Benchmarks（Data Modality: simulation/3D/robot action）**（清单未附方法摘要，此处不替原文概括贡献）。

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
    IN["自然语言任务描述"] --> VS["VLM 场景解析（Qwen-VL-Plus）<br/>物体与关系 → 场景图 G"]
    IN --> VT["VLM 任务解析<br/>操作原语与 object-action 条目"]
    VS --> AR["按类别与属性检索 USD 资产<br/>拒绝采样做无碰撞摆放，装配仿真场景"]
    VT --> Q["每个功能接触区域一条 affordance query"]
    subgraph GS["两阶段抓取选择（§3.2）"]
        A3D["开放词表 3D affordance 模型<br/>输入资产自带点云 + query，输出逐点二值 mask"] --> S1["Stage 1：mean-shift 聚类取接触点<br/>PCA 估法向，采样 K 个接近方向得到候选抓取"]
        S1 --> S2["Stage 2：cuRobo 规划预抓取轨迹<br/>得分 = 1/关节空间弧长，碰撞则为 0，取最高分"]
    end
    Q --> A3D
    AR --> A3D
    S2 --> EX["cuRobo 规划闭合、抬起、任务完成段<br/>仿真执行完整轨迹"]
    EX --> DR["域随机化：背景（纹理或 3DGS 真实场景重建）、<br/>光照、物体纹理、物体位姿、图像噪声"]
    DR --> OUT["可用于模仿学习的演示数据"]
```

图依据：论文 §3 AffordSim Framework 开头概述、§3.1 VLM-Powered Task and Scene Generation、§3.2 Affordance-Aware Trajectory Collection、§3.3 Domain Randomization（50 任务基准见 §3.4，未入图）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2026 |
| 方法名（清单标注） | AffordSim |
| 分组 1 | Action · Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action · Simulation benchmark |
| 分组 2 | Datasets / Benchmarks · Data Modality: simulation/3D/robot action |
| 论文 | <https://arxiv.org/abs/2604.11674> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；网络结构与流程见上方「流程总览」（依据论文方法部分），训练细节与实验请读 [原文](https://arxiv.org/abs/2604.11674)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2604.11674) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「AffordSim」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的只有标题、出处与分类标签；方法与结果未经本页核实。
- 开源状态清单未标注，入库时未打开项目页核查，复现前请自行确认。
- 要深读：从原文入手，再回综述实体页对照 perception → reasoning → action 的位置。

## 常见误区

1. 不要把分类标签当作论文结论；标签来自清单维护者的归类。
2. 不要把本页当作方法证明；它只是清单入口。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [manipulation.md](../tasks/manipulation.md)、[imitation-learning.md](../methods/imitation-learning.md)

## 参考来源

- [`sources/papers/affordance_awesome_affordsim-a-scalable-data-generator-and-benchmark-for-af.md`](../../sources/papers/affordance_awesome_affordsim-a-scalable-data-generator-and-benchmark-for-af.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2604.11674>
- [arXiv PDF](https://arxiv.org/pdf/2604.11674)：已通读方法部分（实验与附录以略读为准）：§3.1–§3.4

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2604.11674)
