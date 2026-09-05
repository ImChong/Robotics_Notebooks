---
title: "A Definition and Roadmap for World Models"
type: entity
tags:
  [
    paper,
    world-model,
    world-action-model,
    shanghai-ai-lab,
    physical-intelligence,
    taxonomy,
    roadmap,
    perspective,
    pomdp,
    marble,
    world-labs,
    curated-index,
    awesome-world-models,
    sun254667-wm,
  ]
status: complete
updated: 2026-09-05
arxiv: "2607.06401"
venue: "arXiv 2026"
summary: "上海人工智能实验室 perspective：世界模型是有限算力下对物理状态转移的压缩；在 Fei-Fei 渲染/模拟/规划功能轴上补观测级/潜空间/3D 架构轴；WAM 不是第四列；倒金字塔数据与三阶段路线。Awesome Surveys 第 532 条升格。"
related:
  - ../concepts/functional-taxonomy-world-models.md
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ./world-labs.md
  - ./marble-world-model.md
  - ../comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md
  - ../overview/robot-world-models-training-loop-taxonomy.md
  - ../concepts/video-as-simulation.md
  - ./paper-data-pyramid-embodied-manipulation.md
  - ./paper-worldscore.md
  - ./worldarena.md
  - ../overview/bfm-41-papers-technology-map.md
  - ./awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
sources:
  - ../../sources/papers/world_model_definition_roadmap_arxiv_2607_06401.md
  - ../../sources/blogs/worldlabs_functional_taxonomy_world_models.md
  - ../../sources/papers/sun_awesome_wm_2607_06401_a-definition-and-roadmap-for-world-model.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# A Definition and Roadmap for World Models

上海 AI Lab Physical Intelligence Team 的 **perspective**：世界模型不是「能出像素的视频生成器」，而是 **有限算力下对物理状态转移的压缩**；在 Fei-Fei **渲染 / 模拟 / 规划** 功能三分之上，补 **观测级 / 潜空间 / 3D 结构化** 架构轴，并主张 **理解优先于预测**。本页由 Awesome World Models 索引卡升格为深读实体。

## 一句话定义

> **世界模型 = 有限算力下对物理状态转移的压缩；渲染 / 模拟 / 规划是读出，不是三种互斥内部模型。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 本文定义：有限算力下对物理状态转移的压缩 |
| WAM | World-Action Model | 规划向功能范式，不是第四条架构轴 |
| POMDP | Partially Observable Markov Decision Process | 局部性的形式化：隐状态、观测、动作 |
| 3DGS | 3D Gaussian Splatting | 显式 3D 表示；Marble 等 splat + collider |
| VLA | Vision-Language-Action | 规划侧端到端动作模型，不必内部持有完整世界模型 |

## 核心信息

| 字段 | 内容 |
|------|------|
| 类型 | 视角 / 路线图（无新基准数字） |
| 机构 | 上海人工智能实验室 Physical Intelligence Team |
| 署名 | 按姓氏字母：Xinyuan Chen, Haoyu Guo, Shi Guo, Bingqi Jiang, Chunhua Shen, Xing Shen, Tianfan Xue, Yufei Xue, Mulin Yu, Weinan Zhang, Bin Zhao, Bowen Zhou, Ming Zhou |
| 发表 | arXiv:2607.06401v1（2026-07） |
| 代码 | 无配套实现 |
| 源码运行时序图 | 不适用（perspective，无官方可运行代码） |
| 项目页 | [arXiv abs](https://arxiv.org/abs/2607.06401v1) · [HTML](https://arxiv.org/html/2607.06401v1) |
| Awesome 索引 | [Awesome World Models](https://github.com/sun254667/awesome-world-models) Surveys & Overviews 第 532/571 条（索引卡 `updated: 2026-08-10`） |

## 为什么重要

[Fei-Fei 功能分类](../concepts/functional-taxonomy-world-models.md) 把世界模型按输出投影成渲染器 / 模拟器 / 规划器，澄清了术语过载，但论文认为它分类的是 **解码产物**，不是内部模型。视频生成器、可微仿真器和模型基规划器被贴上同一标签，评测无法对齐。作者要回答：世界模型 **是什么**、应具备哪些性质、功能轴如何与架构轴正交。

## 核心原理（方法栈）

### 定义 2.1

世界模型是对物理状态转移动力学的压缩，满足有限计算约束：

\[
\hat{s}_{t+\Delta t} = f_\theta(s_t, a_t, \Delta t)
\]

其中 \(s\) 是物理状态（几何、外观、物理属性、力学），不是原始像素。像素是状态的 **投影**。理解是一等目标；预测是理解的手段。

### 三条性质

1. **全模态（omnimodal）**：视觉、语言、力、本体感觉共享同一物理状态，而不是各模态各训一个预测头。
2. **多维异步（multidimensional asynchronicity）**：时间、空间、语义、因果各轴分辨率不同；朴素固定时间步视频预测不是充分条件。
3. **局部性（locality）**：智能体只看到局部观测。形式化为 POMDP \(\langle \mathcal{S}, \mathcal{A}, T, R, \Omega, O, \gamma \rangle\)，世界模型对应转移 \(T\) 与观测 \(O\)。

### 二维分类

| 功能 \ 架构 | 观测级 | 潜空间 | 3D 结构化 |
|-------------|--------|--------|-----------|
| 渲染 | 视频扩散 / AR 像素 | 潜空间解码观测 | NeRF / 3DGS / 网格渲染 |
| 模拟 | 像素空间前向外推 | 潜动力学 / JEPA 类 | 可微物理、碰撞、粒子 |
| 规划 | 像素空间搜索 / 想象 | 潜空间 MPC / 世界动作 | 在显式 3D 状态上规划 |

**WAM 不是第四列。** 它是规划向功能范式，横跨模拟器内部状态。把 WAM 加成架构会把功能与表示混在同一轴上。

### 数据：倒金字塔

与 [Data Pyramid](paper-data-pyramid-embodied-manipulation.md) 同构：网络视频（宽）→ 过滤 / 合成（中）→ 真机（窄、贵）。网络视频监督观测预测，不直接监督可执行 3D 状态或接触力。

### 路线图

1. 统一多模态世界模型（共享状态，多解码器）。
2. 一种物理状态、多种功能解码（渲染 / 模拟 / 规划是读出，不是三套模型）。
3. 基础规模交互式模拟器（可查询、可重置、可对接策略）。

远期 **三位一体**：Agent / Evaluator / World Model。评测器需要世界模型提供反事实与分布外场景，而不能只靠静态视频基准。

## 实验与评测

本文是视角文，**没有**新表格或新 SOTA。评测主张：

- 像素保真（FVD、视觉 Turing）不够，要测 **状态可查询性、物理一致性、动作条件因果、规划可用性**。
- [WorldScore](paper-worldscore.md)、[WorldArena](worldarena.md) 等动态世界基准比单段视频更接近「模拟器」测试，但仍不等于完整物理状态评测。
- 不要用渲染榜单给规划器排名，也不要用任务成功率给纯生成器排名。

## 源码、项目与开放程度

| 项 | 状态（2026-09-05） |
|----|-------------------|
| 官方代码 / 权重 | **确认未开源**（arXiv HTML 无 GitHub / 权重） |
| 项目页 | 仅 arXiv abs / HTML / PDF |
| 可复现实验 | 无（无新实验） |

**源码运行时序图**：不适用。

## 工程实践

- 先写清你的系统落在二维表哪一格，再选数据与损失；不要只说「我们做了世界模型」。
- 规划需要的是可计算状态，不是更高 FVD。缺状态就补几何 / 物理 / 交互监督，而不是只加视频。
- Marble 的 splat + collider 是渲染器与模拟器边界塌缩的产品例证，不是论文实现。
- 真机数据应放在倒金字塔尖，用来校准接触与执行，而不是替代网络视频的覆盖。

## 局限与风险

- 定义 2.1 是规范主张，不是可证伪定理；「压缩」「物理状态」的操作化仍开放。
- 无新实验，对 Genie / Cosmos / Marble / VLA 的归类是阅读框架。
- 未处理社会世界、多主体博弈、非平稳人类规范。
- 倒金字塔与 Data Pyramid 同构，但没有新的数据配比实验。
- 三位一体是展望，没有 Agent–Evaluator–WM 的接口规格。

## 与其他工作对比

| 工作 | 关系 |
|------|------|
| [Fei-Fei 功能分类](../concepts/functional-taxonomy-world-models.md) | 功能轴被吸收；本文批评其未定义内部模型 |
| [生成式世界模型](../methods/generative-world-models.md) | 生成是渲染读出，不是定义本身 |
| [World-Action Models](../concepts/world-action-models.md) | WAM = 规划向范式，不是第四架构 |
| [Data Pyramid](paper-data-pyramid-embodied-manipulation.md) | 倒金字塔与其同构 |
| [训练闭环分类](../overview/robot-world-models-training-loop-taxonomy.md) | 训练闭环 vs 功能×架构，宜叠用 |
| [Awesome World Models](awesome-world-models.md) | 本页由该索引 Surveys 卡升格 |

## 结论

总判：这篇 perspective 把「世界模型」从生成口号收成 **有限算力下的物理状态转移压缩**，并强制功能轴与架构轴分开；它改变的是选模型、选数据和选评测的提问方式，而不是某一条基准曲线。

- 写系统时先填二维表：功能（渲染 / 模拟 / 规划）× 架构（观测 / 潜空间 / 3D）。
- 不要把 WAM 当成第四种架构；规划器读的是模拟器状态。
- 像素指标只证明渲染器；规划与控制要另测状态与因果。
- 数据按倒金字塔配：视频覆盖、合成补状态、真机校准接触。
- 无官方代码，复现对象是论证，不是仓库。
- 与 Fei-Fei 文合读：功能投影回答「输出什么」，定义 2.1 回答「内部必须压缩什么」。
- Awesome 索引卡已并入本页，不要再为 `2607.06401` 新建实体。

## 推荐继续阅读

- [功能分类概念页](../concepts/functional-taxonomy-world-models.md)
- [World-Action Models](../concepts/world-action-models.md)
- [生成式世界模型](../methods/generative-world-models.md)
- [Awesome 技术图谱](../overview/sun-awesome-wm-technology-map.md)

## 关联页面

- [A Functional Taxonomy of World Models](../concepts/functional-taxonomy-world-models.md)
- [World-Action Models](../concepts/world-action-models.md)
- [生成式世界模型](../methods/generative-world-models.md)
- [World Labs](world-labs.md)
- [Marble](marble-world-model.md)
- [VLM / VLN / VLA / VLX 术语分层](../comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md)
- [机器人世界模型训练闭环分类](../overview/robot-world-models-training-loop-taxonomy.md)
- [Video as Simulation](../concepts/video-as-simulation.md)
- [Data Pyramid](paper-data-pyramid-embodied-manipulation.md)
- [WorldScore](paper-worldscore.md)
- [WorldArena](worldarena.md)
- [BFM 技术图谱](../overview/bfm-41-papers-technology-map.md)
- [Awesome World Models](awesome-world-models.md)
- [Awesome 技术图谱](../overview/sun-awesome-wm-technology-map.md)

## 参考来源

- [A Definition and Roadmap for World Models](https://arxiv.org/abs/2607.06401v1)（arXiv:2607.06401v1）
- [本库论文笔记](../../sources/papers/world_model_definition_roadmap_arxiv_2607_06401.md)
- [A Functional Taxonomy of World Models](https://www.worldlabs.ai/blog/taxonomy-of-world-models)（World Labs / Fei-Fei Li，2026-06-03）
- [Awesome World Models 条目笔记](../../sources/papers/sun_awesome_wm_2607_06401_a-definition-and-roadmap-for-world-model.md)
- [Awesome World Models catalog](../../sources/papers/sun_awesome_wm_catalog.md)
- [Awesome World Models 仓库](../../sources/repos/awesome-world-models.md)
