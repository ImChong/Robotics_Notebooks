---

type: entity
tags: [industry, world-models, 3dgs, generative-ai, spatial-intelligence, web, simulation-adjacent, nvidia]
status: complete
updated: 2026-09-05
related:
  - ../methods/generative-world-models.md
  - ../concepts/functional-taxonomy-world-models.md
  - ../entities/paper-sa-2607-06401-a-definition-and-roadmap-for-world-models.md
  - ./marble-world-model.md
  - ./atlas-world-model.md
  - ./spark-3dgs-renderer.md
  - ./aholo-viewer.md
  - ./gs-playground.md
  - ../concepts/video-as-simulation.md
  - ./nvidia-omniverse.md
  - ../comparisons/spark-vs-aholo-web-3dgs-renderers.md
sources:
  - ../../sources/sites/worldlabs-ai.md
  - ../../sources/sites/worldlabs-docs.md
  - ../../sources/blogs/worldlabs_marble_world_model.md
  - ../../sources/blogs/worldlabs_functional_taxonomy_world_models.md
  - ../../sources/blogs/worldlabs_atlas_omni_world_model.md
  - ../../sources/blogs/worldlabs_spark_2_0_streaming_3dgs.md
summary: "World Labs 是公开以「空间智能」与三维世界模型为主线的公司与产品团队：首款产品 Marble（2025-11 GA，文档+World API）从多模态输入生成可导出 3D 世界；2026 年发布 omni 底座 Atlas（早期访问）；并开源 Spark（Web 端 3DGS 渲染栈）。"
---

# World Labs（空间智能与世界生成）

**World Labs** 在公开材料中将自身定位为 **空间智能（spatial intelligence）** 公司与 **前沿世界模型** 研发方：强调模型对三维世界的 **感知、生成、推理与交互**，并把这些能力接到 **叙事、创意、设计、仿真与沉浸式体验** 等用例。与本仓库大量讨论的 **具身像素/视频世界模型**（预测下一帧观测以支持 RL / 规划）相比，其首款产品 **Marble** 更贴近 **生成式 3D 场景资产与可漫游环境** 的产品路径；技术侧又通过开源 **Spark** 将 **3D Gaussian Splatting（3DGS）** 推向 Web 端大场景流式渲染，与机器人学习里「高真实感视觉仿真」路线相邻但目标读者不同。

## 一句话定义

以 **生成式 3D 世界** 与 **Web 端 3DGS 基础设施（Spark）** 为双支点，把「世界模型」从视频预测扩展到 **可编辑、可组合、可导出的三维空间媒介**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Marble | Marble world model | 首款产品：多模态生成持久 3D 世界（SaaS + API） |
| Atlas | Atlas omni world model | 2026-09 公开的多模态自回归扩散底座（早期访问） |
| Spark | Spark 3DGS renderer | 开源 Web 3DGS 栈；Marble 站点与导出推荐运行时 |
| 3DGS | 3D Gaussian Splatting | Marble / Atlas 的高保真场景表征 |
| API | Application Programming Interface | World API（platform.worldlabs.ai）付费生成 |

## 为什么重要

- **概念对齐样本**：「世界模型」一词在产业与论文中跨度极大；World Labs 的公开叙事可作为 **3D 世界生成 / 空间智能** 分支的对照轴，避免与 **EWM 视频基准**（如 [EWMBench](./ewmbench.md)）或 **World Action Models** 混为一谈。2026-06 的 [功能分类](../concepts/functional-taxonomy-world-models.md) 把过载词拆成 Renderer / Simulator / Planner；[上海人工智能实验室定义文](./paper-sa-2607-06401-a-definition-and-roadmap-for-world-models.md) 把它收成功能轴。
- **3DGS 工程参考**：Spark 2.0 公开阐述了 **LoD splat 树、渐进流式、.RAD 随机访问与显存分页** 等系统细节（已消化至 [spark-3dgs-renderer.md](./spark-3dgs-renderer.md)）；同赛道可对照 [Aholo Viewer](./aholo-viewer.md) 与 [选型对比](../comparisons/spark-vs-aholo-web-3dgs-renderers.md)。
- **产品化闭环**：[Marble](./marble-world-model.md) + Marble Labs + **World API** 把 **生成—编辑—导出** 串成创作者工作流（文档给出模型档位、积分与 SPZ/collider/HQ mesh 门禁），便于观察「世界模型」在 **内容生产管线** 而非 **控制回路** 中的落地形态。

## 产品与开源组件（归纳）

| 名称 | 类型 | 说明 |
|------|------|------|
| **Marble** | 浏览器产品 + World API | 2025-11-12 GA。文/图/多图/视频/Chisel → 可编辑世界；导出 SPZ/PLY、collider、HQ mesh。能力与门禁见 [Marble 实体页](./marble-world-model.md) 与 [docs.worldlabs.ai](https://docs.worldlabs.ai/) |
| **Marble Labs** | 教程与案例站 | Showcase、生产向案例与文档/教程入口 |
| **Atlas** | omni 世界模型 | **多模态自回归扩散 Transformer**；统一相机可控生成、稀疏 3D 重建、时空仿真与 Real-to-Sim 机器人传感器 rollout；**早期访问**，详见 [Atlas 实体页](./atlas-world-model.md) |
| **Spark** | 开源 3DGS 渲染器 | 基于 **THREE.js** 与 **WebGL2**；2.0 的 LoD splat 树、**.RAD** 流式与 splat 分页见 [Spark 实体页](./spark-3dgs-renderer.md) 与 [Spark 2.0 博客归档](../../sources/blogs/worldlabs_spark_2_0_streaming_3dgs.md) |

## 团队背景（仅记录公开 About 页表述）

公开 **About** 页列出的联合创始人包括 **Fei-Fei Li、Justin Johnson、Christoph Lassner、Ben Mildenhall**（姓名与顺序以官网为准）。融资与商业条款以官方新闻稿与监管披露为准，本页不作摘录。

## 常见误区或局限

- **不是具身控制基准线**：Marble 的公开材料面向 **空间内容与交互编辑**；若要讨论 **机器人操作 rollout 的物理一致性**，仍应回到解析仿真或 [生成式世界模型](../methods/generative-world-models.md) 论文语境。
- **物理与几何保证**：3DGS 表征擅长外观与实时浏览；**接触动力学、可执行约束与可证明安全性** 不会从渲染栈自动出现。
- **信息时效**：产品能力、定价与 API 以官网为准；本 wiki 页只保留 **知识结构** 与 **可追溯来源**。

## 关联页面

- [世界模型功能分类（Renderer / Simulator / Planner）](../concepts/functional-taxonomy-world-models.md) — 2026-06 官方消歧文
- [世界模型定义与路线图](./paper-sa-2607-06401-a-definition-and-roadmap-for-world-models.md) — 把本页产品放进功能×架构格子
- [生成式世界模型（Generative World Models）](../methods/generative-world-models.md)
- [Marble（多模态世界模型产品）](./marble-world-model.md)
- [Atlas（omni 世界模型）](./atlas-world-model.md)
- [Spark（Web 3DGS 渲染器）](./spark-3dgs-renderer.md)
- [Aholo Viewer](./aholo-viewer.md)
- [GS-Playground（3DGS × 并行仿真）](./gs-playground.md)
- [Video-as-Simulation](../concepts/video-as-simulation.md)
- [NVIDIA Omniverse（具身仿真底座）](./nvidia-omniverse.md)
- [EWMBench（具身世界模型视频生成评测）](./ewmbench.md) — 与「像素 rollout 评测」对照，界定 Marble 类产品不在同一基准轴上

## 参考来源

- [World Labs 官方站点与关联链接归档](../../sources/sites/worldlabs-ai.md)
- [Marble 文档归档](../../sources/sites/worldlabs-docs.md)
- [功能分类博客归档](../../sources/blogs/worldlabs_functional_taxonomy_world_models.md)
- [Marble GA 博客归档](../../sources/blogs/worldlabs_marble_world_model.md)
- [Atlas 技术博客归档](../../sources/blogs/worldlabs_atlas_omni_world_model.md)
- [Spark 2.0 技术博客归档](../../sources/blogs/worldlabs_spark_2_0_streaming_3dgs.md)

## 推荐继续阅读

- [World Labs 首页](https://www.worldlabs.ai/)
- [Marble 文档](https://docs.worldlabs.ai/)
- [A Functional Taxonomy of World Models](https://www.worldlabs.ai/blog/taxonomy-of-world-models)
- [Marble: A Multimodal World Model](https://www.worldlabs.ai/blog/marble-world-model)
- [Atlas: A World Model for Spatial Intelligence](https://www.worldlabs.ai/blog/atlas)
- [Streaming 3DGS worlds on the web（Spark 2.0 技术稿）](https://www.worldlabs.ai/blog/spark-2.0)
- [Spark 文档与示例](https://sparkjs.dev/)
