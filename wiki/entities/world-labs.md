---
type: entity
tags: [industry, world-models, 3dgs, generative-ai, spatial-intelligence, web, simulation-adjacent]
status: complete
updated: 2026-05-16
related:
  - ../methods/generative-world-models.md
  - ./gs-playground.md
  - ../concepts/video-as-simulation.md
  - ./nvidia-omniverse.md
sources:
  - ../../sources/sites/worldlabs-ai.md
summary: "World Labs 是公开以「空间智能」与三维世界模型为主线的公司与产品团队：浏览器产品 Marble 从文本/图像/视频/全景生成可编辑的持久 3D 世界，并开源 Spark（Web 端 3DGS 渲染与流式大场景技术栈）支撑交互式展示与管线集成。"
---

# World Labs（空间智能与世界生成）

**World Labs** 在公开材料中将自身定位为 **空间智能（spatial intelligence）** 公司与 **前沿世界模型** 研发方：强调模型对三维世界的 **感知、生成、推理与交互**，并把这些能力接到 **叙事、创意、设计、仿真与沉浸式体验** 等用例。与本仓库大量讨论的 **具身像素/视频世界模型**（预测下一帧观测以支持 RL / 规划）相比，其首款产品 **Marble** 更贴近 **生成式 3D 场景资产与可漫游环境** 的产品路径；技术侧又通过开源 **Spark** 将 **3D Gaussian Splatting（3DGS）** 推向 Web 端大场景流式渲染，与机器人学习里「高真实感视觉仿真」路线相邻但目标读者不同。

## 一句话定义

以 **生成式 3D 世界** 与 **Web 端 3DGS 基础设施（Spark）** 为双支点，把「世界模型」从视频预测扩展到 **可编辑、可组合、可导出的三维空间媒介**。

## 为什么重要

- **概念对齐样本**：「世界模型」一词在产业与论文中跨度极大；World Labs 的公开叙事可作为 **3D 世界生成 / 空间智能** 分支的对照轴，避免与 **EWM 视频基准**（如 [EWMBench](./ewmbench.md)）或 **World Action Models** 混为一谈。
- **3DGS 工程参考**：Spark 2.0 公开阐述了 **LoD splat 树、渐进流式、.RAD 随机访问与显存分页** 等系统细节，对任何需要在浏览器或轻端设备上承载 **千万级 splat** 的应用（含科研可视化、数字孪生预览）有直接可读性。
- **产品化闭环**：Marble + Marble Labs 把 **生成—编辑—导出** 串成创作者工作流，便于观察「世界模型」在 **内容生产管线** 而非 **控制回路** 中的落地形态。

## 产品与开源组件（归纳）

| 名称 | 类型 | 说明 |
|------|------|------|
| **Marble** | 浏览器产品 | 从文本、图像、视频或 360° 全景生成 **空间一致、高保真、可持久化** 的 3D 世界；支持布局控制、元素级编辑、世界扩展与多格式导出（官方营销页表述） |
| **Marble Labs** | 教程与案例站 | Showcase、生产向案例与文档/教程入口 |
| **Spark** | 开源 3DGS 渲染器 | 基于 **THREE.js** 与 **WebGL2**；2.0 引入可流式传输的大场景 LoD 与相关文件格式（详见官方博客与 sparkjs.dev） |

## 团队背景（仅记录公开 About 页表述）

公开 **About** 页列出的联合创始人包括 **Fei-Fei Li、Justin Johnson、Christoph Lassner、Ben Mildenhall**（姓名与顺序以官网为准）。融资与商业条款以官方新闻稿与监管披露为准，本页不作摘录。

## 常见误区或局限

- **不是具身控制基准线**：Marble 的公开材料面向 **空间内容与交互编辑**；若要讨论 **机器人操作 rollout 的物理一致性**，仍应回到解析仿真或 [生成式世界模型](../methods/generative-world-models.md) 论文语境。
- **物理与几何保证**：3DGS 表征擅长外观与实时浏览；**接触动力学、可执行约束与可证明安全性** 不会从渲染栈自动出现。
- **信息时效**：产品能力、定价与 API 以官网为准；本 wiki 页只保留 **知识结构** 与 **可追溯来源**。

## 关联页面

- [生成式世界模型（Generative World Models）](../methods/generative-world-models.md)
- [GS-Playground（3DGS × 并行仿真）](./gs-playground.md)
- [Video-as-Simulation](../concepts/video-as-simulation.md)
- [NVIDIA Omniverse（具身仿真底座）](./nvidia-omniverse.md)
- [EWMBench（具身世界模型视频生成评测）](./ewmbench.md) — 与「像素 rollout 评测」对照，界定 Marble 类产品不在同一基准轴上

## 参考来源

- [World Labs 官方站点与关联链接归档](../../sources/sites/worldlabs-ai.md)

## 推荐继续阅读

- [World Labs 首页](https://www.worldlabs.ai/)
- [Streaming 3DGS worlds on the web（Spark 2.0 技术稿）](https://www.worldlabs.ai/blog/spark-2.0)
- [Spark 文档与示例](https://sparkjs.dev/)
