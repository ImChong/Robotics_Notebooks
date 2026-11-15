---
type: comparison
tags: [3dgs, web, rendering, tooling, selection]
status: complete
original: ../../sources/blogs/worldlabs_spark_2_0_streaming_3dgs.md
updated: 2026-05-28
related:
  - ../entities/spark-3dgs-renderer.md
  - ../entities/aholo-viewer.md
  - ../entities/world-labs.md
  - ../entities/gs-playground.md
sources:
  - ../../sources/blogs/worldlabs_spark_2_0_streaming_3dgs.md
  - ../../sources/repos/aholo-viewer.md
summary: "Spark（World Labs）与 Aholo Viewer（manycoretech）均为浏览器端大场景 3DGS 渲染栈；前者深度集成 THREE.js/Marble 与 .RAD LoD 树，后者强调 Chunked Streaming LoD 与 3DGS+Mesh 混渲及 TypeScript monorepo 工具链。"
---

# Spark vs Aholo Viewer：Web 大场景 3DGS 渲染选型

两者都解决 **consumer 设备 上交互式浏览千万级 splat** 的问题，但 **生态绑定、格式与混渲能力** 不同。机器人研究若只需 **训练用光真实感观测**，应优先 [GS-Playground](../entities/gs-playground.md) 等仿真向方案，而非本对比。

## 一句话定义

对比 World Labs Spark 与 manycoretech Aholo Viewer 两款面向浏览器的大场景 3DGS 流式渲染器，梳理其生态定位、LoD 架构与许可差异。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GPU | Graphics Processing Unit | 图形处理器，大规模并行仿真训练的算力基础 |
| API | Application Programming Interface | 应用程序编程接口 |
| CAD | Computer-Aided Design | 计算机辅助设计，硬件结构建模 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |

## 为什么重要

- 浏览器端 3DGS 渲染是 Real2Sim 数字孪生预览与远程标注的关键基础设施，不同渲染器在 THREE.js 兼容性、Mesh 混渲与工具链许可上差异显著。

## 核心原理与方法栈

- **Spark**：基于 LoD splat 树与 `.RAD` 流式格式，深度集成 THREE.js 生态。
- **Aholo Viewer**：采用 Chunked Streaming LoD 与 pnpm monorepo 架构，原生支持 3DGS 与三角网格同屏混渲。

## 对比表

| 维度 | [Spark](../entities/spark-3dgs-renderer.md) | [Aholo Viewer](../entities/aholo-viewer.md) |
|------|---------------------------------------------|---------------------------------------------|
| 维护方 | World Labs | manycoretech |
| 3D 框架 | THREE.js | 自研 TS 渲染包（见 `packages/renderer`） |
| GPU API | WebGL2 | Web（README 未强调 WebGPU 必选） |
| 大场景策略 | **LoD splat 树** + **.RAD** 渐进流式 + GPU splat 分页 | **Chunked Streaming LoD** |
| 几何类型 | 3DGS 为主（多对象全局排序） | **3DGS + Mesh** |
| 动态效果 | 可编程 splat 管线、4DGS、重光照（shader graph） | 以文档/架构为准；Playground 侧重示例集成 |
| 资产工具 | `build-lod` → .RAD；Tiny-LoD / Bhatt-LoD | 依赖 `external/splat-transform`（**专有**） |
| 文档/体验 | sparkjs.dev；与 Marble 产品同研 | aholojs.dev；Astro 手册 + URL 压缩 Playground |
| 典型规模（公开案例） | 博客/demo：6M–100M+ splat（.rad） | README 强调「huge 3DGS」；具体上限需实测 |

## 何时优先 Spark

- 团队已用 **THREE.js**，或需要与 **Marble / World Labs** 生成资产同栈。
- 需要公开、可引用的 **.RAD + LoD splat 树 + 虚拟 splat 内存** 技术说明（见 [Spark 2.0 博客](../../sources/blogs/worldlabs_spark_2_0_streaming_3dgs.md)）。
- 强调 **多 3DGS 对象全局深度排序** 与 **4DGS / 实时 splat 特效**。

## 何时优先 Aholo

- 需要 **同一视口内 splat 与 Mesh**（例如网格碰撞体、CAD 与 photogrammetry 叠显）。
- 偏好 **TypeScript monorepo + TypeDoc API + 内容校验脚本** 的二次开发流程。
- 可接受 **专有 splat-transform** 工具链的许可边界。

## 工程实践

- **选型建议**：已有 THREE.js 前端栈且需纯 splat 流式加载时优先 Spark；需 CAD/碰撞体 Mesh 与 3DGS 同屏审阅时优先 Aholo Viewer，并注意其专有工具链边界。

## 共同局限

- **不替代物理仿真**：接触、约束与 RL 训练吞吐量不在二者设计中心。
- **Web 端排序/流式复杂度**：超大场景仍受网络、GPU 内存与主线程/Worker 协作影响，需按目标设备 profiling。

## 关联页面

- [Spark（Web 3DGS 渲染器）](../entities/spark-3dgs-renderer.md)
- [Aholo Viewer](../entities/aholo-viewer.md)
- [World Labs](../entities/world-labs.md)
- [GS-Playground](../entities/gs-playground.md)
- [生成式世界模型](../methods/generative-world-models.md)

## 参考来源

- [World Labs Spark 2.0 博客归档](../../sources/blogs/worldlabs_spark_2_0_streaming_3dgs.md)
- [aholo-viewer 仓库归档](../../sources/repos/aholo-viewer.md)

## 推荐继续阅读

- [Spark 2.0 技术稿](https://www.worldlabs.ai/blog/spark-2.0)
- [Aholo 入门](https://aholojs.dev/en-US/manual/getting-started/)
