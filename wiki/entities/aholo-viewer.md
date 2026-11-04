---

type: entity
tags: [repo, 3dgs, gaussian-splatting, web, mesh, streaming, lod, typescript, linux-foundation]
status: complete
updated: 2026-05-28
related:
  - ./spark-3dgs-renderer.md
  - ./gs-playground.md
  - ./world-labs.md
  - ../comparisons/spark-vs-aholo-web-3dgs-renderers.md
sources:
  - ../../sources/repos/aholo-viewer.md
summary: "Aholo Viewer 是 manycoretech 的高性能 Web 3DGS/Mesh 渲染 monorepo，以 Chunked Streaming LoD 承载超大 splat 场景，并提供 Astro 文档站、Playground 与 TypeDoc API。"
---

# Aholo Viewer

**Aholo Viewer**（[manycoretech/aholo-viewer](https://github.com/manycoretech/aholo-viewer)）是面向 **浏览器** 的 **3D Gaussian Splatting（3DGS）与三角网格（Mesh）** 高性能渲染项目，官网 [aholojs.dev](https://aholojs.dev/)。与 [World Labs Spark](./spark-3dgs-renderer.md) 同属 Web 大场景 splat 赛道，但由独立团队维护，架构为 **pnpm monorepo + TypeScript 渲染包**。

## 一句话定义

用 **Chunked Streaming LoD** 在 Web 上同时承载 **超大 3DGS 与 Mesh**，并以文档站 / Playground 降低集成门槛。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| API | Application Programming Interface | 应用程序编程接口 |
| CAD | Computer-Aided Design | 计算机辅助设计，硬件结构建模 |

## 为什么重要

- **3DGS + Mesh 混渲**：许多机器人/数字孪生场景需要 **splat 外观 + 传统网格碰撞体或 CAD** 同屏；Aholo 公开定位覆盖两类几何。
- **流式 LoD 另一实现样本**：与 Spark 2.0 的 **LoD splat 树 + .RAD** 对照，帮助理解「Chunked Streaming LoD」在工程上的不同切分。
- **开发者工具链完整**：`pnpm check:content` 校验手册/示例一致性；Playground 用 `lz-string` 在 URL 中保存代码，便于分享复现。

## 仓库与依赖边界

| 路径 | 角色 |
|------|------|
| `packages/renderer/` | 渲染器 TS 源码（API 由 TypeDoc 生成） |
| `website/` | Astro：手册、示例、Playground |
| `external/egs-core/` | **必需子模块**（上游，默认只读） |
| `external/splat-transform/` | **专有** splat 处理工具（可用但不可再分发，见 COPYRIGHT） |

构建要求：**Node ≥ 22.22.1**、**pnpm**；克隆须 `git clone --recurse-submodules`。

## 流程总览（Chunked Streaming LoD）

```mermaid
flowchart TB
  subgraph ingest [资产侧]
    S[3DGS / Mesh 资产]
    T[splat-transform 等工具链]
  end
  subgraph stream [流式 LoD]
    C[分块 Chunk]
    L[Streaming LoD 选择]
  end
  subgraph web [浏览器]
    R[packages/renderer]
    V[视锥内 splat + mesh 绘制]
  end
  S --> T --> C --> L --> R --> V
```

（具体 chunk 划分与 LoD 策略以官方 `docs/architecture.md` 为准；本页仅保留公开 README 级归纳。）

## 常见误区或局限

- **`splat-transform` 非开源**：内容生成/处理工具为专有许可，集成时需遵守 [COPYRIGHT](https://github.com/manycoretech/aholo-viewer/blob/main/external/splat-transform/COPYRIGHT.md)。
- **≠ 机器人仿真**：不提供与 [GS-Playground](./gs-playground.md) 同类的 **物理步进 + 批量训练观测**；侧重交互浏览与内容交付。
- **生态绑定**：与 Spark 的 THREE.js 生态不同，选型时应比对团队现有前端栈与许可。

## 源码运行时序图

入口：`pnpm dev`（仓库根：构建 renderer、生成 API 文档并启动 website 的 Astro 开发服务）；分块资产预览：`node external/splat-dev-server/bin/server.js <资产目录>`；离线分块由 `packages/splat-transform-native` 的 `auto_chunk_lod` 生成 `lod-meta.json`

```mermaid
sequenceDiagram
    autonumber
    participant O as 离线 splat-transform-native auto_chunk_lod
    participant SV as splat-dev-server src/server.js
    participant A as 应用页面帧循环 website 示例
    participant V as egs Viewer.render
    participant L as splat-utils LodSplat
    participant R as LodSplat ResourceManager
    participant LD as DefaultLoadResource fetch 与 parseSplatData

    O->>SV: 输出分块文件与 lod-meta.json 由服务目录托管
    A->>V: 每帧 render 触发 scene.update
    A->>L: LodSplat.tick camera 更新视图与投影矩阵
    A->>L: LodSplat.flush 进入调度
    L->>L: updateVisibility 视锥剔除并计算节点预算
    L->>L: schedule 计算各节点目标 LOD 与 coverage
    L->>R: request 按 coverage 与 refinement 请求资源
    R->>LD: load idx 调用 loadResource
    LD->>SV: fetch 分块 url 命中 lod-meta.json 路由或静态文件
    SV-->>LD: 分块字节流
    LD-->>R: 解析得到 SplatData
    R-->>L: resource has idx 就绪
    L->>L: upgrade merge commit 替换节点 proxy
    loop 后续帧
        A->>L: tick 与 flush 继续细化或卸载
    end
```

图依据：`fcd3b19`（默认分支 HEAD）的 `package.json`（scripts）、`README.md`（Commands 与 Structure）、`packages/renderer/src/index.ts`（局部）、`packages/renderer/src/splat-loader.ts`（局部）、`external/splat-dev-server/bin/server.js`、`external/splat-dev-server/src/server.js`（局部）、`packages/splat-transform-native/source/src/pipeline/tasks/auto_chunk_lod.cpp`（仅按 `lod-meta.json` 命中行）；外部子模块 `external/egs-core` 固定在 `288143f`，依据其 `packages/egs/src/Viewer.ts`（`render` 段）、`packages/egs/src/ViewerContext.ts`（`createViewerContext`）、`packages/utils/splat-utils/lod/index.ts`（`tick`、`updateVisibility`、`schedule`、`flush`、`DefaultLoadResource` 段）、`packages/utils/splat-utils/lod/ResourceManager.ts`（`request` 与 `load` 段）。`upgrade`、`merge`、`commit` 内部与 `streaming.ts` 未逐行读，图中仅示意。未运行代码。

## 关联页面

- [Spark（Web 3DGS）](./spark-3dgs-renderer.md)
- [Spark vs Aholo 对比](../comparisons/spark-vs-aholo-web-3dgs-renderers.md)
- [GS-Playground](./gs-playground.md)
- [World Labs](./world-labs.md)

## 参考来源

- [aholo-viewer 仓库归档](../../sources/repos/aholo-viewer.md)
- 源码核对（2026-11-02）：manycoretech/aholo-viewer@fcd3b19（默认分支 HEAD），读过 package.json、README.md、packages/renderer/src/index.ts（局部）、packages/renderer/src/splat-loader.ts（局部）、external/splat-dev-server/bin/server.js、external/splat-dev-server/src/server.js（局部）；子模块 manycoretech/egs@288143f 读过 packages/egs/src/Viewer.ts（局部）、packages/egs/src/ViewerContext.ts（局部）、packages/utils/splat-utils/lod/index.ts（局部）、packages/utils/splat-utils/lod/ResourceManager.ts（局部）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [Aholo 入门手册](https://aholojs.dev/en-US/manual/getting-started/)
- [GitHub：manycoretech/aholo-viewer](https://github.com/manycoretech/aholo-viewer)
