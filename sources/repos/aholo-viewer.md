# aholo-viewer

> 来源归档

- **标题：** Aholo Viewer
- **类型：** repo
- **来源：** manycoretech（GitHub 组织）
- **链接：** https://github.com/manycoretech/aholo-viewer
- **官网：** https://aholojs.dev/
- **入库日期：** 2026-05-28
- **一句话说明：** 面向 Web 的高性能 **3DGS + Mesh** 渲染 monorepo；以 **Chunked Streaming LoD** 承载超大 splat 场景，TypeScript 渲染包 + Astro 文档站，依赖 `external/egs-core` 子模块与专有 `splat-transform` 工作区包。
- **沉淀到 wiki：** 是 → [`wiki/entities/aholo-viewer.md`](../../wiki/entities/aholo-viewer.md)

---

## 核心定位

Aholo Viewer 与 World Labs [Spark](../../wiki/entities/spark-3dgs-renderer.md) 同属 **浏览器端 3D Gaussian Splatting（3DGS）** 工程栈，但由 **manycoretech** 独立维护：强调 **Chunked Streaming LoD**、**3DGS 与三角网格同场景渲染**，以及 **pnpm monorepo + TypeDoc API** 的开发者体验。

## 仓库结构（README 归纳）

```text
aholo-viewer/
  packages/renderer/    # 渲染器 TypeScript 源码包
  website/              # Astro：手册、示例、Playground、API 文档
  external/egs-core/    # 必需上游子模块（只读对待）
  external/splat-transform/  # 必需工作区包（专有许可，见 COPYRIGHT）
  docs/                 # 架构与 AI 协作说明
```

## 技术要点

| 主题 | 说明 |
|------|------|
| 表征 | 3DGS splat + Mesh |
| 大场景 | **Chunked Streaming LoD** schema |
| 工具链 | Node ≥ 22.22.1、pnpm；`pnpm build:renderer` / `docs:api` |
| Playground | URL 用 `lz-string` 压缩源码；示例成对 `<slug>.json` + `<slug>.ts` |
| 许可 | 主体 MIT；`external/splat-transform/` 为专有，不可再分发工具本身 |

## 对 wiki 的映射

- [`wiki/entities/aholo-viewer.md`](../../wiki/entities/aholo-viewer.md) — 实体页与 Spark 对照
- [`wiki/comparisons/spark-vs-aholo-web-3dgs-renderers.md`](../../wiki/comparisons/spark-vs-aholo-web-3dgs-renderers.md) — Web 3DGS 渲染栈选型

## 与本项目其他资料的关系

| 资料 | 关系 |
|------|------|
| [worldlabs_spark_2_0_streaming_3dgs.md](../blogs/worldlabs_spark_2_0_streaming_3dgs.md) | 同赛道：Spark 2.0 LoD + .RAD 流式 |
| [gs_playground.md](gs_playground.md) | 机器人侧批量 3DGS 仿真渲染，非 Web 交互浏览 |
| [worldlabs-ai.md](../sites/worldlabs-ai.md) | World Labs / Marble / Spark 产业侧归档 |
