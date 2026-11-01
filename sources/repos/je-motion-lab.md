# JE Motion Lab（mocap-web-viewer）

> 来源归档（ingest）

- **类型：** repo
- **链接：** <https://github.com/JEBbecke/mocap-web-viewer>
- **项目主页：** <https://jemolab.com/>
- **在线应用：** <https://app.jemolab.com/>
- **许可证：** PolyForm Noncommercial 1.0.0；商业使用需另行授权。
- **入库日期：** 2026-10-29
- **源码状态：** 仓库公开提供应用源码；匿名统计使用的 Cloudflare Worker 与数据库在仓库外管理。
- **沉淀到 wiki：** [JE Motion Lab](../../wiki/entities/je-motion-lab.md)

## 入口与结构

README 提供 Node.js 22.12+ 的本地运行步骤：`npm ci` 后 `npm run dev`。应用使用 TypeScript、React、Vite、Three.js、Zustand、uPlot 与 h5wasm。 `src/workers/import.worker.ts` 根据文件选择 importer 并返回解析数据；`src/viewer/Viewer3D.tsx` 负责三维视图。

README 描述 C3D 和特定机构 H5 导入、数据查看、裁剪与导出。参见 [README](https://github.com/JEBbecke/mocap-web-viewer/blob/main/README.md) 和 [格式限制](https://github.com/JEBbecke/mocap-web-viewer/blob/main/docs/PARSERS.md)。[Releases](https://github.com/JEBbecke/mocap-web-viewer/releases) 提供版本与源码包；截至核查时 v0.6.0 没有额外二进制附件。
