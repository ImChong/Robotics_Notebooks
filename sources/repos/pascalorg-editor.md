# Pascal Editor — 开源 3D 建筑编辑器与 MCP

- **类型：** repo / 官方源码和架构说明
- **作者：** Pascal / pascalorg
- **代码：** https://github.com/pascalorg/editor
- **产品：** https://editor.pascal.app
- **核对版本：** [78aaf93277e2392909560481e3c311f1cb765cfd](https://github.com/pascalorg/editor/tree/78aaf93277e2392909560481e3c311f1cb765cfd)
- **核对日期：** 2026-11-08
- **核对范围：** 已通读根 README 和 docs/architecture.md；核对 packages/cli/README.md、packages/mcp/README.md 中本地运行、持久化、连接约束，阅读 packages/mcp/src/bin/pascal-mcp.ts、packages/mcp/src/server.ts 中可执行入口/注册工具逻辑，抽查 packages/core/src/store/use-scene.ts。未构建或实测 CLI。
- **通读：** 已完成
- **开放状态：** 开源编辑器、Viewer/Core/Nodes/CLI/MCP/IFC converter，仓库 LICENSE 为 MIT；托管工作区服务的账号/费用条件与开源本地运行分开。
- **归入 wiki：** [Pascal Editor](../../wiki/entities/pascal-editor.md)
- **官网资料归档：** [Pascal 官方文档](../sites/pascal-editor-docs.md)

## 架构与源码证据

- Turborepo：`apps/editor` 为 Next.js 宿主；`packages/core` 的 Zustand scene graph，`packages/viewer` 的 React Three Fiber/Three.js 渲染，`packages/editor` 的工具和面板，`packages/nodes` 的内建节点。
- `packages/mcp/src/bin/pascal-mcp.ts` 通过 `SceneBridge` + 本地 `SceneStore` 构建 `createPascalMcpServer`，按 `--http`/默认 stdio 连接；`packages/mcp/src/server.ts` 注册 scene tools、resources、prompts。
- 官方架构图的数据流：交互 → tool handler → `useScene.createNode/updateNode` → dirty 标记 → NodeRenderer + registry → 每帧 system 更新 geometry。
- CLI `npx @pascal-app/cli editor` 把 editor 和认证的本地 MCP 服务启动在动态 loopback 端口；本地项目位于 `~/.pascal/data/pascal.db`；Web runtime 首次下载并校验摘要。
- 本地 MCP 与托管 `https://editor.pascal.app/api/mcp` 为两种独立工作区入口，不能误以为免鉴权访问托管项目。

## 建筑语义与仿真边界

- 场景节点：Site → Building → Level → Wall/Slab/Roof/Zone/Item/Scan/Guide；以 `parentId` 组织扁平字典。
- 坐标：右手系，X/Z 地面、Y 向上，单位 m；导出至 ROS/URDF/MuJoCo/Isaac 前必须转换坐标和碰撞、惯性等物理数据。
- 官方 README 列出 `@pascal-app/ifc-converter` 的 IFC→Pascal nodes 支持；不能推断所有 IFC 语义无损转换。

## 一手链接

- [根 README](https://github.com/pascalorg/editor/blob/78aaf93277e2392909560481e3c311f1cb765cfd/README.md)
- [架构说明](https://github.com/pascalorg/editor/blob/78aaf93277e2392909560481e3c311f1cb765cfd/docs/architecture.md)
- [CLI / MCP 本地入口](https://github.com/pascalorg/editor/blob/78aaf93277e2392909560481e3c311f1cb765cfd/packages/mcp/src/bin/pascal-mcp.ts)
- [MCP 注册入口](https://github.com/pascalorg/editor/blob/78aaf93277e2392909560481e3c311f1cb765cfd/packages/mcp/src/server.ts)
