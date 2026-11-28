# URDF-Studio

- **标题**: URDF-Studio
- **链接**: https://github.com/OpenLegged/URDF-Studio
- **类型**: repo / workstation / design-tool
- **作者**: OpenLegged Community
- **许可证**: Apache-2.0
- **入库日期**: 2026-07-08
- **通读日期**: 2026-11-28
- **核对范围**: 通读 `OpenLegged/URDF-Studio`（v2.0.0）`README.md`、完整 `src/` 目录架构及核心管线源码 `src/app/App.tsx`、`src/core/parsers/importRobotFile.ts`（756 行）、`src/app/hooks/useFileExport.ts`（725 行）与 `src/features/file-io/export_formats.ts`。
- **通读：** 已完成
- **一句话说明**: OpenLegged 开源的统一模式 Web 机器人设计工作站（React 19.2 + R3F + Three.js 0.181 + Zustand 5 + Apache-2.0），通过可复用的 `src/core` 与 `@urdf-studio/react-robot-canvas` 支持 URDF/MJCF/Xacro/SDF/USD 多格式导入解析、在环编辑器与带 CSV BOM/PDF 报告的多格式打包导出。

## 核心内容摘要（2026-11-28 通读 v2.0.0 源码更新）

### 1. 当前架构与核心能力（v2.0.0）
- **统一 Single-mode Editor：** README 明确当前应用采用单模式统一编辑器（内含 **topology、geometry/collision/measurements、hardware configuration** 页签；早期“Skeleton/Detail/Hardware 三模式切换”表述已合并入统一编辑器面板）。
- **工程边界与分层（`app -> features -> store -> shared -> core -> types`）：**
  - `src/app/App.tsx`（`AppContent`）：编排 `useFileImport`、`useFileExport`、`useRobotLoadWorkflow`、`useAIWorkspaceSession`、`scheduleUsdRuntimeStartupIdlePrewarm` 与 `studioAgentPorts`；暴露测试与插件调试接口。
  - `src/core/parsers/importRobotFile.ts`：`resolveRobotFileData` 分派 `parseURDF`、`parseMJCF`/`parseCanonicalPhysicalMJCF`、`parseSDF`、`processXacroWithDiagnostics`，对 `.usd/.usda/.usdc` 先返回 `needs_hydration` 交由后台 USD WASM 运行时解析，并在 `finalizeImportedRobotData` 收尾标准化。
  - `src/app/hooks/useFileExport.ts`：整合 `addRobotAssetsToZip`、`generateRobotBomCsv`、`stripClosedLoopConstraintsForUrdfExport` 与源保留导出（Source-preserving export），打包支持 URDF、MJCF、USD（WASM 转换与 prepared export cache）、SDF、Xacro、CSV BOM、PDF 检验报告及 `.usp` 工程包。
  - **可发布子包与可选 AI 助手：** `@urdf-studio/react-robot-canvas@0.1.0` 供外部站点嵌入 3D 画布；AI 助手（OpenAI/Gemini BYOK）为可选增强，无 Key 时编辑器与全部导入导出功能完全独立可用。

### 2. 技术栈
- **React 19.2**, TypeScript 5.8, Vite 6.2, **Three.js 0.181**, **React Three Fiber 9**, Zustand 5, Tailwind CSS 4.1, Monaco Editor, JSZip, jsPDF, libarchive.js。

### 3. 独特价值
- **工作站定位**: 不仅仅是查看器，而是完整的设计创作环境，具备工作区和文件管理能力。
- **工程实用性**: 关注真实硬件工程需求，如电机库和 BOM 清单生成。
- **模块化设计**: 提供可复用的 `@urdf-studio/react-robot-canvas` 包。

## 对 Wiki 的映射
- **wiki/entities/urdf-studio.md** (新建)
- **wiki/entities/botworld.md** — URDF Studio 作为 BotWorld 内置工作区
- **references/repos/utilities.md** (更新)
