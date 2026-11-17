# OpenDesign（nexu-io/open-design）：官方代码仓库

> 一手资料归档 · repo · 2026-11-17

- **标题：** OpenDesign — Collaborative AI Design Agent Workspace（协作式 AI 设计智能体工作区）
- **类型：** repo / coding-agent / design-workspace
- **仓库：** https://github.com/nexu-io/open-design
- **项目官网：** https://open-design.ai/
- **许可证：** Apache-2.0（仓库 README 与 LICENSE）
- **开源状态：** **已开源**。可公开查阅产品源码、skills、design-templates、design-systems、plugins，以及 Web / daemon / Electron 代码；OpenDesign Cloud 为独立的托管模型服务，不能据代码开放推断云额度免费。
- **核对日期：** 2026-11-17
- **核对版本：** GitHub `main` 分支，截至本次入库的公开文档与源码（随上游继续演进）。
- **核对范围：** 通读官方 `README.md`（产品、平台兼容、源码安装、设计系统、插件、架构、路线与许可）、`QUICKSTART.md`（Docker、本地开发、Local CLI/BYOK 两种执行路径）和 `docs/architecture.md`（进程分层、HTTP/SSE、文件与权限边界）；另外核对 `apps/daemon/src/runtimes/registry.ts`、`apps/web/src/providers/daemon.ts` 与 `apps/daemon/src/routes/runs.ts` 的接口及部分入口代码。后者为定点阅读，未审计全部源码及运行测试。
- **通读：** 已完成
- **一句话说明：** 用可替换 Coding Agent、`DESIGN.md` 设计系统和可预览的项目文件，把需求 brief 连接到可协作修改的原型与可导出产物。

## 原文核对要点

1. **四类可组合资源：** `skills/` 是执行中可调用的能力，`design-templates/` 是可渲染起点，`design-systems/` 以 `DESIGN.md` 描述视觉语言，`plugins/` 提供工作流/扩展；各层职责不同。
2. **本地 CLI：** Web/Electron 发起项目运行，由 Express daemon 根据 runtime registry 启动已安装的 coding-agent CLI；文件型运行写项目文件，SSE 事件更新工作区与预览。对应 `apps/web/src/providers/daemon.ts`、`apps/daemon/src/routes/runs.ts`、`apps/daemon/src/runtimes/registry.ts`。
3. **BYOK / plain-stream：** 当运行时没有文件工具，使用完整 `<artifact>` 文本块交接，宿主解析并生成沙箱预览；与本地 CLI 直接写文件的模式不能混为一谈。
4. **技术结构：** Next.js Web UI、Node/Express daemon、SQLite、CLI runtime adapter、Electron desktop；主要通信为 HTTP + SSE，而不是旧设计稿中的 WebSocket。
5. **运行入口：** `corepack enable && pnpm install && pnpm tools-dev run web`（从源码）；Docker 配置参见 `QUICKSTART.md`，公开部署须设置访问鉴权、来源与反向代理 SSE 策略。
6. **平台边界：** 官方 README 列出 macOS/Windows 预构建桌面版本；Linux 当前需要从源码运行；本身不是 CAD、机器人仿真器或 URDF/MJCF 工具。

## 主要技术入口

- [README](https://github.com/nexu-io/open-design/blob/main/README.md)
- [QUICKSTART](https://github.com/nexu-io/open-design/blob/main/QUICKSTART.md)
- [Architecture](https://github.com/nexu-io/open-design/blob/main/docs/architecture.md)
- [Web daemon provider](https://github.com/nexu-io/open-design/blob/main/apps/web/src/providers/daemon.ts)
- [Daemon run routes](https://github.com/nexu-io/open-design/blob/main/apps/daemon/src/routes/runs.ts)
- [Runtime registry](https://github.com/nexu-io/open-design/blob/main/apps/daemon/src/runtimes/registry.ts)
- [License](https://github.com/nexu-io/open-design/blob/main/LICENSE)

## 关联资料

- [官网来源](../sites/open-design-ai.md)
- [Wiki 实体](../../wiki/entities/open-design.md)
