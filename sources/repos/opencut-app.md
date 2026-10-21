# OpenCut（OpenCut-app/OpenCut）

- **类型：** GitHub 仓库
- **仓库：** <https://github.com/OpenCut-app/OpenCut>
- **官网：** <https://opencut.app>
- **重写版预览：** <https://new.opencut.app>
- **默认分支：** `main`
- **源码状态：** 已公开；官方仓库可访问
- **许可证：** MIT
- **核验日期：** 2026-10-16
- **一句话说明：** OpenCut 是面向 Web、桌面与移动端的开源视频编辑器项目；当前主仓库正在重写架构，计划以共享 Rust 核心支持多端并扩展插件与 AI agent 接口。

## 项目状态

官方 README 将该仓库标记为从头重写中。计划包括 Editor API、第三方插件、共享 Rust 核心、MCP server、无界面批量渲染和编辑器内脚本功能。README 指出，当前可直接使用的旧版本仍由 [OpenCut 官网](https://opencut.app) 提供；重写版预览位于 [new.opencut.app](https://new.opencut.app)。

旧版代码位于 [opencut-classic](https://github.com/OpenCut-app/opencut-classic)，GitHub 将其标记为 archived。使用者应区分当前在线编辑器与进行中的重写版，避免把路线图功能误认为已经全部交付。

## 复现与贡献边界

- 新版仓库提供 web、api 与 desktop 开发入口；README 指引使用 Proto 安装锁定工具，并通过 Moon 执行开发任务。
- 项目当前表示架构仍在设计中，暂未开放外部代码贡献。
- 用户提供的功能描述提及时间轴、蒙版、贴纸与导出；具体功能以当前在线版本为准，重写版规划以官方 README 为准。

## 关联资料

- [OpenCut 官方网站与版本状态](../sites/opencut.md)
- [OpenCut GitHub README](https://github.com/OpenCut-app/OpenCut/blob/main/README.md)
- [旧版源码仓库](https://github.com/OpenCut-app/opencut-classic)
