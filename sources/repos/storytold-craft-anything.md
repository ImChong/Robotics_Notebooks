# Craft Anything

- **类型：** 开源软件分析与重建工具
- **仓库：** https://github.com/storytold/craft-anything
- **核对版本：** [5430c5a7](https://github.com/storytold/craft-anything/tree/5430c5a7c647439e42305d3780686847e61bb789)
- **许可证：** MIT 或 Apache-2.0；素材与品牌另有条款
- **入库日期：** 2026-11-04
- **开放状态：** 源码公开，README 提供桌面端、CLI 和 MCP 入口；项目标为 early alpha。

## 要点

官方 README 描述 Study → Build → Check → Ready 工作流。默认 Clean room 根据可见信息分析，Deep 模式允许读取内部代码和文件。构建依赖 coding agent 或 AI 后端；官方列出 Claude Code 的实际运行支持，部分其他后端仍待实测。网站检查覆盖多个视口；应用与游戏的行为检查仍不完整。

## 阅读范围与待办

已阅读 README，查看 CLI 的 `rebuild_cmd.rs` 及 `crates/ai/src/rebuild/mod.rs` 的主调用链；尚未完整核查各目标类型的捕获和检查实现，也未实机运行。详见 [实体节点](../../wiki/entities/craft-anything.md)。
