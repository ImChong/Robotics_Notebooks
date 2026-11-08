# Craft Anything

- **类型：** 开源软件分析与重建工具（Rust；桌面端、CLI、MCP）
- **项目主页：** https://getartcraft.com/apps/craft-anything
- **官方源码：** https://github.com/storytold/craft-anything
- **核对版本：** [56aa3197](https://github.com/storytold/craft-anything/tree/56aa31976b4ff0950e6b34ba020b0a568d2462fa)（2026-11-08 检索时 main）
- **许可证：** MIT 或 Apache-2.0 二选一；品牌和内含资产遵循独立条款
- **入库日期：** 2026-11-04；本次复核：2026-11-08
- **开放状态：** 官方代码与公开 Releases 可访问，属于 early alpha；Claude Code 的端到端重建流程已支持，其他后端/行为检查有部分待验证。
- **核对范围：** 已逐节通读官方 README.md 全文（工作流、模式、安装、CLI、已实现/局限、许可）；阅读 AGENTS.md 的架构与规范、crates/engine/src/clean_room.rs，核对 apps/craft-anything-cli/src/rebuild_cmd.rs 全文、crates/ai/src/rebuild/mod.rs 的 run/check 主循环与 crates/ai/src/rebuild/study.rs 的 Study 主流程；对 docs/STATUS.md 进行定向核查。未完整审计全部工具及平台实现。
- **通读：** 已完成

## 原始资料（官方一手）

1. [README：完整项目说明](https://github.com/storytold/craft-anything/blob/56aa31976b4ff0950e6b34ba020b0a568d2462fa/README.md)。
2. [AGENTS.md：分层架构、工具/evidence/clean-room 约束](https://github.com/storytold/craft-anything/blob/56aa31976b4ff0950e6b34ba020b0a568d2462fa/AGENTS.md)。
3. [CLI 入口：rebuild_cmd.rs](https://github.com/storytold/craft-anything/blob/56aa31976b4ff0950e6b34ba020b0a568d2462fa/apps/craft-anything-cli/src/rebuild_cmd.rs)。
4. [重建主循环：rebuild/mod.rs](https://github.com/storytold/craft-anything/blob/56aa31976b4ff0950e6b34ba020b0a568d2462fa/crates/ai/src/rebuild/mod.rs)。
5. [目标检查：study.rs](https://github.com/storytold/craft-anything/blob/56aa31976b4ff0950e6b34ba020b0a568d2462fa/crates/ai/src/rebuild/study.rs)。
6. [Clean room 允许的工具表：clean_room.rs](https://github.com/storytold/craft-anything/blob/56aa31976b4ff0950e6b34ba020b0a568d2462fa/crates/engine/src/clean_room.rs)。
7. [官方状态说明](https://github.com/storytold/craft-anything/blob/56aa31976b4ff0950e6b34ba020b0a568d2462fa/docs/STATUS.md)与[工具状态](https://github.com/storytold/craft-anything/blob/56aa31976b4ff0950e6b34ba020b0a568d2462fa/docs/tool-status.md)。

## 原始资料要点（忠实摘录后的归纳）

- 官方工作流是 **Study → Build → Check → Ready/Refine**。先收集目标证据，再交给 coding agent 实现，在选定检查器下比较差异并按轮次修正。
- **Clean room（默认）** 只依据用户/访客可见的资料与受允许的外部观察工具；**Deep** 可读取内部源码与文件。该访问限制不等于操作系统级沙箱，coding agent 权限仍由其自身环境控制。
- **入口** 分为 egui 桌面界面、`craft-anything-cli` 和 MCP server，底层共用 Rust 引擎及证据系统。官网当前 README 列有 320 工具，但工具数随版本变化，勿当作常量。
- **网站案例**：`study.rs` 调用 `web_site_open`、`web_site_capture`、`web_page_states_capture`，并捕获相关页面；构建后按选定视口和页面检查布局/视觉差异。应用与游戏行为检查尚有不完整范围。
- **源码路径**：CLI 的 `run` 创建 `Workspace`/`RunOptions` 并调用 `RebuildRun::start`；`Worker::run` 识别目标类型、执行 `study`、驱动 builder、使用 `checks_with_a_retry` 检查，并在阈值未达且轮次允许时调用 `kind.refine`。
- **使用限制**：官方声明 early alpha；应仅分析自己所有或获许可检查的软件；不应假设重建相似度意味着功能等价、合法复制或安全隔离。

## 核查边界

已完成主要原文 README 的通读和关键源码路径复核，但**没有在本环境构建/运行 Craft Anything**，未逐项测试跨平台行为、具体检查器准确性或每一种 coding agent 后端。结果是文献与代码层面的核查，而非实测性能结论。

## 对 wiki 的映射

- [Craft Anything：软件证据驱动的分析与重建](../../wiki/entities/craft-anything.md)。
