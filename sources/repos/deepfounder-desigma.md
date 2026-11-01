# Desigma（deepfounder-ai/desigma）

> 来源归档（repo；核对日期：2026-10-27）

- **仓库：** <https://github.com/deepfounder-ai/desigma>
- **类型：** 开源桌面界面设计工具
- **许可：** 仓库包含 Apache-2.0 和 MIT 许可文件。
- **开放状态：** 源码公开，仓库提供构建、测试命令和发行版本。
- **功能：** 本地优先的 Rust 桌面编辑器，支持自动布局、组件、原型、矢量编辑、协作，以及导入 Figma REST JSON 和 `.fig` 文件。
- **实现：** `desigma-doc` 负责文档模型，`desigma-engine` 负责编辑逻辑，`desigma-render` 使用 tiny-skia 渲染，`desigma-fig` 处理 Figma 导入，`desigma-app` 是桌面编辑器，`desigma-mcp` 提供本地 MCP 服务。
- **运行：** `cargo run --release -p desigma-app`；`cargo test --workspace` 执行测试。Linux 构建依赖窗口系统开发头文件，详见 README。
- **限制：** README 标记为 early alpha；列出协作、插件、PDF 导出、部分 `.fig` 特性和发行包签名等不足。演示文稿的初版 PDF 导出为栅格页，文字不可选中。

## 对 wiki 的映射

- [Desigma 实体页](../../wiki/entities/desigma.md)
