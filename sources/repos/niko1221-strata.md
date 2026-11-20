# Niko1221/Strata：本地大模型推理引擎

- **类型：** repo
- **原始链接：** https://github.com/Niko1221/Strata
- **入库日期：** 2026-11-20
- **核对范围：** README 全文、terminal/strata.py CLI 入口及 terminal/ 目录结构
- **通读：** 已完成

## 原始资料要点（归纳，非全文转载）

- 免费开源本地推理系统，提供桌面与终端应用，服务本地模型聊天、编码工具连接等；不是长期记忆管理器。
- 官方 README 指向 Windows/Linux/macOS 安装与模型选型，2026-11-20 核对页面展示 v0.2.16。
- terminal/strata.py 在 main 中转调 strata_cli.cli.main；终端项目下存在 serve/ 推理服务目录。
- PolyStrata 的官方 README 明确说明它在特定 Qwen3.8 模型上复用 Strata 引擎，其他模型另用 PolyStrata 自有引擎。

## 对 wiki 的映射

- [Niko1221/Strata：本地大模型推理引擎](../../wiki/entities/niko-strata-local-inference.md)
