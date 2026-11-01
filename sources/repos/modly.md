# Modly 官方仓库

> 来源归档（repo）

- **标题：** Modly — Local AI 3D Model Generator
- **类型：** repo / image-to-3d / mesh / desktop / local-ai
- **维护者：** Lightning Pixel
- **代码：** <https://github.com/lightningpixel/modly>（应用代码已开源，MIT）
- **官网：** <https://modly3d.app/>
- **入库日期：** 2026-10-28
- **一句话说明：** 基于本地模型扩展，从图片生成三维网格的桌面应用。

## 核查与入口

README 提供 Windows、Linux 和 Apple Silicon macOS 安装包与源码启动步骤。应用使用 JavaScript 前端和 Python 后端；工作流示例为 Image → Generate Mesh → Add to Scene。官方扩展包括 Hunyuan3D 2 Mini、TripoSG 和 Trellis2 GGUF，扩展从 GitHub 安装，模型权重按需下载。模型和扩展须分别核查许可与显存需求。

CLI 可连接运行中的桌面应用，示例：`python tools/modly-cli/agent.py generate --image ./input.png --output ./export.glb`。README 将独立启动 API 的开发辅助路径与完整桌面桥接区分开。

## 对 wiki 的映射

- [Modly](../../wiki/entities/modly.md)
- [官网归档](../sites/modly.md)
