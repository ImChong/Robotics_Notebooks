# Redot Engine（Redot-Engine/redot-engine）

> 来源归档

- **标题：** Redot Engine — Multi-platform 2D and 3D game engine
- **类型：** repo
- **来源：** Redot community / Redot-Engine
- **链接：** <https://github.com/Redot-Engine/redot-engine>
- **分支：** master
- **入库日期：** 2026-10-06
- **核查日期：** 2026-10-06
- **一句话说明：** Redot 官方引擎源码仓库，包含 C++ 引擎、编辑器、平台构建脚本、许可证和版本发布记录。
- **项目页：** [Redot 官方项目页与文档](../sites/redotengine-docs.md)
- **沉淀到 wiki：** [Redot Engine 实体页](../../wiki/entities/redot-engine.md)

## 仓库定位

README 将 Redot 定义为从 Godot 分叉而来的社区驱动 2D/3D 游戏引擎。仓库记载分叉发生于 2024 年 9 月，目标是独立维护与社区驱动开发。源码树包含引擎核心、编辑器、平台服务器、驱动、模块、测试与第三方依赖。

## 构建与下载

- **二进制：** 编辑器和导出模板可从 Redot 官网或 GitHub Releases 获取。
- **源码构建：** README 指向官方逐平台编译文档；仓库提供 Nix 工作流示例 `nix run .`，缺少二进制时会安装构建依赖并编译。
- **重要依赖：** 发布构建需要与引擎版本匹配的导出模板；Web、移动端等目标需核对平台特定文档和插件兼容性。

## 许可证与发布

- **引擎源码：** MIT；LICENSE.txt 保留 Redot 与 Godot 贡献者版权声明。
- **当前版本快照（2026-10-06）：** 最新正式版 Redot LTS 26.2，发布于 2026-06-30；最新候选版 Redot 26.3-rc.2，发布于 2026-09-30。
- **运行范围：** 官方 FAQ 列出桌面编辑器与桌面、Android、iOS、Web 导出路径。控制台导出因厂商许可限制，官方团队不提供开源模板。
- **第三方组件：** 仓库版权和许可文件提示引擎包含其他许可的第三方组件，分发二进制或修改版时应检查完整许可清单。

## 官方入口

- 项目说明：[README.md](https://github.com/Redot-Engine/redot-engine/blob/master/README.md)
- MIT 许可证：[LICENSE.txt](https://github.com/Redot-Engine/redot-engine/blob/master/LICENSE.txt)
- 发布历史：[GitHub Releases](https://github.com/Redot-Engine/redot-engine/releases)
- 构建说明：[官方编译文档](https://docs.redotengine.org/contributing/development/compiling/)

## 对 wiki 的映射

- 项目页：[redotengine-docs.md](../sites/redotengine-docs.md)
- 实体页：[redot-engine.md](../../wiki/entities/redot-engine.md)
