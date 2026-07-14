# Epic Games GitHub 组织（github.com/epicgames）

- **类型**：repo / 组织索引
- **入口**：<https://github.com/epicgames>
- **主体**：Epic Games, Inc.
- **收录日期**：2026-07-14
- **抓取说明**：以 **2026-07-14** GitHub API 公开组织仓库列表与 README 摘要为准；**UnrealEngine 主仓库为私有**，不在本组织公开列表中完整呈现引擎源码。

## 一句话

**Epic Games** 在 GitHub 公开 **工具链、中间件与生态配套仓库**（如 Pixel Streaming、MetaHuman DNA、Blender 互操作、Lore VCS），而 **完整 Unreal Engine 源码** 需 Epic 账号与许可协议后在独立私有仓库获取。

## 为什么值得保留

- 机器人/仿真工作流常需 **Pixel Streaming 远程可视化**、**MetaHuman DNA / RigLogic**、**DCC 导入插件** 等配套仓，而非仅引用闭源引擎本体。
- **State of Unreal 2026** 宣布 **Lore** 开源版本控制系统，组织页是追踪 Epic 工程文化变化的一手入口。
- 与 [unrealengine-github.md](./unrealengine-github.md) 区分：**组织公开仓** vs **引擎私有主仓**。

## 组织概况（2026-07-14）

| 字段 | 值 |
|------|-----|
| 组织 URL | <https://github.com/epicgames> |
| 公开定位 | 游戏引擎生态的开源工具、SDK 与示例 |
| 许可模式 | 各仓库独立（MIT、Epic 源码许可等） |

## 与 UE / 数字人 / 远程呈现相关的公开仓库（节选）

| 仓库 | Stars（约） | 说明 |
|------|------------|------|
| [lore](https://github.com/epicgames/lore) | ~7.9k | 下一代开源版本控制系统；面向代码 + 大二进制资产（State of Unreal 2026 重点发布） |
| [PixelStreamingInfrastructure](https://github.com/EpicGames/PixelStreamingInfrastructure) | ~519 | 官方 Pixel Streaming 前后端；浏览器远程驱动 UE 应用 |
| [MetaHuman-DNA-Calibration](https://github.com/EpicGames/MetaHuman-DNA-Calibration) | ~595 | MetaHuman DNA 校准相关工具 |
| [BlenderTools](https://github.com/EpicGames/BlenderTools) | ~3.2k | Blender ↔ Unreal 工作流插件 |
| [Signup](https://github.com/epicgames/Signup) | ~3.6k | Epic 账号注册与 **Unreal Engine 源码访问** 指引 |
| [raddebugger](https://github.com/epicgames/raddebugger) | ~7.2k | 原生多进程图形调试器（RAD 合作） |
| [unreal-engine-skills-for-claude-code-plugin](https://github.com/epicgames/unreal-engine-skills-for-claude-code-plugin) | ~133 | Claude Code 的 UE 技能插件（与 5.8 MCP 方向呼应） |

> **注**：`PixelStreamingInfrastructure-archived-2024` 已迁移至 [EpicGamesExt/PixelStreamingInfrastructure](https://github.com/EpicGamesExt/PixelStreamingInfrastructure)。

## Signup 仓库要点（引擎源码访问）

公开 README 说明：开发者可通过 Epic Games 账号关联 GitHub，在同意 **Unreal Engine EULA / 源码许可** 后访问 **私有** [EpicGames/UnrealEngine](https://github.com/EpicGames/UnrealEngine) 仓库（分支如 `ue5-main`）。未授权账号无法 clone 引擎本体。

## 对 wiki 的映射

| 主题 | 目标 wiki |
|------|-----------|
| UE5 引擎与开源生态边界 | `wiki/entities/unreal-engine-5.md` |
| MetaHuman DNA / Devkit | `wiki/entities/metahuman.md` |

## 参考链接

- <https://github.com/epicgames>
- <https://github.com/epicgames/Signup>
- <https://github.com/epicgames/lore>
