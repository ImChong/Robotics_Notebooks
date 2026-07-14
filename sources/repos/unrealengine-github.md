# Unreal Engine 源码仓库（EpicGames/UnrealEngine）

- **类型**：repo（私有）
- **入口**：<https://github.com/EpicGames/UnrealEngine/tree/ue5-main>
- **主体**：Epic Games, Inc.
- **收录日期**：2026-07-14
- **抓取说明**：仓库对未授权 GitHub 账号 **不可见**；以下基于 Epic 公开文档、Signup 指引与 Release Notes 对 `ue5-main` 分支的说明编译，**非源码摘录**。

## 一句话

**EpicGames/UnrealEngine** 是 Unreal Engine 的 **官方 C++ 源码主仓库**；`ue5-main` 为 UE5 系列开发主线，需 Epic 账号 + 源码许可关联 GitHub 后方可访问，社区改进可经授权 fork 向 Epic 提交。

## 为什么值得保留

- 工程团队自定义插件、修改 Chaos 物理、集成机器人中间件（ROS bridge、传感器插件）时，常需明确 **「闭源二进制 Launcher」与「可 fork 的 ue5-main」** 的分工。
- UE 5.8 Release Notes 写明包含 **GitHub 社区贡献**；维护者应知道贡献路径依赖本仓库许可，而非仅使用预编译编辑器。
- 与 [epicgames-github-org.md](./epicgames-github-org.md) 的公开工具仓互补。

## 访问与分支（公开信息）

| 项目 | 说明 |
|------|------|
| 可见性 | **Private** — 未关联 Epic 许可的 GitHub 用户 API 返回 404 |
| 推荐分支 | `ue5-main` — UE5 主线开发（用户指定入口） |
| 获取步骤 | 1) 注册 Epic Games 账号；2) 关联 GitHub；3) 接受 Unreal Engine 源码 EULA；4) 按 [Signup](https://github.com/epicgames/Signup) 指引授权访问 |
| 发布形态 | 里程碑对应 **Launcher 二进制发行**（如 5.8）；源码分支持续前进 |

## 与机器人研究相关的典型源码树（概念级，非逐文件清单）

维护者在获得源码后，常查阅的子系统路径包括：

| 子系统 | 典型用途 |
|--------|----------|
| `Engine/Source/Runtime/Engine` | 核心 Actor、Component、World 循环 |
| `Engine/Plugins/Experimental/PhysicsControl` | 物理驱动动画与控制 |
| `Engine/Source/Runtime/Experimental/Chaos` | Chaos 刚体/破坏/布料求解 |
| `Engine/Plugins/Runtime/AR` / 传感器插件 | AR、相机与自定义传感器扩展 |
| 项目级 Game Module | AirSim / CARLA / SPEAR 等 **上层仿真器** 多以 **UE 模块或插件** 形式挂载于引擎发行版之上 |

> 本知识库 **不** 镜像引擎源码；仿真选型见 [SPEAR](../../wiki/entities/spear-sim.md)、[AirSim](../../wiki/entities/airsim.md) 等 **基于 UE 的研究栈** 实体页。

## UE 5.8 与源码主线关系

- **5.8** 为公开 **二进制 + 文档** 里程碑；`ue5-main` 在发布后仍继续前进（含 Nanite 半透明等主线实验特性，论坛讨论见 Release 周期）。
- Epic 称 **5.8 为计划内最后一个 UE5 主版本**（保留 5.9 选项），同时推进 **UE6** Early Access（目标 2027 年底）。

## 对 wiki 的映射

| 主题 | 目标 wiki |
|------|-----------|
| UE5 引擎本体、源码 vs 二进制 | `wiki/entities/unreal-engine-5.md` |

## 参考链接

- <https://github.com/EpicGames/UnrealEngine>（需授权）
- <https://github.com/EpicGames/UnrealEngine/tree/ue5-main>（需授权）
- <https://github.com/epicgames/Signup>
- <https://dev.epicgames.com/documentation/unreal-engine/unreal-engine-5-8-release-notes>
