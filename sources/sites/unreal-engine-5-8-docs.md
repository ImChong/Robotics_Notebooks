# Unreal Engine 5.8 官方文档（Epic Developer Community）

- **类型**：网站 / 官方技术文档
- **入口**：<https://dev.epicgames.com/documentation/unreal-engine/unreal-engine-5-8-documentation>
- **主体**：Epic Games, Inc.
- **收录日期**：2026-07-14
- **抓取说明**：以 **2026-07-14** 对文档根页侧栏索引与 [5.8 Release Notes](https://dev.epicgames.com/documentation/unreal-engine/unreal-engine-5-8-release-notes) 公开摘要的抓取为准；Experimental / Beta 功能以 Release Notes 版本标记为准。

## 一句话

Epic **Unreal Engine 5.8 文档**是 UE5 当前主线的 **工程工作流总索引**：覆盖基础入门、内容管线、虚拟世界构建、渲染、AI 插件、动画、Gameplay、移动端、音频/媒体、测试优化与发布，并链到各子系统 Release Notes。

## 为什么值得保留

- [unreal-engine-5-com.md](./unreal-engine-5-com.md) 侧重 **产品营销与发布会新闻**；本页为 **编辑器功能、渲染与物理子系统** 的一手工程索引。
- 5.8 Release Notes 明确 **MCP 插件**、**Chaos / Dataflow**、**Mesh Terrain**、**MetaHuman Crowds**、**Lumen Lite** 等条目，直接服务机器人栈中「UE 作视觉后端 / 数字人 / 场景编辑」的维护与选型。
- 文档根页列出 **22** 个一级主题入口，便于 agent 与人类维护者按模块深挖，而非只停留在营销特性名。

## 文档根页一级目录（2026-07-14）

| 模块 | 链接 | 摘要 |
|------|------|------|
| What's New | [whats-new](https://dev.epicgames.com/documentation/unreal-engine/whats-new) | 各版本新特性 |
| Understanding the Basics | [understanding-the-basics-of-unreal-engine](https://dev.epicgames.com/documentation/unreal-engine/understanding-the-basics-of-unreal-engine) | 入门必备概念与技能 |
| Working with Content | [working-with-content-in-unreal-engine](https://dev.epicgames.com/documentation/unreal-engine/working-with-content-in-unreal-engine) | 外部 DCC 资产导入与设置 |
| Building Virtual Worlds | [building-virtual-worlds-in-unreal-engine](https://dev.epicgames.com/documentation/unreal-engine/building-virtual-worlds-in-unreal-engine) | 交互环境与关卡工具 |
| Designing Visuals, Rendering, and Graphics | [designing-visuals-rendering-and-graphics-with-unreal-engine](https://dev.epicgames.com/documentation/unreal-engine/designing-visuals-rendering-and-graphics-with-unreal-engine) | 光照、材质、VFX、后处理 |
| AI Features, Tools, and Plugins | [ai-features-tools-and-plugins-in-unreal-engine](https://dev.epicgames.com/documentation/unreal-engine/ai-features-tools-and-plugins-in-unreal-engine) | 引擎内 AI 特性与插件 |
| Creating Visual Effects | [creating-visual-effects-in-niagara-for-unreal-engine](https://dev.epicgames.com/documentation/unreal-engine/creating-visual-effects-in-niagara-for-unreal-engine) | Niagara 粒子系统 |
| Gameplay Tutorials | [gameplay-tutorials-for-unreal-engine](https://dev.epicgames.com/documentation/unreal-engine/gameplay-tutorials-for-unreal-engine) | 常见玩法元素教程 |
| Blueprints Visual Scripting | [blueprints-visual-scripting-in-unreal-engine](https://dev.epicgames.com/documentation/unreal-engine/blueprints-visual-scripting-in-unreal-engine) | 可视化脚本 |
| Programming with C++ | [programming-with-cplusplus-in-unreal-engine](https://dev.epicgames.com/documentation/unreal-engine/programming-with-cplusplus-in-unreal-engine) | C++ 编程 |
| Gameplay Systems | [gameplay-systems-in-unreal-engine](https://dev.epicgames.com/documentation/unreal-engine/gameplay-systems-in-unreal-engine) | 玩法机制与响应式世界 |
| Mobile Development | [getting-started-with-mobile-development-in-unreal-engine](https://dev.epicgames.com/documentation/unreal-engine/getting-started-with-mobile-development-in-unreal-engine) | 移动平台开发 |
| Animating Characters and Objects | [animating-characters-and-objects-in-unreal-engine](https://dev.epicgames.com/documentation/unreal-engine/animating-characters-and-objects-in-unreal-engine) | 2D/3D 角色与物体动画 |
| Motion Design | [motion-design-in-unreal-engine](https://dev.epicgames.com/documentation/unreal-engine/motion-design-in-unreal-engine) | 广播级动态图形 |
| Creating User Interfaces | [creating-user-interfaces-with-umg-and-slate-in-unreal-engine](https://dev.epicgames.com/documentation/unreal-engine/creating-user-interfaces-with-umg-and-slate-in-unreal-engine) | UMG / Slate UI |
| Working with Audio | [working-with-audio-in-unreal-engine](https://dev.epicgames.com/documentation/unreal-engine/working-with-audio-in-unreal-engine) | 音频工具 |
| Working with Media | [working-with-media-in-unreal-engine](https://dev.epicgames.com/documentation/unreal-engine/working-with-media-in-unreal-engine) | 线性与虚拟制作媒体 |
| Setting Up Your Production Pipeline | [setting-up-your-production-pipeline-in-unreal-engine](https://dev.epicgames.com/documentation/unreal-engine/setting-up-your-production-pipeline-in-unreal-engine) | 生产管线效率工具 |
| Testing and Optimizing Your Content | [testing-and-optimizing-your-content](https://dev.epicgames.com/documentation/unreal-engine/testing-and-optimizing-your-content) | 性能与质量验证 |
| Sharing and Releasing Projects | [sharing-and-releasing-projects-for-unreal-engine](https://dev.epicgames.com/documentation/unreal-engine/sharing-and-releasing-projects-for-unreal-engine) | 多平台发布 |
| Samples and Tutorials | [samples-and-tutorials-for-unreal-engine](https://dev.epicgames.com/documentation/unreal-engine/samples-and-tutorials-for-unreal-engine) | 样例场景与教程 |

## Unreal Engine 5.8 Release Notes 摘录（机器人/仿真相关）

> The release of Unreal Engine 5.8 brings further improvements to the UE5 toolset. This release delivers improvements in a wide variety of areas, including Rendering, Character and Animation, Worldbuilding, PCG, and more.

### 渲染

| 条目 | 状态 | 要点 |
|------|------|------|
| **MegaLights** | Production Ready | 降噪与性能改进，目标 60 fps；新增场景调试/优化工具 |
| **Lumen Lite** | Beta | 中等质量 GI（Irradiance Fields + Probe Occlusion），约为 High Quality 两倍速度，面向 Switch 2 / 低端 PC 60 fps |
| **Substrate NPR Shading** | Experimental | 基于 Substrate 的风格化 NPR 渲染 |

详见 [Lumen Performance Guide](https://dev.epicgames.com/documentation/unreal-engine/lumen-performance-guide-for-unreal-engine)。

### 角色、动画与 MetaHuman

- **Control Rig Physics** → Beta：力驱动动画、与现有动画分层合成。
- **Control Rig Dynamics**（Experimental）：轻量粒子求解器，适合毛发/布料/配饰等 **游戏内 cosmetic 物理**。
- **MetaHuman Crowds**（Experimental）：Mass + Collections，高保真个体与远景 ISK 切换。
- **Mesh to MetaHuman（全身）**：任意拓扑人形网格 → MetaHuman 拓扑与 rig。
- **MetaHuman Animator**：单相机无标记 **面部+全身**（Experimental）；Linux/macOS 面部离线/实时路径扩展。

### 世界构建

| 条目 | 状态 | 要点 |
|------|------|------|
| **Mesh Terrain** | Experimental | 网格地形：3D 雕刻、洞穴/悬挑、Nanite、虚拟纹理；旨在突破 heightfield Landscape 限制 |
| **Procedural Vegetation Editor (PVE)** | Experimental | 编辑器内生长 Nanite 植被，支持风场 |

### 物理与仿真（Chaos）

| 子系统 | 5.8 要点 |
|--------|----------|
| **Chaos Destruction** | Dataflow 非破坏性 Geometry Collection 工作流 |
| **Chaos Dataflow** | Cloth / Destruction / Hair / Flesh / MetaHuman 等程序化物理资产节点图；**Production Ready**（Cloth Panel Editor） |
| **Chaos Visual Debugger (CVD)** | 编辑器内或独立实例调试物理模拟 |
| **Chaos Core** | 向 Job Graph 迁移、Chaos++ solver Dataflow 节点 |
| **Chaos Modular Vehicles** | 动态 Scene Graph 组件拼装车辆，客户端预测 + 服务端权威物理 |
| **Mover + ChaosMover** | 运动预测、网络与动画驱动移动框架持续演进 |

### 框架与开发者工具

| 条目 | 状态 | 要点 |
|------|------|------|
| **MCP Server** | Experimental | **Model Context Protocol** 插件，使 agentic AI 连接 Unreal Editor，理解项目并协助构建资产/系统 |
| **Iris** | Production Ready | 新一代网络复制系统 |
| **Mass Framework** | — | 大规模实体模拟框架扩展（人群、NPC 等） |
| **PCG** | — | 程序化内容生成性能与 UX 改进 |
| **Movie Render Graph** | Production Ready | 离线高质量渲染管线 |

### 社区贡献

Release Notes 注明本版本包含 GitHub 社区开发者提交的改进（需关联 Epic 账号与源码许可方可向引擎主分支贡献）。

## 对 wiki 的映射

| 主题 | 目标 wiki |
|------|-----------|
| UE5/5.8 引擎与 Chaos/MCP/渲染 | `wiki/entities/unreal-engine-5.md` |
| MetaHuman 子系统细节 | `wiki/entities/metahuman.md` |
| 程序化地形概念 | `wiki/concepts/procedural-terrain-generation.md` |

## 参考链接

- <https://dev.epicgames.com/documentation/unreal-engine/unreal-engine-5-8-documentation>
- <https://dev.epicgames.com/documentation/unreal-engine/unreal-engine-5-8-release-notes>
- <https://dev.epicgames.com/documentation/unreal-engine/lumen-performance-guide-for-unreal-engine>
