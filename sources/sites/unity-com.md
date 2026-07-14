# Unity 官网与 Unity Engine 产品页（unity.com）

- **类型**：网站 / 产品营销页 + 引擎产品说明
- **入口**：
  - 官网首页：<https://unity.com/>
  - Unity Engine 产品页：<https://unity.com/products/unity-engine>
  - 下载：<https://unity.com/download>
  - 许可与方案：<https://unity.com/products>
- **主体**：Unity Technologies
- **收录日期**：2026-07-14
- **抓取说明**：以 **2026-07-14** 对首页与 Engine 产品页的公开文案为准；版本号与 Beta 功能以 [Unity Manual 6.5](https://docs.unity3d.com/Manual/index.html) 交叉核对。

## 一句话

**Unity** 是全球使用最广的 **实时 3D 游戏引擎** 之一，提供 **Unity Editor**、**C# / .NET 脚本**、跨平台构建与 **Unity Gaming Services（UGS）** 增长/变现工具链；在机器人研究与工程中更常作为 **视觉渲染、交互演示与部分仿真客户端宿主**（如 [Flightmare](../../wiki/entities/flightmare.md)、AirSim Unity 分支、[MuJoCo Unity 插件](../repos/mujoco.md)），而非默认替代 [MuJoCo](../../wiki/entities/mujoco.md) / [Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md) 的 **控制级万并行 RL 物理后端**。

## 为什么值得保留

- 知识库中 [Flightmare](../../wiki/entities/flightmare.md)、[AirSim](../../wiki/entities/airsim.md)、[OpenLoong](../../wiki/entities/openloong.md) Unity RL Playground、[MuJoCo](../repos/mujoco.md) Unity 绑定等均已引用 Unity，但此前缺少对 **引擎本体与官方文档入口** 的一手溯源页。
- 官网与 Engine 页明确 **Unity 6 / 6.5** 代际、**Unity AI**（Assistant / Generators / MCP / AI Gateway）、**多平台部署** 与 **Unity Industry** 行业线，影响仿真可视化与数字孪生选型。
- 与 [Unreal Engine 5](../../wiki/entities/unreal-engine-5.md) 形成 **双主流实时 3D 宿主** 对照：Unity 偏 **C# 生态、移动/XR 覆盖广、研究型轻量客户端**；UE 偏 **影视级渲染管线与 Chaos 物理叙事**。

## 官网公开要点（2026-07-14）

### 定位

> Unity® is the world's leading game engine, supported by the most successful game development community in history and powered by a system that ensures each decision is informed by what players love.

> Build great — Develop, deploy, and grow your game in one place, on your terms.

### 三阶段产品叙事（Develop → Deploy → Grow）

| 阶段 | 公开摘要 |
|------|----------|
| **Develop** | 2D/3D 任意风格；全球最大开发者社区之一；庞大 Asset Store 生态 |
| **Deploy** | 桌面、iOS、Android、Switch、PlayStation、Xbox、Meta Quest、Web、Apple Vision Pro 等 **25+ 平台**；内置玩家洞察优化留存 |
| **Grow** | 获客、变现（Ads、IAP、Economy）、LiveOps 与 **Unity Dashboard** 运营闭环 |

### 最新动态（首页 Featured）

| 条目 | 摘要 |
|------|------|
| **Unity AI Beta（Unity 6）** | 编辑器内置 **Unity-tuned agent**，或通过 **AI Gateway** 与 **MCP Server** 安全连接外部 AI 工具 |
| **Unity 6.5** | 2D、图形、Shader、光照等改进（与 Manual 6.5 对齐） |
| **IAP 5.4 / D2C** | Dashboard 无代码 Webshop，直接向玩家直销 |

### Unity Industry

官网单列 **Unity Industry**：汽车、制造、零售、医疗等 **非游戏实时 3D 应用**（与游戏引擎共用 Editor 与渲染栈）。

## Unity Engine 产品页要点

### 定位

> Build 2D and 3D experiences in any style, for any platform. The Unity engine gives you the power and flexibility to realize your creative vision.

### 当前版本（产品页，2026-07-14）

| 项目 | 说明 |
|------|------|
| **最新主线** | **Unity 6** 已发布；稳定性、支持与性能改进 |
| **文档对齐** | Manual 标注 **Unity 6.5（6000.5）** 为当前 User Manual 版本 |
| **历史版本** | Release archive 提供各 Editor 安装包与 Release Notes |

### Engine features（产品页八大模块）

| 模块 | 公开摘要 |
|------|----------|
| **Flexibility** | 任意风格游戏/应用；任意平台；项目规模一视同仁 |
| **Working in Unity** | 新建项目默认 Sample Scene（Camera + Light） |
| **Interaction** | 角色动画、音频与沉浸式交互工具 |
| **Graphics and visuals** | 实时光照、着色与渲染优化；移动至高配 PC 可扩展 |
| **Scripting** | **.NET + C#**；Visual Studio / JetBrains Rider；从单物体到数千实体 |
| **Performance** | Profiler 等工具链，尽早发现性能瓶颈 |
| **Multiplayer** | 可扩展多人体验与社区连接（配合 UGS Multiplayer 文档） |
| **Collaboration** | 集中式工作流与实时协作 |
| **LiveOps** | 可扩展 LiveOps 生态保持玩家活跃 |

### 官方学习路径（Engine 页 Tutorials）

| 路径 | 受众 |
|------|------|
| **Essentials Pathway** | Unity 零基础入门 |
| **Junior Programmer Pathway** | 从零到可就业级 C# 编程 |
| **Creative Core Pathway** | 引擎创意向核心能力进阶 |

### 开发者资源入口

| 资源 | 链接 |
|------|------|
| **Documentation** | 全功能 Manual + Scripting API（见 [unity-manual-6-5.md](./unity-manual-6-5.md)） |
| **Advanced guides** | C#、2D 美术、动画、移动/PC/主机/VR 性能优化 |
| **Demos** | 官方样例与生态演示项目 |

## 对 wiki 的映射

| 主题 | 目标 wiki |
|------|-----------|
| Unity 引擎本体、与机器人仿真栈关系 | `wiki/entities/unity-engine.md` |
| UE5 对照宿主引擎 | `wiki/entities/unreal-engine-5.md` |
| Unity 渲染客户端四旋翼仿真 | `wiki/entities/flightmare.md` |
| Unity 分支视觉 UAV 仿真 | `wiki/entities/airsim.md` |
| MuJoCo 官方 Unity 插件 | `wiki/entities/mujoco.md` |
| OpenLoong Unity RL Playground | `wiki/entities/openloong.md` |

## 参考链接

- <https://unity.com/>
- <https://unity.com/products/unity-engine>
- <https://unity.com/download>
- <https://unity.com/products/unity-industry>
- <https://learn.unity.com/>
- <https://discussions.unity.com/>
