# Unreal Engine 5 产品主页（unrealengine.com）

- **类型**：网站 / 产品营销页
- **入口**：<https://www.unrealengine.com/unreal-engine-5>
- **主体**：Epic Games, Inc.
- **收录日期**：2026-07-14
- **抓取说明**：直连返回 Cloudflare 安全校验；以 **2026-07-14** 经 Jina Reader 镜像抓取首页公开文案，并与 [State of Unreal 2026](https://www.unrealengine.com/news/state-of-unreal-2026-top-news-from-the-show) 新闻稿交叉核对。

## 一句话

**Unreal Engine 5（UE5）** 是 Epic 面向游戏、影视、仿真与数字孪生等行业的 **下一代实时 3D 创作工具**：以 Nanite、Lumen、World Partition 等特性支撑大规模高保真世界，并内置角色动画、建模、程序化音频与跨行业工作流。

## 为什么值得保留

- 机器人知识库中 [AirSim](../repos/airsim.md)、[CARLA](../../wiki/entities/carla.md)、[SPEAR](../../wiki/entities/spear-sim.md)、[MetaHuman](./metahuman-com.md)、[MATRiX](../../wiki/entities/matrix-simulation-platform.md) 等均以 UE 为 **视觉/场景宿主**；此前缺少对 **引擎本体** 的一手溯源页。
- 官网给出 UE5 **代际特性叙事**（Nanite / Lumen / TSR / World Partition / Control Rig / MetaSounds），与工程文档、5.8 Release Notes 形成「产品定位 → 技术细节」链路。
- **State of Unreal 2026** 宣布 **UE 5.8 已发布**、**UE6 路线** 与 **MCP 插件**，影响后续仿真与 agentic 工具链选型。

## 公开产品叙事（编译）

### 定位

> Unreal Engine enables game developers and creators across industries to realize next-generation real-time 3D content and experiences with greater freedom, fidelity, and flexibility than ever before.

> The world's most open and advanced real-time 3D creation tool.

### 核心特性（官网 Key Features，UE5 代际）

| 特性 | 公开摘要 |
|------|----------|
| **Nanite + Virtual Shadow Maps** | 虚拟化微多边形几何；直接导入数百万面片网格并在 60 fps 级实时渲染中保持保真；按感知流式处理细节 |
| **Lumen** | 全动态全局光照与反射；间接光随直射光或几何变化实时适应；无需 lightmap 烘焙或反射捕获 |
| **Temporal Super Resolution (TSR)** | 内置高质量升采样，以较低内部分辨率渲染、输出接近高分辨率像素保真 |
| **World Partition** | 大世界自动网格划分与流式加载；**One File Per Actor** 支持多人同区协作；**Data Layers** 支持同空间日夜等变体 |
| **Characters & animation** | Control Rig、Sequencer、重定向；可用机器学习做实时变形；运行时按速度/地形调整动画 |
| **Modeling** | 编辑器内网格编辑、Geometry Scripting、UV、烘焙与属性 |
| **MetaSounds** | 可编程程序化音频 DSP 图，类比 Material Editor |

### 免费 UE5 样例工程（官网 Sample Projects）

| 样例 | 说明 |
|------|------|
| [Stack O Bot](https://fab.com/s/805a99aa100f) | 小型 sandbox，展示 UE5 新特性；配套 [Stack O Bot 学习路径](https://dev.epicgames.com/community/learning/paths/yG/stack-o-bot) |
| [Lyra Starter Game](https://fab.com/s/3fe3f994dd6d) | 与 UE5 同步演进的玩法样例，展示最佳实践 |
| [City Sample](https://fab.com/s/5e8f5eda64d8) | 《The Matrix Awakens》城市场景开源拆解；含建筑、车辆与 MetaHuman 人群 |

更多资产见 [Fab 市场](https://www.fab.com/)。

## State of Unreal 2026 要点（2026-06-17 新闻，与 UE5/5.8 相关）

| 主题 | 摘要 |
|------|------|
| **UE 5.8 发布** | 聚焦性能与核心特性成熟；Lumen 轻量变体支持 Switch 2 / PC 60 fps；**Mesh Terrain**（Experimental）突破 heightfield 地形限制 |
| **Production Ready（5.8）** | MegaLights、Audio Insights、Dataflow for Chaos Cloth、Live Link Hub、Iris、Movie Render Graph 等 |
| **MCP 插件（Experimental）** | UE 5.8 集成 **Model Context Protocol**，使 Claude / Gemini 等模型可作为理解 UE 工作流的协作方 |
| **UE6 路线** | 目标 2027 年底 Early Access；Gameplay 编程向 **Verse** 演进；开放标准互操作；继续强化渲染、迭代速度、移动平台 |
| **UE5 大版本节奏** | 5.8 为 **计划内最后一个 UE5 主版本**（保留 5.9 选项） |
| **Lore VCS** | Epic 开源下一代版本控制系统 [lore](https://github.com/epicgames/lore)，面向代码 + 大二进制资产协作 |

## 对 wiki 的映射

| 主题 | 目标 wiki |
|------|-----------|
| UE5 引擎本体、与机器人仿真栈关系 | `wiki/entities/unreal-engine-5.md` |
| MetaHuman / 数字人 | `wiki/entities/metahuman.md` |
| SPEAR / 通用 UE Python 仿真 | `wiki/entities/spear-sim.md` |
| AirSim / UAV 视觉仿真 | `wiki/entities/airsim.md` |
| CARLA / 自动驾驶城市仿真 | `wiki/entities/carla.md` |
| MATRiX / MuJoCo+UE5 联合仿真 | `wiki/entities/matrix-simulation-platform.md` |

## 参考链接

- <https://www.unrealengine.com/unreal-engine-5>
- <https://www.unrealengine.com/download>
- <https://www.unrealengine.com/news/state-of-unreal-2026-top-news-from-the-show>
- <https://www.unrealengine.com/news/unreal-engine-5-8-is-now-available>
