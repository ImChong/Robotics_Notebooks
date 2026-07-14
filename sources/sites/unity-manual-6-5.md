# Unity 6.5 User Manual 与 Unity Docs 门户

- **类型**：网站 / 官方技术文档
- **入口**：
  - **Unity Manual（引擎）**：<https://docs.unity3d.com/Manual/index.html>
  - **Unity Docs 统一门户（含中文）**：<https://docs.unity.com/zh-cn>（locale `zh-CN`；产品文档按命名空间拆分）
  - **Unity Docs 根索引（llms.txt）**：<https://docs.unity.com/llms.txt>
  - **Unity AI 文档**：<https://docs.unity.com/ai/llms.txt>
- **主体**：Unity Technologies
- **收录日期**：2026-07-14
- **抓取说明**：Manual 以 **2026-07-14** 页眉 **Unity 6.5（6000.5）** 与 Featured content 为准；`docs.unity.com/zh-cn` 为 Next.js 统一文档站（与 legacy `docs.unity3d.com/Manual` 并存，引擎 Manual 仍以 unity3d.com 为主干）。

## 一句话

**Unity 6.5 User Manual** 是 Unity Editor 的 **工程工作流总索引**：覆盖渲染、物理、动画、脚本、XR、包管理与 Unity Services；**docs.unity.com** 则聚合 **UGS（Grow/Multiplayer/Cloud）** 与 **Unity AI** 等产品线文档，并支持 **中文 locale** 入口。

## 为什么值得保留

- [unity-com.md](./unity-com.md) 侧重 **产品营销与 Engine 特性名**；本页为 **编辑器子系统、包与服务文档** 的一手工程索引。
- Manual 首页 **Highlights of Unity 6** 明确 **渲染性能、多人、多平台、Unity AI、视觉特效** 等模块链接，直接服务机器人栈中「Unity 作视觉/交互宿主」的维护与版本锁定。
- `docs.unity.com/llms.txt` 列出 **60+** 产品文档入口（Ads、Cloud Save、Multiplayer、Vivox、Asset Transformer 等），便于 agent 按产品线深挖，避免与引擎 Manual 混淆。

## Unity 6.5 Manual 版本信息（2026-07-14）

| 字段 | 值 |
|------|-----|
| **文档版本** | Unity 6.5（6000.5） |
| **语言** | 默认 English；portal 支持 zh-CN 等 locale |
| **升级入口** | Manual 链到 *New in Unity* 与 *Upgrade Unity* |

## Manual 首页 Highlights（Unity 6 家族）

| 主题 | Manual 路径 | 摘要 |
|------|-------------|------|
| **Boost rendering performance** | [render-pipelines](https://docs.unity3d.com/Manual/render-pipelines.html) | URP/HDRP 等可扩展渲染；光照与 VFX 最新进展 |
| **Multiplayer game creation** | [multiplayer](https://docs.unity3d.com/Manual/multiplayer.html) | 官方多人包与服务简化联机开发 |
| **Expand multiplatform reach** | [PlatformSpecific](https://docs.unity3d.com/Manual/PlatformSpecific.html) | 移动浏览器优化 runtime 与各平台新特性 |
| **Unlock possibilities with Unity AI** | [unity-ai](https://docs.unity3d.com/Manual/unity-ai.html) | 编辑器与 Dashboard 内 AI：代码生成、资产生成、运行时推理 |
| **Achieve more engaging visuals** | render-pipelines | Lighting、Shader Graph、VFX Graph 更新 |
| **Enhance productivity** | features（Cinemachine 等外链） | Profiler、ProBuilder、Cinemachine、UI Toolkit |

## Manual Featured content（一级模块）

| 模块 | 链接 | 机器人/仿真注记 |
|------|------|-----------------|
| **Unity AI** | [unity-ai](https://docs.unity3d.com/Manual/unity-ai.html) | Assistant 上下文帮助、Generators 资产生成、**Sentis** 运行时模型推理 |
| **Animation** | [AnimationSection](https://docs.unity3d.com/Manual/AnimationSection.html) | Mecanim、Timeline；人体表演 → 重定向链路上游 |
| **Audio** | [Audio](https://docs.unity3d.com/Manual/Audio.html) | 多模态仿真次要通道 |
| **Building Blocks** | [building-blocks](https://docs.unity3d.com/Manual/building-blocks.html) | 官方最佳实践玩法积木 |
| **2D** | [Unity2D](https://docs.unity3d.com/Manual/Unity2D.html) | 2D 物理与玩法（机器人栈较少用） |
| **Lighting** | [LightingOverview](https://docs.unity3d.com/Manual/LightingOverview.html) | 实时光照；视觉 Sim2Real 域随机化 |
| **Multiplayer** | [multiplayer](https://docs.unity3d.com/Manual/multiplayer.html) | 多人包与服务 |
| **Package management** | [PackagesList](https://docs.unity3d.com/Manual/PackagesList.html) | 扩展 Editor 与运行时能力 |
| **Physics** | [PhysicsSection](https://docs.unity3d.com/Manual/PhysicsSection.html) | 3D 刚体/关节/碰撞仿真（**非**足式接触金标准） |
| **Platform development** | [PlatformSpecific](https://docs.unity3d.com/Manual/PlatformSpecific.html) | 各构建目标配置 |
| **Rendering** | [render-pipelines](https://docs.unity3d.com/Manual/render-pipelines.html) | URP/HDRP/Built-in 选型 |
| **Scripting** | [scripting](https://docs.unity3d.com/Manual/scripting.html) | C# 游戏逻辑与 Editor 扩展 |
| **UI** | [UIToolkits](https://docs.unity3d.com/Manual/UIToolkits.html) | UI Toolkit |
| **Unity services** | [UnityServices](https://docs.unity3d.com/Manual/UnityServices.html) | Monetization、Cloud Build、Multiplayer 等 |
| **Visual effects** | [visual-effects](https://docs.unity3d.com/Manual/visual-effects.html) | VFX Graph、后处理 |
| **XR** | [XR](https://docs.unity3d.com/Manual/XR.html) | AR/MR/VR；遥操作与沉浸式数据采集 |

## docs.unity.com 门户结构（llms.txt 摘要）

`docs.unity.com` 按 **产品线** 拆分文档（非全部等于引擎 Manual）：

| 类别 | 代表产品文档 | 说明 |
|------|--------------|------|
| **创作加速** | [Unity AI](https://docs.unity.com/ai/llms.txt) | Dashboard 配置、Unity Credits、编辑器 AI 工具 |
| **增长与变现** | Ads、IAP、Economy、Grow | 商业化与 LiveOps |
| **多人与云** | Multiplayer、Relay、Lobby、Matchmaker、Cloud Save | UGS 后端服务 |
| **工具链** | Hub、Version Control、Build Automation、CLI | 安装、协作与 CI |
| **跨引擎** | Authentication/Matchmaker/Vivox for Unreal | Unity 服务亦可挂接 UE 项目 |

中文入口 **<https://docs.unity.com/zh-cn>** 与英文 **<https://docs.unity.com/en-us>** 共享同一文档 CDN（`cdn.docs.unity.com`），locale 由路由与 polyfill 决定。

## 对 wiki 的映射

| 主题 | 目标 wiki |
|------|-----------|
| Unity 引擎与文档入口总览 | `wiki/entities/unity-engine.md` |
| 产品营销与 Engine 特性 | [unity-com.md](./unity-com.md) |
| UE5 文档对照 | `wiki/entities/unreal-engine-5.md` |

## 参考链接

- <https://docs.unity3d.com/Manual/index.html>
- <https://docs.unity3d.com/ScriptReference/index.html>
- <https://docs.unity.com/zh-cn>
- <https://docs.unity.com/llms.txt>
- <https://docs.unity.com/ai/llms.txt>
- <https://unity.com/releases/editor/qa/lts-releases>
