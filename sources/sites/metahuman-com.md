# MetaHuman 官网（metahuman.com）

- **类型**：网站 / 产品主页（Epic Games / Unreal Engine 生态）
- **入口**：<https://www.metahuman.com/>
- **主体**：Epic Games, Inc.
- **收录日期**：2026-06-19
- **抓取说明**：以 **2026-06-19** 对首页、`/create`、`/animate`、`/uses`、`/download` 及新闻 **MetaHuman 5.8** 公开文案的抓取为准；版本号与插件列表会随 UE 发布周期更新。

## 一句话

**MetaHuman** 是 Epic 面向 **高保真数字人（digital human）** 的创作与动画平台：在 **Unreal Engine** 内通过 **MetaHuman Creator** 与 **MetaHuman Animator** 快速生成带毛发与服装、已绑定的写实角色，并支持单相机无标记动捕表演驱动；**5.8** 起核心 **RigLogic / DNA** 库以 **MIT** 开源（OpenRigLogic / MetaHuman Devkit）。

## 为什么值得保留

- 机器人知识库中 **人体参考运动、遥操作化身、数字孪生可视化** 常与 **DCC / 游戏引擎角色管线** 交叉，但此前缺少对 MetaHuman 这一主流数字人栈的独立溯源页。
- 与 [Mixamo](./mixamo.md)（Adobe 在线角色库）、[Blender](./blender-org.md)（开源 DCC）形成 **资产创作 → 表演捕捉 → 引擎渲染** 链路上的对照节点。
- **5.8** 将 **面部工作流扩展到全身**、推出 **Mesh to MetaHuman（全身）**、**MetaHuman Crowds** 与 **OpenRigLogic**，对「视频/单相机 → 人体表演 → 下游重定向或可视化」有直接影响。

## 公开产品叙事（编译自官网，2026-06-19）

### 定位

> High-fidelity digital humans made easy.  
> Create and animate photorealistic digital humans, fully rigged and complete with hair and clothing, in minutes.

面向影视、游戏、实时交互等任意需要 **可信人类角色** 的 3D 项目；强调与 **任意引擎或渲染器** 配合时的人类可信度与情绪表达。

### 两大主工作流

| 模块 | 公开能力摘要 |
|------|----------------|
| **Create**（`/create`） | 从 MetaHuman 数据库组装角色；**Mesh to MetaHuman** 将外部网格/扫描/生成模型转为 MetaHuman 拓扑与 rig；可在其他 DCC 中适配与扩展；Fab 上买卖成品 MetaHuman 与配件 |
| **Animate**（`/animate`） | **MetaHuman Animator**：摄像头或麦克风驱动 **实时面部**；离线求解达制作级面部动画；角色 **开箱即用全身 mocap 或关键帧**；5.8 起 **单离机位相机** 同时捕捉 **面部 + 全身**（Experimental，集成 Meshcapade 无标记动捕插件） |

### 下载与生态（`/download`）

- **MetaHuman Creator / Animator** 随 **Unreal Engine** 发布（需从 UE 官网获取最新版）。
- **DCC 插件**：MetaHuman for **Maya**、**Houdini**、**Marvelous Designer** 等。
- **Animator 插件**：Depth Processing、**Markerless Motion Capture**（Fab，Windows）等。
- **Groom Starter / Advanced Kit**（Maya、Houdini）。

### MetaHuman 5.8 要点（新闻稿 2026-06-18）

| 主题 | 摘要 |
|------|------|
| **MetaHuman Crowds**（Experimental） | 新资产类型 **Collections**：移动端数百、高端平台数千角色；近景高保真 Actor 与远景 **ISK Instanced Skinned Meshes** 按相机距离切换；Mass 编排、Nanite（可用时）、可缩放面部法线贴图 |
| **Mesh to MetaHuman（全身）** | _creator 内集成_：任意拓扑人形网格（扫描、DCC、Meshy/Tripo 等生成）→ 单次工作流生成 **全身 MetaHuman 拓扑与 rig**（支持风格化） |
| **Animator 全身无标记** | 单相机离线处理面部/身体/同步；无需头盔相机或标记点；集成 **Meshcapade** Markerless Motion Capture |
| **OpenRigLogic / Devkit** | **RigLogic** 与 **DNA** 库 GitHub **MIT** 开源，供第三方应用保持与 Creator/Animator 兼容 |
| **其他** | 未烘焙纹理导出与覆盖；Creator 内自定义灯光场景预览（Lumen）；Animator 求解质量、音频驱动眨眼与情绪、**Linux/macOS** 平台支持；Live Link Face 实时视频回传 UE |

## 对 wiki 的映射

- 升格页面：[wiki/entities/metahuman.md](../../wiki/entities/metahuman.md)
- 交叉引用：[wiki/concepts/motion-retargeting.md](../../wiki/concepts/motion-retargeting.md)（工具表）、[wiki/entities/mixamo.md](../../wiki/entities/mixamo.md)、[wiki/entities/blender.md](../../wiki/entities/blender.md)

## 参考链接

- 官网：<https://www.metahuman.com/>
- 创建：<https://www.metahuman.com/create>
- 动画：<https://www.metahuman.com/animate>
- 应用场景：<https://www.metahuman.com/uses>
- 下载：<https://www.metahuman.com/download>
- 许可说明：<https://www.metahuman.com/license>
- 5.8 发布：<https://www.metahuman.com/news/metahuman-5-8-is-now-available>
- OpenRigLogic：<https://github.com/EpicGames/OpenRigLogic>（以仓库当前状态为准）
- Unreal Engine：<https://www.unrealengine.com/>
