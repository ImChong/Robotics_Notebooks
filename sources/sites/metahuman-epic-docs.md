# MetaHuman 官方文档（Epic Developer Community）

- **类型**：网站 / 官方技术文档
- **入口**：<https://dev.epicgames.com/documentation/metahuman/metahuman-documentation>
- **主体**：Epic Games, Inc.
- **收录日期**：2026-06-19
- **抓取说明**：以 **2026-06-19** 对文档根页及侧栏所列 **15** 篇 MetaHuman 文档公开摘要的抓取为准；UE 版本号与 Experimental 功能以 Release Notes 为准。

## 一句话

Epic **MetaHuman 官方文档**说明如何在 **Unreal Engine 5** 内 **创建、动画驱动与使用** 完全绑定的高保真数字人，并索引 Creator、Animator、Capture、Crowds、Devkit、DCC 插件与 Fab 市场等子系统。

## 为什么值得保留

- [metahuman.com](./metahuman-com.md) 侧重 **产品营销与发布新闻**；本页为 **工程工作流与插件/API 索引** 的一手来源。
- 文档明确 **Animator 三种生成路径**（实时单目/移动设备/音频、离线深度、离线单目视频）、**UE Cine / Optimized / UEFN Export** 装配管线，以及 **MetaHuman Facial Description Standard**（Control Curves 规范）。
- 与机器人栈的交叉点：**表演捕捉指南**、**无标记单目视频处理**、**Python 脚本自动化**、**OpenRigLogic Devkit**——可作为数字人表演 → 下游重定向的文档锚点。

## 文档根页定义（编译）

> MetaHuman is a complete framework that gives any creator the power to create, animate and use fully rigged, photorealistic digital humans in a variety of projects powered by Unreal Engine 5.

## 文档目录（侧栏，2026-06-19）

| 文档 | 摘要 |
|------|------|
| [MetaHuman Documentation](https://dev.epicgames.com/documentation/metahuman/metahuman-documentation) | 总入口与框架概述 |
| [What's New](https://dev.epicgames.com/documentation/metahuman/whats-new) | 各版本新特性与 Beta/Experimental 说明 |
| [MetaHuman Creator Overview](https://dev.epicgames.com/documentation/metahuman/metahuman-creator-overview) | 环境搭建、硬件要求、插件总览、术语表、数据使用政策 |
| [MetaHuman Creator](https://dev.epicgames.com/documentation/metahuman/metahuman-creator) | 在 UE 内组装面部/身体/发型/服装/材质，输出可动画游戏级角色 |
| [MetaHuman Animator](https://dev.epicgames.com/documentation/metahuman/metahuman-animator) | 从视频/音频/深度数据生成 MetaHuman 动画 |
| [MetaHuman Capture](https://dev.epicgames.com/documentation/metahuman/metahuman-capture) | iOS/Android **Live Link Face** 安装与初始设置 |
| [MetaHumans in Unreal Engine](https://dev.epicgames.com/documentation/metahuman/metahumans-in-unreal-engine) | **UE Cine / UE Optimized** 管线装配后于关卡中使用角色 Blueprint |
| [MetaHuman Crowds in Unreal Engine](https://dev.epicgames.com/documentation/metahuman/metahuman-crowds-in-unreal-engine) | **UE 5.8 Experimental**：Collections 大规模人群、LOD 与动画 |
| [MetaHuman Devkit in Unreal Engine](https://dev.epicgames.com/documentation/metahuman/metahuman-devkit-in-unreal-engine) | Devkit 技术集合；含 **OpenRigLogic** 章节 |
| [MetaHuman for Maya](https://dev.epicgames.com/documentation/metahuman/metahuman-for-maya) | 在 Maya 中使用与编辑 MetaHuman 角色部分 |
| [MetaHuman for Houdini](https://dev.epicgames.com/documentation/metahuman/metahuman-for-houdini) | HDA 组装角色或创作 MetaHuman 兼容 Groom，导出至 UE |
| [MetaHuman in UEFN](https://dev.epicgames.com/documentation/metahuman/metahuman-in-unreal-editor-for-fortnite) | **UEFN Export** 管线角色在 UEFN 关卡中使用 |
| [MetaHumans on Fab](https://dev.epicgames.com/documentation/metahuman/metahumans-on-fab) | Fab 市场买卖 MetaHuman 角色与兼容资产 |
| [MetaHuman Content Examples](https://dev.epicgames.com/documentation/metahuman/metahuman-content-examples) | 示例工程（如时装走秀关卡） |
| [MetaHuman Facial Description Standard](https://dev.epicgames.com/documentation/metahuman/mh-standards-docs) | Animator 驱动的 **Control Curves** 列表；烘焙 AnimSequence 或 Animator 导出的面部动画基础表示 |

## MetaHuman Animator 三种生成方式（文档摘录）

文档将动画生成分为三类：

1. **实时（Realtime）**：任意单目摄像机（含 webcam）、移动设备（Live Link Face）、或 **音频源**；需启用 MetaHuman Live Link 插件，在关卡中为角色指定 Live Link Subject。
2. **离线深度（Depth）**：iOS **TrueDepth** 或立体头盔相机（HMC）采集的深度数据。
3. **离线单目（Mono Video）**：单目视频或音频的离线处理；含 **无标记镜头（markerless footage）** 导入流程与 **音频驱动** Quickstart。

子主题还包括：**Performance Capture Guidelines**（身体与面部表演捕捉最佳实践）、**Mesh to MetaHuman**（从网格或视频创建 MetaHuman Identity）、**Python Scripting for MetaHuman Animator**、**Asset Reference**。

## Creator Overview 配套条目

- **Hardware Requirements** — 推荐硬件
- **Plugins Overview** — 所需 UE 插件一览
- **Recommended Asset Naming Conventions** — 资产命名约定
- **Terminology** — 术语表
- **MetaHuman Data Use** — 数据使用政策

## 装配管线（文档中的三种出口）

| 管线 | 文档页 | 用途 |
|------|--------|------|
| **UE Cine** | MetaHumans in UE | 影视级质量 |
| **UE Optimized** | MetaHumans in UE | 游戏/实时优化 |
| **UEFN Export** | MetaHuman in UEFN | Fortnite 创意生态 |

## 对 wiki 的映射

- 主实体页：[wiki/entities/metahuman.md](../../wiki/entities/metahuman.md)
- 营销站对照：[metahuman-com.md](./metahuman-com.md)

## 参考链接

- 文档根页：<https://dev.epicgames.com/documentation/metahuman/metahuman-documentation>
- Creator：<https://dev.epicgames.com/documentation/metahuman/metahuman-creator>
- Animator：<https://dev.epicgames.com/documentation/metahuman/metahuman-animator>
- Devkit / OpenRigLogic：<https://dev.epicgames.com/documentation/metahuman/metahuman-devkit-in-unreal-engine>
- 面部曲线标准：<https://dev.epicgames.com/documentation/metahuman/mh-standards-docs>
- 官网：<https://www.metahuman.com/>
