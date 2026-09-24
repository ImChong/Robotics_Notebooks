# KeenTools for Blender（官方 add-on 仓库）

- **标题**：KeenTools for Blender
- **类型**：repo
- **来源**：KeenTools
- **链接**：<https://github.com/KeenTools/keentools-blender>
- **克隆**：`https://github.com/KeenTools/keentools-blender.git`
- **入库日期**：2026-09-24
- **许可**：GPL-3.0（add-on 源码）
- **一句话说明**：KeenTools 官方 **Blender 插件** 开源仓库，提供 **FaceBuilder**、**FaceTracker**、**GeoTracker** 等在 Blender 内的 UI 与集成层；**几何/跟踪核心算法在闭源 KeenTools Core Library** 中。
- **沉淀到 wiki：** 是 → [`wiki/entities/keentools-facebuilder.md`](../../wiki/entities/keentools-facebuilder.md)

## 仓库概况（2026-09-24）

| 字段 | 值 |
|------|-----|
| 默认分支 | `dev` |
| 主要语言 | Python |
| Stars | ~160+ |
| Topics | blender, blender-addon, facebuilder, geotracker, keentools |
| 创建 | 2020-06-11 |

## 开源边界（必读）

| 层级 | 许可 / 获取方式 |
|------|-----------------|
| **本仓库（add-on）** | **GPLv3**，源码完整公开 |
| **KeenTools Core Library** | **闭源**；订阅或试用；联网可自动安装；见 <https://keentools.io/download/core> |
| **Cloud API** | 商业 API，与本地 Core 分离 |

> FAQ（[产品页](../sites/keentools-facebuilder-blender.md)）：*The add-on is an open source adapter for KeenTools Core Library.*

## README 摘要

> KeenTools for Blender is a blender addon that allows using some of the KeenTools functionality in Blender.

插件通过 Blender **N-panel（Sidebar）** 暴露 FaceBuilder / FaceTracker / GeoTracker 等功能；安装流程为下载 KeenTools Blender Pack ZIP → Blender Preferences → Install add-on → 在线或离线安装 Core Library。

## 与机器人研究/工程的关联点

- **人类化身外观**：少图构建写实头部，供遥操作 UI、数字孪生、演示视频中的 **operator avatar** 使用。
- **MetaHuman 管线**：导出 **MH texture** + 网格，对接 Epic **Mesh to MetaHuman**（见 [keentools-fbb-metahuman.md](../sites/keentools-fbb-metahuman.md)）。
- **面部表演**：**FaceTracker** + **ARKit FACS blendshape** + **Live Link Face** — 与 [MetaHuman Animator](../../wiki/entities/metahuman.md) 的 Live Link 生态相邻，但拓扑与输出格式不同。
- **非机器人运动学标准**：输出为 **头部 mesh / 纹理 / blendshape**，不是 URDF/MJCF 人体模型；接入控制栈前需单独重定向或仅作视觉层。

## 对 wiki 的映射

- [wiki/entities/keentools-facebuilder.md](../../wiki/entities/keentools-facebuilder.md)
- [wiki/entities/blender.md](../../wiki/entities/blender.md)
- [wiki/entities/metahuman.md](../../wiki/entities/metahuman.md)

## 参考链接

- 仓库：<https://github.com/KeenTools/keentools-blender>
- KeenTools 官网：<https://keentools.io/>
- FaceBuilder 产品页：<https://keentools.io/products/facebuilder-for-blender>
