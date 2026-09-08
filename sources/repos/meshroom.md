# Meshroom（alicevision/Meshroom）

> 来源归档

- **标题：** Meshroom — Node-based Visual Programming Toolbox
- **类型：** repo
- **来源：** AliceVision 社区（GitHub 组织 `alicevision`）
- **链接：** https://github.com/alicevision/Meshroom
- **项目页：** http://meshroom.org — [`sources/sites/meshroom-org.md`](../sites/meshroom-org.md)
- **后端算法库：** https://github.com/alicevision/AliceVision
- **克隆：** `https://github.com/alicevision/Meshroom.git`
- **许可：** MPL-2.0
- **版本（入库时）：** 2025.1 系列（README / meshroom.org 新闻；GitHub `develop` 分支持续开发）
- **入库日期：** 2026-09-08
- **一句话说明：** 开源节点式视觉编程框架：用图（Graph）编排多视图摄影测量、相机跟踪、HDR 全景、纹理与插件化 ML 节点；默认捆绑 AliceVision 摄影测量管线，并可通过 MeshroomHub 插件扩展至 3DGS（MrGSplat）等。
- **开源状态：** **已开源** — MPL-2.0；提供 [预编译二进制](https://github.com/alicevision/meshroom/releases)；Python 插件 API + 命令行节点封装。
- **沉淀到 wiki：** 是 → [`wiki/entities/meshroom.md`](../../wiki/entities/meshroom.md)

## 仓库概况（2026-09-08 GitHub API / README）

| 字段 | 值 |
|------|-----|
| 托管 | GitHub（`alicevision/Meshroom`） |
| 默认分支 | `develop` |
| 主要语言 | Python |
| Stars / Forks | ~12.9k / ~1.2k |
| Topics | `3d-reconstruction`, `alicevision`, `camera-tracking`, `computer-vision`, `hdr-imaging`, `image-stitching`, `meshroom`, `multi-view-stereo`, `photogrammetry`, `structure-from-motion`, `texturing`, `workflow-automation` |
| 手册 | https://meshroom-manual.readthedocs.io |
| FAQ | https://github.com/alicevision/meshroom/wiki |

## 为何值得保留

- **机器人 Real2Sim 上游「场景采集 → 几何/纹理」栈。** 多视图照片经 SfM + MVS 可导出 mesh / 稠密点云 / 相机位姿，再喂仿真（USD/Omniverse）、3DGS 训练（[MrGSplat](https://github.com/meshroomHub/mrGSplat)）或质检浏览；与 [GS-Playground](../../wiki/entities/gs-playground.md) 强调的 **仿真内批量 3DGS 渲染** 形成上下游分工。
- **节点图 + 增量缓存是工程可维护性样板。** 改参数只失效下游节点、复用中间缓存；支持本地与 render farm 分布式执行——适合长管线摄影测量与批量数据集生产。
- **插件生态覆盖 CV/ML 全链。** 默认 AliceVision 插件（摄影测量、HDR、全景、光度立体等）；MeshroomHub 提供分割（mrSegmentation）、单目深度（mrDepthEstimation）、RoMa 匹配、**3D Gaussian Splatting（mrGSplat）**、地理配准（mrGeolocation）等。

## README / 手册要点（归纳）

### 核心概念

| 概念 | 说明 |
|------|------|
| **Graph** | 互连节点集合，定义完整数据处理工作流 |
| **Node** | 基本运算单元；边表示数据流 |
| **Attribute** | 节点参数；修改后仅失效下游，保留上游缓存 |
| **Template** | 插件提供的现成管线模板，可自定义保存 |
| **Local / Renderfarm** | 本地或农场并行；节点锁与日志/资源统计 |

### UI 分区

- **Graph Editor** — 编排管线
- **Node Editor** — Attributes / Log / Statistics / Status / Documentation / Notes
- **2D & 3D Viewer** — 图像与三维结果预览
- **Image Gallery** — 输入文件列表

### 默认捆绑插件（AliceVision）

- **多视图摄影测量** — SfM + MVS → 相机位姿、稠密点云、网格、纹理（[管线概览](http://alicevision.github.io/#photogrammetry)）
- **相机跟踪** — 视频相机运动估计
- **HDR 融合** — 多曝光包围摄影
- **全景拼接** — 含鱼眼与电动云台
- **光度立体 / 多视图光度立体** — 单视图或多视图几何增强

### MeshroomHub 扩展（节选）

| 插件 | 能力 |
|------|------|
| [mrGSplat](https://github.com/meshroomHub/mrGSplat) | 多视图图像 → 3D Gaussian Splatting；与 AliceVision 摄影测量管线衔接 |
| [mrSegmentation](https://github.com/meshroomHub/mrSegmentation) | 自然语言提示的 AI 分割 |
| [mrDepthEstimation](https://github.com/meshroomHub/mrDepthEstimation) | 单目深度估计 |
| [mrRoma](https://github.com/meshroomHub/mrRoma) | RoMa 稠密特征匹配 |
| [mrGeolocation](https://github.com/meshroomHub/mrGeolocation) | GPS + OSM / 高程 / 法国 IGN Lidar 地理上下文 |
| [MeshroomMicMac](https://github.com/alicevision/MeshroomMicMac) | IGN MicMac 摄影测量算法桥接（探索性） |

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 实体页（主） | [`wiki/entities/meshroom.md`](../../wiki/entities/meshroom.md) |
| Real2Sim / 3DGS 下游 | [`wiki/entities/gs-playground.md`](../../wiki/entities/gs-playground.md)、[`wiki/entities/spark-3dgs-renderer.md`](../../wiki/entities/spark-3dgs-renderer.md) |
| Real2Sim 概念 | [`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md)、[`wiki/methods/crisp-real2sim.md`](../../wiki/methods/crisp-real2sim.md) |
| 生成式世界 / 3D 资产 | [`wiki/methods/generative-world-models.md`](../../wiki/methods/generative-world-models.md) |
| 项目页归档 | [`sources/sites/meshroom-org.md`](../sites/meshroom-org.md) |

## 参考链接

- 源码仓库：<https://github.com/alicevision/Meshroom>
- AliceVision 算法库：<https://github.com/alicevision/AliceVision>
- 官方站点：<http://meshroom.org>
- 手册：<https://meshroom-manual.readthedocs.io>
- 预编译发布：<https://github.com/alicevision/meshroom/releases>
- 插件索引：<https://github.com/meshroomHub>
