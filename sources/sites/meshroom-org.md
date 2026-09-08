# Meshroom 官方站点（meshroom.org）

> 来源归档

- **标题：** Meshroom — Node-based Visual Programming Toolbox
- **类型：** site / project-page
- **URL：** <http://meshroom.org>
- **代码：** <https://github.com/alicevision/Meshroom> — [`sources/repos/meshroom.md`](../repos/meshroom.md)
- **维护方：** AliceVision 社区
- **入库日期：** 2026-09-08
- **一句话说明：** Meshroom 官方主页：节点式数据处理工具箱定位、新闻（2025.1 插件架构、考古数字化、MicMacRoom 等）与文档入口。

## 开源核查（步骤 2.5，截至 2026-09-08）

| 核查项 | 结论 |
|--------|------|
| 项目页是否链到代码 | **是** — 页内与 README 均指向 `github.com/alicevision/Meshroom` |
| 预编译二进制 | **有** — <https://github.com/alicevision/meshroom/releases> |
| 可运行实现 | **有** — 官方 release 可直接运行 GUI；源码见 INSTALL.md |
| 训练/推理权重 | **视插件而定** — 如 mrSegmentation / mrDepthEstimation 等插件自带或下载模型；核心摄影测量不依赖闭源权重 |
| 综合判定 | **已开源**（MPL-2.0） |

## 页面要点（2026-09-08 首页）

- **定位：** 开源节点式视觉编程框架，用于创建、管理与执行复杂数据处理管线。
- **增量计算：** 修改节点属性仅失效下游，复用缓存中间结果。
- **执行模式：** 本地 + render farm 分布式并行；内置 2D/3D 可视化。
- **2025.1 新闻：** 统一插件架构；AliceVision 插件整合摄影测量、相机跟踪、HDR 全景、Lidar meshing、RAW 转换与色彩校准等标准 CV 管线。
- **生态新闻：** MicMacRoom（MicMac × Meshroom）、mrHelloWorld 插件教程、法国国家考古博物馆数字化工具、Cap Digital 开源分享等。

## 关联资料

- 仓库归档：[`sources/repos/meshroom.md`](../repos/meshroom.md)
- Wiki：[`wiki/entities/meshroom.md`](../../wiki/entities/meshroom.md)
- AliceVision 算法：<https://github.com/alicevision/AliceVision>
- 手册：<https://meshroom-manual.readthedocs.io>
