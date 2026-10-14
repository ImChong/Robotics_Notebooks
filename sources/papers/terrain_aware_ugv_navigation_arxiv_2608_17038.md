# Terrain-Aware Local Path Planning with Global DEM Data Integration for Autonomous UGV Navigation（arXiv:2608.17038）

> 来源归档（论文摘录 / ingest）

- **类型：** arXiv 预印本（cs.RO）
- **论文：** <https://arxiv.org/abs/2608.17038>
- **HTML：** <https://arxiv.org/html/2608.17038v1>
- **PDF：** <https://arxiv.org/pdf/2608.17038>
- **作者：** Devender Singh, Issah Nazif Suleiman, Paul Mitten, Glenn Cutler, Vinicius Prado da Fonseca, Matthew Hamilton
- **机构：** Memorial University of Newfoundland；Compusult Ltd.
- **提交日期：** 2026-08-17
- **许可：** arXiv 页面链接 CC BY-SA 4.0
- **代码：** 截至 2026-10-14，作者个人学术主页与实验室发表页均未链接本论文的可运行代码仓。
- **一句话说明：** 将 SRTM DEM 上的全局 A* 路径与 LiDAR 地面/障碍感知、坡度可通行性分析和在线路径修正结合，在 Gazebo 的 Nanuk UGV 仿真中评估。
- **沉淀到 wiki：** 是 → [论文实体页](../../wiki/entities/paper-terrain-aware-ugv-dem-local-planning.md)

## 核心摘录（面向 wiki 编译）

### 1. DEM 全局代价图与 A*

使用 SRTM 1 Arc-Second Global DEM（约 30 m 分辨率）建立纽芬兰圣约翰斯附近约 1 km 半径的高程区域。代价图组合坡度与高程，并对极端高程施加惩罚；A* 在代价图上生成全局路径。插值可让格网变细，但不会恢复源 DEM 未提供的细小沟坎和障碍。

- **对 wiki 的映射：** [A* 全局路径规划](../../wiki/methods/a-star.md)；[论文实体页](../../wiki/entities/paper-terrain-aware-ugv-dem-local-planning.md)

### 2. LiDAR 驱动的局部修正

局部模块对实时 LiDAR 点云做地面分割（Ground Plane Fitting 与 RANSAC）、坡度估计和可通行性判断，再执行障碍规避与自适应转向。单独的动态路径调整节点根据里程计计算横向偏差，以比例控制修正转向；障碍避让时暂时覆盖该修正。

- **对 wiki 的映射：** [DWA 局部路径规划](../../wiki/methods/dwa.md)；[地形适应](../../wiki/concepts/terrain-adaptation.md)；[论文实体页](../../wiki/entities/paper-terrain-aware-ugv-dem-local-planning.md)

### 3. 仿真结果与证据边界

在 Blender 生成的自定义地形和 Compusult Nanuk UGV 模型上，作者将提出方法与直线轨迹基线比较。挑战地形报告 95% 避障率、平均坡度 2.7°（基线分别为 0% 与 8°）；到达目标用时 77 s（基线被困），路径偏差 4.25 m（基线被困）。平缓地形上两者均为 43 s、0.7° 坡度、0 m 偏差。作者把 Nanuk 真机外场试验列为未来工作，因此当前结果是仿真证据。

- **对 wiki 的映射：** [论文实体页](../../wiki/entities/paper-terrain-aware-ugv-dem-local-planning.md)

## 项目页与开源核查（截至 2026-10-14）

- arXiv 提供论文、HTML、PDF 与 CC BY-SA 4.0 许可信息。
- 作者 [Devender Singh 个人学术主页](https://idevender.github.io/) 的新闻与论文列表链接 arXiv 和 PDF，未为本论文提供独立项目页或代码仓；作者页对另一篇有代码的论文则明确列出 Code 与 Project page。
- Memorial University 的 [Visual and Analytic Computing Lab publications](https://www.vaclab.ca/publications/) 列出该预印本，无代码入口。
- **代码状态：未发现公开实现。** arXiv TeX 源码不是 Gazebo 仿真工程；不能据此推断代码已开源。
- **真机状态：尚未验证。** 论文将 Nanuk UGV 外场试验列为 future work。

## 关联归档

- 作者主页：[Devender Singh](https://idevender.github.io/)
- 实验室发表列表：[Visual and Analytic Computing Lab Publications](https://www.vaclab.ca/publications/)
- Wiki：[Terrain-Aware UGV 论文实体](../../wiki/entities/paper-terrain-aware-ugv-dem-local-planning.md)

