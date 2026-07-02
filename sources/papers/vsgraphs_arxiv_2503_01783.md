# vS-Graphs：紧耦合视觉 SLAM 与可优化 3D 场景图（arXiv:2503.01783）

> 论文来源归档（ingest）

- **标题：** vS-Graphs: Tightly Coupling Visual SLAM and 3D Scene Graphs Exploiting Hierarchical Scene Understanding
- **类型：** paper / vslam / rgb-d / scene-graph / semantic-mapping / indoor-navigation
- **arXiv：** <https://arxiv.org/abs/2503.01783> · PDF：<https://arxiv.org/pdf/2503.01783>
- **DOI：** <https://doi.org/10.48550/arXiv.2503.01783>
- **期刊：** IEEE Robotics and Automation Letters (RA-L) 2026（仓库标注）
- **作者：** Ali Tourani¹, Saad Ejaz¹, Hriday Bavle¹, Miguel Fernandez-Cortizas¹, David Morilla-Cabello², Jose Luis Sanchez-Lopez¹, Holger Voos¹
- **作者单位：** ¹卢森堡大学 SnT 自动化与机器人研究组 · ²萨拉戈萨大学 I3A
- **项目页：** <https://snt-arg.github.io/vsgraphs-results/>
- **代码：** <https://github.com/snt-arg/visual_sgraphs>
- **入库日期：** 2026-07-02
- **一句话说明：** 在 **ORB-SLAM 3.0** 基线上新增 **建筑构件识别**（墙/地面）与 **结构元素推断**（房间/楼层）线程，将环境驱动语义实体纳入 **可联合优化的分层 3D 场景图**，在标准 RGB-D 基准与自采 **AutoSense** 数据上相对 SOTA VSLAM 平均 **ATE 降低 15.22%**，纯视觉场景理解精度接近 **LiDAR S-Graphs**。

## 核心摘录（面向 wiki 编译）

### 1) 紧耦合 VSLAM + 可优化 3D 场景图

- **要点：** 受 **LiDAR S-Graphs** 启发，在 **ORB-SLAM 3.0** 多线程架构上扩展 **Building Component Recognition** 与 **Structural Element Recognition** 线程；Atlas 除几何地图外存储 **墙/地面** 与 **房间/楼层** 实体，在 **Local/ Global Bundle Adjustment** 中与 KeyFrame 位姿、地图点 **联合优化**；相对 Hydra、HOV-SG 等 **离线场景图构建器** 与 **标记依赖** 方案，实现 **实时、无 fiducial** 的在线分层场景理解。
- **对 wiki 的映射：** [`wiki/entities/paper-vs-graphs-visual-slam-scene-graph.md`](../../wiki/entities/paper-vs-graphs-visual-slam-scene-graph.md)

### 2) 建筑构件：全景分割 + RANSAC 平面拟合

- **要点：** 当前版本以 **RGB-D** 为输入；KeyFrame 点云经 **Panoptic-FCN (pFCN)** 或 **YOSO** 全景分割筛出墙/地面类，再经下采样、距离滤波与 **RANSAC** 平面拟合，并用垂直/水平几何约束验证；构件在 Atlas 中关联合并，支撑后续结构推断与回环时冗余实体融合。
- **对 wiki 的映射：** 同上实体页；[`wiki/entities/orb-slam3.md`](../../wiki/entities/orb-slam3.md)（基线对照）

### 3) 结构元素：房间与楼层的几何约束优化

- **要点：** 每 2 s 从已定位墙/地面推断 **房间**（至少两面墙围合自由空间簇，支持非曼哈顿布局的平行/垂直墙代价）与 **楼层**（房间集合的共面参考）；房间级代价含墙平行/垂直与质心一致性项，纳入 BA 提升轨迹与地图一致性；可选 **ArUco** 标记为结构元素提供语义标签（非定位必需）。
- **对 wiki 的映射：** 同上实体页

### 4) 轨迹、建图与场景理解评测

- **要点：** 在 **ICL / OpenLORIS / ScanNet / TUM-RGBD / AutoSense** 上对比 ORB-SLAM3、ElasticFusion、BAD SLAM；全数据集 vS-Graphs（YOSO + 结构元素）相对基线 **ATE 平均改善 15.22%**（BC+SE 配置 13.82%）；AutoSense 上 **建图 RMSE 中位数更低** 且点云约少 **10.15%**；多房间序列上墙/房间检测 **精度/召回** 与 **LiDAR S-Graphs** 相当；平均 **22±3 FPS**（基线 29±3 FPS）。
- **对 wiki 的映射：** 同上实体页；[`wiki/overview/navigation-slam-autonomy-stack.md`](../../wiki/overview/navigation-slam-autonomy-stack.md)

### 5) 与相关路线的定位差异

- **要点：** 相对 **语义特征过滤类 VSLAM**（SaD/OVD/YDD-SLAM 等）推进到 **环境布局级实体**；相对 **Hydra / HOV-SG** 强调 **在线 SLAM 内嵌** 而非离线全图；相对 **标记驱动布局 SLAM**（SemUco 等）无需预置 marker；相对 **LiDAR S-Graphs** 用 **RGB-D 视觉验证** 达到相近结构检测，但曲面/低纹理场景仍受限。
- **对 wiki 的映射：** 同上实体页；[`wiki/comparisons/lidar-slam-lio-vio-selection.md`](../../wiki/comparisons/lidar-slam-lio-vio-selection.md)

## 对 wiki 的映射（仓库层）

- 代码归档：[sources/repos/visual_sgraphs.md](../repos/visual_sgraphs.md)

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
- [x] 实体页与导航/ORB-SLAM3 交叉引用
- [x] 代码仓库归档
