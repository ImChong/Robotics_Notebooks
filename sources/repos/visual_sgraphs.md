# visual_sgraphs（vS-Graphs）

> 来源归档

- **标题：** visual_sgraphs / vS-Graphs
- **类型：** repo
- **链接：** <https://github.com/snt-arg/visual_sgraphs>
- **论文：** arXiv:2503.01783（RA-L 2026）
- **Stars：** ~50+（2026-07）
- **入库日期：** 2026-07-02
- **一句话说明：** 基于 **ORB-SLAM 3.0** 的 RGB-D VSLAM，在线构建可优化的分层 **3D 场景图**（墙/地面 → 房间/楼层），提升轨迹与建图精度。
- **沉淀到 wiki：** [paper-vs-graphs-visual-slam-scene-graph](../../wiki/entities/paper-vs-graphs-visual-slam-scene-graph.md)、[navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)

---

## 核心定位

- **基线：** ORB-SLAM 3.0 多线程 VSLAM（Tracking / Local Mapping / Loop Closure）
- **新增线程：** Building Component Recognition（全景分割 + 平面拟合）、Structural Element Recognition（房间/楼层推断）
- **输入：** RGB-D（当前版本）；可选 ArUco 语义增强
- **输出：** 位姿轨迹 + 几何地图 + 分层 3D 场景图
- **相关项目：** [LiDAR S-Graphs](https://github.com/snt-arg/lidar_situational_graphs)（同团队激光版）

---

## 对 wiki 的映射

- 论文实体：[paper-vs-graphs-visual-slam-scene-graph](../../wiki/entities/paper-vs-graphs-visual-slam-scene-graph.md)
- 基线对照：[orb-slam3](../../wiki/entities/orb-slam3.md)
- 导航栈总览：[navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)
