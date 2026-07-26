# GO2 三维语义建图与 SAM 2D→3D：维护者答疑整理

- **类型**：`personal`（社区答疑 / 知识库对照整理，非正式出版物）
- **日期**：2026-07-26
- **触发场景**：Lumina 具身智能社区话题——「有没有 GO2 技术资料库」「对 GO2 的 3D 语义建图感兴趣」「点云建图在狗子移动时效果不理想；点云建图 + SAM 如何从 2D 转为 3D 自适应识别（提到 CMU 相关工作）」
- **用途**：为独立 Query 产物 [GO2 三维语义建图与 SAM 流水线](../../wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md) 提供可追溯编译来源；正文以 wiki query 页为准。

## 库内现状（对照 Robotics_Notebooks）

- **几何建图基础已较完整**：[`point_lio_unilidar`](../repos/point_lio_unilidar.md)、[FAST-LIO](../repos/fast_lio.md)、[LIO-SAM](../repos/lio_sam.md)。
- **Real-Time Polygonal Semantic Mapping** 仍是 Paper Notebooks「待深读」索引占位（[`humanoid_pnb_...`](../papers/humanoid_pnb_real-time-polygonal-semantic-mapping-for-humanoi.md)），**尚未**形成完整的「GO2 + 相机 + SAM + 3D 语义融合」方案页——本答疑补的是选型与流水线洞见，不是替代该论文深读。

## 核心结论（可操作）

1. **先几何、后语义**：运动重影/墙面变厚/地图撕裂，优先查 LiDAR–IMU 时间同步、逐点时间戳、去畸变、外参、IMU 初始化、振动、回环、动态物体；不要先叠 SAM。
2. **SAM 不做 2D→3D**：SAM/SAM2 产出二维 mask；类别来自检测器/VLM；三维来自投影 + 外参 + 位姿 + 跨帧融合。
3. **CMU 两条线勿混**：[`autonomy_stack_go2`](../repos/autonomy_stack_go2.md) = 几何自主导航；[MSCV Semantic 3D Mapping](../sites/cmu-mscv-semantic-3d-mapping.md) = DETR→SAM→2D 标签投影到 3D。
4. **推荐栈**：Point-LIO 高频几何 + 关键帧语义；后续可看 DualMap / OVO / OV-SAM3D / FindAnything。

## 推荐一手项目（开源状态摘要）

| 项目 | 角色 | 开放程度（入库日核查） |
|------|------|------------------------|
| [point_lio_unilidar](../repos/point_lio_unilidar.md) | GO2 L1/L2 几何建图基线 | **已开源** |
| [autonomy_stack_go2](../repos/autonomy_stack_go2.md) | GO2 全栈几何自主导航 | **已开源** |
| [CMU MSCV Semantic 3D Mapping](../sites/cmu-mscv-semantic-3d-mapping.md) | DETR+SAM→3D 伪标注流水线说明 | **项目页实体** [`wiki/entities/cmu-mscv-semantic-3d-mapping.md`](../../wiki/entities/cmu-mscv-semantic-3d-mapping.md)；独立仓待补 |
| [DualMap](../repos/dualmap.md) | 在线开放词汇语义地图 + ROS | **已开源** |
| [OVO](../repos/ovo-semantic-mapping.md) | 在线开放词汇语义映射 + SAM2 | **已开源** |
| [OV-SAM3D](../repos/ov-sam3d.md) | 离线多视角 SAM→3D 实例 | **已开源** |
| [FindAnything](../sites/findanything.md) | 对象级体素子地图；机载演示 | **项目页实体** [`wiki/entities/findanything.md`](../../wiki/entities/findanything.md)；宣称并入 OKVIS2-X |

## 对 wiki 的映射

| 要点 | 目标页 |
|------|--------|
| 几何故障树 + SAM 投影数学 + 落地顺序 | `wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md` |
| GO2 全栈几何导航开源仓 | `wiki/entities/autonomy-stack-go2.md` |
| 在线开放词汇语义（ROS） | `wiki/entities/dualmap.md` |
| 在线 RGB-D 开放词汇语义 | `wiki/entities/ovo-semantic-mapping.md` |
| 离线多视角 SAM→3D | `wiki/entities/ov-sam3d.md` |
| FindAnything 项目页占位（仓待补） | `wiki/entities/findanything.md` |
| CMU MSCV DETR+SAM 项目页占位 | `wiki/entities/cmu-mscv-semantic-3d-mapping.md` |
| GO2 L1 Point-LIO 工程注意 | `wiki/entities/point-lio-unilidar.md` |
| 导航/SLAM 栈中 GO2 入口 | `wiki/overview/navigation-slam-autonomy-stack.md` |
| 多边形语义建图论文占位勿等同本方案 | `wiki/entities/paper-notebook-real-time-polygonal-semantic-mapping-for-humanoi.md` |
