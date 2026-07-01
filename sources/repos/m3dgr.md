# M3DGR

> 来源归档

- **标题：** M3DGR
- **类型：** repo / dataset
- **链接：** https://github.com/sjtuyinjie/M3DGR
- **Stars：** ~413（2026-07）
- **会议：** IROS 2025
- **入库日期：** 2026-07-01
- **一句话说明：** 地面机器人 **多传感器、多场景、大规模基线** SLAM 数据集；Ultra-Fusion 论文扩展退化轨迹并在此对 60+ 系统做鲁棒性评测。
- **沉淀到 wiki：** [paper-ultra-fusion-multi-sensor-slam](../../wiki/entities/paper-ultra-fusion-multi-sensor-slam.md)、[lidar-slam-lio-vio-selection](../../wiki/comparisons/lidar-slam-lio-vio-selection.md)

---

## 核心定位

**M3DGR**（*A Multi-sensor, Multi-scenario and Massive-baseline SLAM Dataset for Ground Robots*，IROS 2025）为 **地面移动机器人** 提供多模态同步数据与 **大规模 SLAM 基线对照**，覆盖 **轮速、视觉、LiDAR、IMU、GNSS** 等组合及 **夜间弱光、长廊 LiDAR 退化、长时程、GNSS 拒止** 等挑战性场景。

Ultra-Fusion 论文在 M3DGR 上按 **WIO / VWIO / LWIO / LVWIO** 分组报告 ATE，并扩展仿真退化轨迹用于系统性鲁棒性评测。

- **关联论文 / 方法：** [Ultra-Fusion](../papers/ultra_fusion_arxiv_2606_21223.md)
- **代码（方法）：** [Ultra-Fusion](ultra_fusion.md)

---

## 对 wiki 的映射

- 实体页：[paper-ultra-fusion-multi-sensor-slam](../../wiki/entities/paper-ultra-fusion-multi-sensor-slam.md)
- 选型对比：[lidar-slam-lio-vio-selection](../../wiki/comparisons/lidar-slam-lio-vio-selection.md)
- 导航栈总览：[navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)
