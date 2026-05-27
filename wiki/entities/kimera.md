---
type: entity
tags: [repo, semantic-slam, vio, mit, spark]
status: complete
updated: 2026-05-27
related:
  - ../entities/orb-slam3.md
  - ../comparisons/lidar-slam-lio-vio-selection.md
sources:
  - ../../sources/repos/kimera.md
summary: "Kimera 是 MIT SPARK 语义 SLAM 套件：Kimera-VIO、Kimera-RPGO、Kimera-Semantics 模块化组合。"
---

# Kimera

**Kimera** 将 **视觉-惯性里程计、鲁棒位姿图与语义网格** 组合为度量-语义地图。

## 为什么重要

- **语义 SLAM 代表**：同时输出几何与语义标签。
- **模块化**：各子仓可独立升级与替换。

## 核心结构/机制

| 子项目 | 说明 |
|--------|------|
| **Kimera-VIO** | 立体+IMU 前端 |
| **Kimera-RPGO** | 回环与鲁棒优化 |
| **Kimera-Semantics** | 3D 语义重建 |

## 常见误区或局限

- **复杂度高**：集成与调参成本高于单一 VIO 库。
- **硬件**：立体+IMU 同步要求高。

## 参考来源

- [sources/repos/kimera.md](../../sources/repos/kimera.md)
- [MIT-SPARK/Kimera](https://github.com/MIT-SPARK/Kimera)

## 关联页面

- [ORB-SLAM3](../entities/orb-slam3.md)
- [LiDAR / LIO / VIO 选型](../comparisons/lidar-slam-lio-vio-selection.md)

## 推荐继续阅读

- https://arxiv.org/abs/1910.02490
