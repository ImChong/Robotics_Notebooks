---
type: entity
tags: [repo, tsdf, pose-graph, ethz, mapping]
status: complete
updated: 2026-05-27
related:
  - ../comparisons/lidar-slam-lio-vio-selection.md
  - ../overview/navigation-slam-autonomy-stack.md
sources:
  - ../../sources/repos/voxgraph.md
summary: "Voxgraph 基于 Voxblox TSDF 子图做位姿图优化，支持多会话建图与度量地图对齐。"
---

# Voxgraph

**Voxgraph** 在 **TSDF 子图** 上进行 **位姿图优化**，适合多会话对齐与稠密地图融合。

## 为什么重要

- **ETHZ ASL 路线**：与 Voxblox 稠密建图生态一致。
- **多会话**：适合重复巡检场景地图合并。

## 核心结构/机制

| 概念 | 说明 |
|------|------|
| **Submap** | TSDF 局部块 |
| **PGO** | 子图间位姿约束 |
| **Voxblox** | 底层体积建图 |

## 常见误区或局限

- **生态规模**：社区小于 FAST-LIO/LIO-SAM。
- **算力**：TSDF 融合消耗 GPU/CPU 资源。

## 参考来源

- [sources/repos/voxgraph.md](../../sources/repos/voxgraph.md)
- [ethz-asl/voxgraph](https://github.com/ethz-asl/voxgraph)

## 关联页面

- [LiDAR / LIO / VIO 选型](../comparisons/lidar-slam-lio-vio-selection.md)
- [导航·SLAM·自动驾驶栈总览](../overview/navigation-slam-autonomy-stack.md)

## 推荐继续阅读

- https://arxiv.org/abs/1910.00229
