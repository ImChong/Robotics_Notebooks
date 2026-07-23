---
type: concept
tags: [navigation, mapping, lidar, costmap, filtering, mobile-robot]
status: complete
updated: 2026-07-23
related:
  - ../methods/a-star.md
  - ../methods/dwa.md
  - ../entities/navigation2.md
  - ../methods/lidar-odometry-fusion.md
  - ../entities/humanoid-system-curriculum.md
sources:
  - ../../sources/courses/shenlan_humanoid_system_theory_practice.md
  - ../../sources/repos/navigation2.md
summary: "动态障碍物剔除：从激光/点云中区分静态度与行人/运动物体，生成干净的二维占据栅格导航地图，避免全局规划被瞬态障碍污染。"
---

# 动态障碍物滤波（导航地图制作）

## 一句话定义

**动态障碍物滤波**在建图或代价地图流水线中识别并剔除 **非静态占用**，使二维导航地图主要反映墙体/家具等持久结构——对应课程第 4.1 节「动态障碍物剔除与二维导航地图制作」。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Occupancy Grid | Occupancy Grid Map | 二维占据概率栅格 |
| Costmap | Cost Map | Nav2 膨胀后的规划代价层 |
| SLAM | Simultaneous Localization and Mapping | 建图过程需避免动态污染 |
| Raycast | Ray Casting | 光束模型更新自由/占用 |
| ROI | Region of Interest | 滤波感兴趣空间范围 |

## 为什么重要

- 人不停走动时若直接 SLAM，地图会出现「幽灵墙」；A\* 全局路径会被瞬态障碍永久挡死。
- 局部层可用膨胀/voxel 清扫处理动态体，但 **静态层地图质量** 仍决定长距离规划可用性。

## 核心原理

常见手段（可组合）：

1. **时序一致性**：多帧射线投票，短暂占用不固化为地图。
2. **高度/聚类分割**：行人点云簇与地面、墙面分离后不写入静态层。
3. **外参运动检测**：与里程计不一致的扫描段标记为动态。
4. **Nav2 分层 costmap**：static / obstacle / inflation 分离，动态只进 obstacle 层。

## 工程实践

- 建图阶段尽量 **少人空场**；无法避免时开动态滤波或事后地图编辑。
- 与 [A\*](../methods/a-star.md) / [DWA](../methods/dwa.md) 分工：本概念保证 **输入地图**；局部避障负责 **运行时动态**。

## 局限与风险

- 过激进滤波会抹掉应保留的可移动家具；过保守则地图脏。
- 人形摇晃导致的扫描拖影可能被误判为动态。

## 关联页面

- [Navigation2](../entities/navigation2.md)
- [人形系统课程策展](../entities/humanoid-system-curriculum.md)

## 参考来源

- [深蓝学院人形系统课程大纲](../../sources/courses/shenlan_humanoid_system_theory_practice.md)
- [Navigation2 归档](../../sources/repos/navigation2.md)

## 推荐继续阅读

- Nav2 Costmap 2D 官方文档（分层与插件）
