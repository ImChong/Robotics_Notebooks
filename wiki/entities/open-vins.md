---
type: entity
tags: [repo, vio, msckf, research, ros]
status: complete
updated: 2026-05-27
related:
  - ../entities/vins-fusion.md
  - ../comparisons/lidar-slam-lio-vio-selection.md
sources:
  - ../../sources/repos/open_vins.md
summary: "OpenVINS 是开源视觉-惯性导航研究平台：MSCKF 类滤波、可配置特征与 ROS 集成，便于算法对比实验。"
---

# OpenVINS

**OpenVINS** 面向 **VIO 研究** 的可扩展滤波框架，强调可复现与模块配置。

## 为什么重要

- **学术基准**：便于与优化式 VIO（如 VINS）对照。
- **RPNG 实验室**：持续维护的 ROS 集成与数据集工具。

## 核心结构/机制

| 特性 | 说明 |
|------|------|
| **MSCKF** | 多状态约束卡尔曼 |
| **Features** | 可换检测/跟踪 |
| **Calibration** | 在线/离线标定工具链 |

## 常见误区或局限

- **工程部署**：大规模产品化常选 VINS-Fusion 或商用 VIO；OpenVINS 偏研究。
- **极端运动**：需仔细调过程噪声与特征管理。

## 参考来源

- [sources/repos/open_vins.md](../../sources/repos/open_vins.md)
- [rpng/open_vins](https://github.com/rpng/open_vins)

## 关联页面

- [VINS-Fusion](../entities/vins-fusion.md)
- [LiDAR / LIO / VIO 选型](../comparisons/lidar-slam-lio-vio-selection.md)

## 推荐继续阅读

- https://docs.openvins.com/
