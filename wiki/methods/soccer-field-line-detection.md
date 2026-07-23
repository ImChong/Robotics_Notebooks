---
type: method
tags: [computer-vision, soccer, perception, yolo, geometry, humanoid]
status: complete
updated: 2026-07-23
related:
  - ../methods/object-detection.md
  - ../concepts/soccer-field-simulation.md
  - ../methods/visual-line-matching-localization.md
  - ../entities/intel-realsense.md
  - ../entities/booster-robocup-demo.md
  - ../tasks/humanoid-soccer.md
  - ../entities/humanoid-system-curriculum.md
sources:
  - ../../sources/courses/shenlan_humanoid_system_theory_practice.md
summary: "球门与足球场线交点检测：用 YOLO 等检测球/门，并结合线分割或关键点回归得到场地线交点，为后续线匹配定位提供观测量。"
---

# 足球场线与球门检测

## 一句话定义

**球门与场地线交点检测**从机载图像中识别 **球、球门与场地线几何特征（含线–线交点）**，为 RoboCup 风格视觉定位提供测量——课程第 6.4 节与 YOLO11 实践。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| YOLO | You Only Look Once | 球/门实时检测常用族 |
| KPI | Keypoint / Intersection | 线交点或角点观测量 |
| Homography | Homography | 场地平面到图像的单应 |
| NMS | Non-Maximum Suppression | 检测框去重 |
| ROI | Region of Interest | 场地区域裁剪 |

## 为什么重要

- 仅有球检测不够：自定位依赖 **场线结构**；交点是稀疏、可匹配的强特征。
- 衔接到 [线匹配定位](./visual-line-matching-localization.md) 与 [EKF 融合](./visual-line-ekf-fusion.md)。
- 数据采集与评测依赖 [足球场仿真环境](../concepts/soccer-field-simulation.md)。

## 主要技术路线

| 路线 | 输出 | 备注 |
|------|------|------|
| YOLO 检测球/门 | 框 + 类 | 课程 YOLO11 实践主线 |
| 线分割 / Hough | 线段参数 | 经典几何管线 |
| 关键点 / 交点回归 | L/T/X KPI | 直接服务线匹配 |
| 检测 + 拓扑后处理 | 过滤后的场地图元 | 见 [感知后处理](../concepts/perception-coordinate-postprocessing.md) |

## 核心原理

常见两段式：

1. **实例检测**：YOLO 检出 ball / goal / 可选 robot。
2. **线几何**：线分割、Hough、或直接关键点头回归得到 L/T/X 型交点。
3. **后处理**：按场地拓扑过滤不可能组合（见 [感知后处理](../concepts/perception-coordinate-postprocessing.md)）。

## 工程实践

- 课程实践标明 **YOLO11**：在仿真场采图 → 标注 → 训练 → 真机/仿真推理。
- 指标：球 mAP、交点像素误差、端到端定位误差。
- 对照：[Booster RoboCup demo](../entities/booster-robocup-demo.md) 中的场线感知叙述。

## 局限与风险

- 反光地板、曝光会让白线断裂；需形态学或时序平滑。
- 交点类别不平衡，稀有角点易漏检。

## 关联页面

- [目标检测](./object-detection.md)
- [足球场仿真](../concepts/soccer-field-simulation.md)
- [人形系统课程策展](../entities/humanoid-system-curriculum.md)

## 参考来源

- [深蓝学院人形系统课程大纲](../../sources/courses/shenlan_humanoid_system_theory_practice.md)

## 推荐继续阅读

- Ultralytics YOLO11 文档（训练与导出）
