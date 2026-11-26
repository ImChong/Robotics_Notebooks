---
type: method
tags: [vision, robotics, perception]
status: complete
updated: 2026-11-26
summary: "把视觉观测转换为机器人可使用的对象、空间或场景信息。"
sources:
  - ../../sources/sites/methods-primary-robot-visual-perception-2026-11-26.md
original:
  - ../../sources/sites/methods-primary-robot-visual-perception-2026-11-26.md
related:
  - ../concepts/perception-coordinate-postprocessing.md
  - visual-representation-learning.md
  - ../queries/robot-perception-stack-selection-loop.md
---

# Robot Visual Perception · 机器人视觉感知流程

## 一句话定义

把视觉观测转换为机器人可使用的对象、空间或场景信息。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| RGB-D | Red Green Blue and Depth | 给像素颜色配套深度观测 |
| TF | Transform | 将光学坐标下的估计变换到任务坐标 |
| ROI | Region of Interest | 图像裁剪区域，影响像素坐标和投影参数 |

## 为什么重要

检测框、分割掩码或置信分数本身不能直接成为机器人目标。闭环需要把感知结果变成带参考坐标系、采样时间和有效性信息的几何量。

## 主要技术路线（核心机制）

1. 首先将图像与匹配的 `CameraInfo` 对齐。区分原始图像内参 K 和畸变 D、立体校正 R 与校正图像的投影矩阵 P；分辨率、裁切和校正模式变化后，不能沿用旧像素标定。
2. RGB-D 案例：把检测或分割结果投回校正像素，剔除无效深度，按深度语义恢复光学坐标点。像素投影本身只给视线；无深度、无其他几何约束时不能恢复度量位置。用图像采样时刻的变换把结果送入机器人基座或任务坐标系。

### 流程总览

```mermaid
flowchart LR
  img["图像与 CameraInfo 对齐"]
  infer["检测或分割"]
  pix["校正像素与有效深度"]
  pose["光学坐标的点或位姿"]
  tf["时序变换到任务坐标"]
  validate["检查尺度新鲜度和不确定性"]
  img --> infer --> pix --> pose --> tf --> validate
```

## 工程实践

- 先用已知尺寸、已知相机位姿的板件校验像素重投影与机器人坐标；RGB 与深度不同光心时确认已完成配准。
- 发布位置时附带坐标系、时间戳与有效标记。回放长尾光照、反光、遮挡与运动模糊；分别度量检测错误、空间误差和采集到控制的端到端延迟。

## 局限与风险

单帧高置信检测不构成距离、法向或姿态可靠性。对运动物体，使用当前 TF 配准旧图像会引入随速度增加的误差。

## 关联页面

- [感知坐标后处理](../concepts/perception-coordinate-postprocessing.md)
- [视觉表征学习](visual-representation-learning.md)
- [机器人感知技术栈选型](../queries/robot-perception-stack-selection-loop.md)

## 参考来源

- [方法一手资料与核对范围](../../sources/sites/methods-primary-robot-visual-perception-2026-11-26.md)

## 推荐继续阅读

- [ROS 2 sensor_msgs CameraInfo](https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/CameraInfo.msg)
- [ROS image_geometry PinholeCameraModel](https://github.com/ros-perception/vision_opencv/blob/rolling/image_geometry/src/pinhole_camera_model.cpp)
