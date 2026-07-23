---
type: concept
tags: [perception, coordinate-transform, calibration, robotics, soccer, tf]
status: complete
updated: 2026-07-23
related:
  - ../formalizations/3d-coordinate-transforms-vision-robotics.md
  - ../formalizations/homogeneous-coordinates-transform.md
  - ../methods/soccer-field-line-detection.md
  - ../methods/visual-line-matching-localization.md
  - ../methods/visual-line-ekf-fusion.md
  - ../entities/intel-realsense.md
  - ../entities/humanoid-system-curriculum.md
sources:
  - ../../sources/courses/shenlan_humanoid_system_theory_practice.md
summary: "感知后处理与坐标变换：检测框/线特征经相机模型与 TF 链进入 base/场地系，并做置信度与拓扑校验；课程 7.1，足球视觉闭环胶水层。"
---

# 感知后处理与坐标变换

## 一句话定义

**感知后处理与坐标变换**把检测器输出的 **像素量**（框、线、交点）经相机内参、深度/平面假设与外参链，变成 **机器人 base 或场地世界系下的几何量**，并做过滤与拓扑校验——课程第 7.1 节。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TF | Transform Frame | ROS 坐标树 |
| \(K\) | Intrinsic matrix | 像素 ↔ 相机射线 |
| Extrinsics | Camera Extrinsics | 相机到头/base 的 \(T\) |
| \(T^a_b\) | Transform | \(a\) 系下看 \(b\) 的位姿（约定需统一） |
| NMS | Non-Maximum Suppression | 检测框去重 |
| ROI | Region of Interest | 感兴趣图像区域 |

## 为什么重要

- **检测对、坐标错 = 整条定位链错**：线匹配与 [EKF](../methods/visual-line-ekf-fusion.md) 吃的是场地系几何，不是像素。
- **可复用胶水层**：同一后处理可服务寻球、对准球门、报告「球在左前方 2 m」。
- 课程把本节放在检测与线匹配之间，强制建立 **像素 → 米制** 的工程习惯。

## 核心原理

### 典型变换链

\[
\text{pixel}\xrightarrow{K^{-1}}\text{ray}
\xrightarrow{\text{depth / ground plane}}\text{camera}
\xrightarrow{T_{\text{cam}}^{\text{base}}}\text{base}
\xrightarrow{T_{\text{base}}^{\text{field}}}\text{field}
\]

| 步骤 | 关键风险 |
|------|----------|
| 反投影 | 内参过期、畸变未去 |
| 深度/平面求交 | 深度空洞；俯仰大时平面假设坏 |
| 外参 | 头关节未进 TF，相机随动未补偿 |
| 场地系 | 尚未全局定位时只能出 base 系相对量 |

齐次变换细节见 [三维坐标变换](../formalizations/3d-coordinate-transforms-vision-robotics.md)、[齐次坐标](../formalizations/homogeneous-coordinates-transform.md)。

```mermaid
flowchart LR
  DET["YOLO / 线检测"] --> FILT["置信度 / NMS"]
  FILT --> CAM["像素 → 相机系"]
  CAM --> BASE["相机 → base"]
  BASE --> FIELD["base → field\n若已定位"]
  FIELD --> MATCH["线匹配 / 决策"]
```

### 后处理规则（足球语境）

1. **置信度门控**：低分球/门直接丢弃，宁缺毋滥。
2. **几何可行**：球门宽度、线夹角超出场地模型 → 拒识。
3. **时间对齐**：图像戳与关节/IMU 戳对齐，避免摆头残影。
4. **输出结构化**：统一消息（类、场地点、协方差提示、时间戳）。

## 工程实践

### 标定清单（[RealSense](../entities/intel-realsense.md) + G1）

| 项 | 做法 |
|----|------|
| 内参 | SDK 厂参或重新标定板 |
| 头–相机外参 | 标定板或 CAD + 实测；写入 URDF/TF |
| 场地平面 | 标定俯仰；或用深度拟合地面 |
| 验证 | 已知距离地标，误差应在厘米～分米级（教学） |

### 调试技巧

- RViz 同时显示检测投影回图像与场地系 marker，肉眼查「左右颠倒」。
- 日志记录原始检测 + 变换后点，离线回放比盯真机快。
- 无深度时的平面求交：限制最大有效距离，避免天上噪点落到百米外。

### 与后续模块契约

| 下游 | 需要的坐标 |
|------|------------|
| [线匹配](../methods/visual-line-matching-localization.md) | 相机或 base 系线/交点 |
| [EKF 融合](../methods/visual-line-ekf-fusion.md) | 场地系观测或可投影残差 |
| 踢球决策 | base 或场地系球位置 |

## 局限与风险

- **平面假设**在机器人大幅俯仰/球不在地面时失效。
- **深度在强光/黑球**上不可靠 → 允许 RGB-only 回退并增大观测噪声。
- **误区**：在检测节点里写死魔法数外参，不进 TF——后续换相机必炸。

## 关联页面

- [场地线检测](../methods/soccer-field-line-detection.md)
- [线匹配视觉定位](../methods/visual-line-matching-localization.md)
- [线特征 EKF 融合](../methods/visual-line-ekf-fusion.md)
- [Intel RealSense](../entities/intel-realsense.md)
- [人形系统课程策展](../entities/humanoid-system-curriculum.md)

## 参考来源

- [深蓝学院人形系统课程大纲](../../sources/courses/shenlan_humanoid_system_theory_practice.md)

## 推荐继续阅读

- [三维坐标变换（视觉与机器人）](../formalizations/3d-coordinate-transforms-vision-robotics.md)
- ROS 2 `tf2` 教程（静态/动态广播与监听）
