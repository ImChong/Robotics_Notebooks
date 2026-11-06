---
type: concept
tags: [calibration, perception, computer-vision, manipulation, machine-vision]
status: complete
updated: 2026-11-06
related:
  - ./hand-eye-calibration.md
  - ./perception-coordinate-postprocessing.md
  - ../formalizations/3d-coordinate-transforms-vision-robotics.md
  - ../formalizations/homogeneous-coordinates-transform.md
  - ../methods/levenberg-marquardt.md
  - ../queries/robot-perception-stack-selection-loop.md
  - ../queries/contact-wrench-closed-loop.md
sources:
  - ../../sources/blogs/wechat_dafan_optics_three_calibrations_2026-11-04.md
  - ../../sources/blogs/wechat_xinjiqishijue_three_calibrations_repost_2026-11-06.md
  - ../../sources/papers/zhang_flexible_camera_calibration_2000.md
  - ../../sources/papers/tsai_lenz_hand_eye_calibration_1989.md
  - ../../sources/sites/opencv-calib3d-hand-eye.md
summary: "张正友（相机内参+畸变）→ 九点（2D 平面像素↔机器人坐标）→ 手眼（3D 六自由度 AX=XB）三种标定的对象、输入输出与选型递进关系。"
---

# 相机标定三层：张正友 / 九点 / 手眼

## 一句话定义

视觉引导机器人时常说的三种「标定」其实解决三个不同问题：**张正友标定**求相机内参与畸变（相机自己准不准），**九点标定**求 2D 平面上像素到机器人坐标的映射，**手眼标定**求相机与机器人之间的 3D 刚体位姿；它们是递进而非并列关系（文章观点，与库内手眼页的分层一致）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Intrinsics | Camera intrinsic parameters | 焦距、主点（及畸变系数） |
| Extrinsics | Camera extrinsic parameters | 相机相对某坐标系的位姿 |
| AX=XB | Hand-eye matrix equation | 手眼标定的经典方程形式 |
| PnP | Perspective-n-Point | 由已知几何板的图像点求板相对相机位姿 |
| FK | Forward Kinematics | 关节角 → 末端位姿 |

## 为什么重要

- 内参 / 畸变不校正，像素无法换算成物理尺寸，后面的测量与定位都有系统偏差（文章表述）。
- 选错标定层：只需平面抓取却做完整 3D 手眼是浪费；3D 抓取却只做九点则映射不成立（这一条是我对文章递进逻辑的推论）。
- 与 [手眼标定](./hand-eye-calibration.md) 页互补：该页深入 AX=XB 与 OpenCV API，本页给出三层的分工与选型。

## 核心原理

| | 张正友标定 | 九点标定 | 手眼标定 |
|---|---|---|---|
| 求什么 | 内参 + 径向 / 切向畸变 | 2D 平面变换矩阵（仿射 / 透视） | 相机↔机器人 6 自由度位姿 |
| 维度 | 相机自身（与机器人无关） | 平面（2D） | 3D |
| 数据 | 平面板多姿态图像 | 9 个机器人坐标已知的点 + 对应像素 | 多组（机器人位姿，板相对相机位姿） |
| 典型场景 | 尺寸检测、所有视觉应用前置 | 2D 分拣、上下料、贴装 | 3D 抓取、精密装配、多传感器融合 |

### 张正友标定

- 一手来源：Zhang 2000 TPAMI，平面图案**至少两个**朝向，运动无需已知，闭式解 + 最大似然非线性精修，建模径向畸变（[归档](../../sources/papers/zhang_flexible_camera_calibration_2000.md)）。
- 文章写 15~25 组图像：这是工程经验，不是论文下限（口径不同）。
- OpenCV 的 `calibrateCamera` 是常见实现（库内手眼归档提到「先内参再手眼」；此处未逐项核对其内部算法）。

### 九点标定

- 做法（文章）：在机器人工作平面取 9 个点，机器人坐标已知，拍得对应像素，解出平面变换矩阵（常称「手眼矩阵」）。
- 部署：眼在手（相机随臂）/ 眼在外（相机固定）两种，二者采集过程不同（见文章归档）。
- **推测（我的，未查一手来源）：** 平面仿射变换仅 6 个参数，3 个不共线点即可确定，9 点是冗余观测以最小二乘平均噪声；九点法无统一论文出处，属工程惯例。

### 手眼标定

- 模型：AX=XB，A 为机械臂末端运动，B 为相机观测的板运动，X 为待求手眼变换；解析解（Tsai–Lenz、Park–Martin）给初值，重投影误差优化精调（文章表述；Tsai–Lenz 见 [摘录](../../sources/papers/tsai_lenz_hand_eye_calibration_1989.md)，OpenCV 枚举见 [API 归档](../../sources/sites/opencv-calib3d-hand-eye.md)）。
- 须采集有足够旋转的多组位姿，避免共面 / 共轴退化。数量上文章写 15~25 组，库内手眼页写 ≥10–15 组，**均为经验值，彼此不一致**。

```mermaid
flowchart LR
  Z[张正友：内参 + 畸变] --> N[九点：2D 像素↔机器人平面]
  Z --> H[手眼：3D 相机↔机器人 AX=XB]
  N -.平面场景轻量化.-> H
```

## 选型（文章建议，作者表述）

- 只要内参 / 畸变：张正友。
- 2D 平面抓取：九点够用。
- 3D 抓取、精密装配：完整手眼标定。
- 文末的标定板产品推荐为厂商商业内容，本页不收。

## 工程要点（转载版补充，作者表述）

2026-11-06 新机器视觉转载了一篇与大凡光学原文基本相同的浓缩版（见 [转载归档](../../sources/blogs/wechat_xinjiqishijue_three_calibrations_repost_2026-11-06.md)），新增了下面几点：

- **精度链：** 张正友标定的精度主要取决于标定板几何精度与图像采集质量；九点标定的精度同时受标定板精度、相机标定精度、机械臂重复定位精度共同影响，任一环节短板都会拉低最终抓取精度。**一手出处未找到**，可理解为各环节误差叠加（我的推断，不作定量结论）。
- **布点与验证：** 九个点要均匀覆盖整个工作区域，避免只集中在画面中心；标定后用标准工件在多个位置做抓取验证。
- **复校：** 机械臂磨损、碰撞、温度变化会使手眼矩阵漂移，产线应定期复校；换设备或拆装碰撞后须重新标定。
- **流程：** 先相机标定，再九点标定，且标定板摆放与工作姿态与真实工况一致。

## 局限

- 来源是标定板厂商的科普文，**没有实验数据与论文清单**；「一表分清」对照表是图片，未取到。
- 精度数字（如标定板 ±1μm）是产品参数，未核实，不作为结论。

## 关联页面

- [手眼标定](./hand-eye-calibration.md)
- [感知后处理与坐标变换](./perception-coordinate-postprocessing.md)
- [三维坐标变换](../formalizations/3d-coordinate-transforms-vision-robotics.md)
- [齐次坐标与变换](../formalizations/homogeneous-coordinates-transform.md)
- [Levenberg–Marquardt](../methods/levenberg-marquardt.md)
- [Query：机器人视觉感知栈选型闭环](../queries/robot-perception-stack-selection-loop.md)
- [Query：接触力旋量闭环知识链](../queries/contact-wrench-closed-loop.md)

## 参考来源

- [大凡光学：三种标定原理（微信归档）](../../sources/blogs/wechat_dafan_optics_three_calibrations_2026-11-04.md)
- [新机器视觉转载版归档](../../sources/blogs/wechat_xinjiqishijue_three_calibrations_repost_2026-11-06.md)
- [Zhang 2000 摘录](../../sources/papers/zhang_flexible_camera_calibration_2000.md)
- [Tsai & Lenz 1989 摘录](../../sources/papers/tsai_lenz_hand_eye_calibration_1989.md)
- [OpenCV calib3d 手眼 API 归档](../../sources/sites/opencv-calib3d-hand-eye.md)
