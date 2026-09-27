# OpenCV calib3d — 手眼标定 API（4.x）

- **标题：** Camera Calibration and 3D Reconstruction — Hand-Eye Calibration
- **类型：** site / API reference
- **URL：** <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>
- **入库日期：** 2026-09-27
- **一句话说明：** OpenCV `calib3d` 模块提供 **手眼** 与 **机器人-世界-手眼** 标定函数，多种经典闭式/优化算法（含 Tsai–Lenz）。

## 核心 API（面向 wiki 编译）

### `cv::calibrateHandEye`

- **用途：** 已知多组 **机器人末端相对基座** 位姿 \((R_{g2b}, t_{g2b})\) 与 **标定板相对相机** 位姿 \((R_{t2c}, t_{t2c})\)，求解 **相机相对夹爪（或基座）** 的固定外参 \(R_{c2g}, t_{c2g}\)（Eye-in-hand / Eye-to-hand 由输入序列约定决定）。
- **方法枚举 `HandEyeCalibrationMethod`（常见）：** `CALIB_HAND_EYE_TSAI`、`CALIB_HAND_EYE_PARK`、`CALIB_HAND_EYE_HORAUD`、`CALIB_HAND_EYE_ANDREFF`、`CALIB_HAND_EYE_DANIILIDIS`。
- **典型输入：** `R_gripper2base`、`t_gripper2base`、`R_target2cam`、`t_target2cam`（向量，每组姿态一对）。
- **典型输出：** `R_cam2gripper`、`t_cam2gripper`（或等价 eye-to-hand 解释，取决于采集协议）。

### `cv::calibrateRobotWorldHandEye`

- **用途：** **机器人-世界-手眼** 联合标定：标定板固定在 **机器人基座/世界** 上（非手持板），同时估计 **相机–末端** 与 **基座–世界/板** 关系。
- **适用：** 固定工位、板贴桌面/机架，腕部相机观测板时末端位姿变化仍受 FK 约束。

## 工程读点

- 先 **内参**（`calibrateCamera` / 出厂参数）再手眼；姿态序列需 **足够旋转激励**，避免共面退化。
- OpenCV 的 `CALIB_HAND_EYE_TSAI` 对应经典 Tsai–Lenz 闭式解；精度不足时可换方法或用非线性 refinement。
- Python 绑定：`cv2.calibrateHandEye`、`cv2.calibrateRobotWorldHandEye`（参数与 C++ 一致）。

## 对 wiki 的映射

- [手眼标定（概念页）](../../wiki/concepts/hand-eye-calibration.md)
- [Tsai–Lenz 1989 论文摘录](../papers/tsai_lenz_hand_eye_calibration_1989.md)
