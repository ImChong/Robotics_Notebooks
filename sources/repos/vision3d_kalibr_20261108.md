# Kalibr — 一手资料归档

> 来源层（官方代码/仓库）；视觉与三维重建一手资料专题，2026-11-08。

- **标题：** Kalibr
- **类型：** repo
- **链接：** https://github.com/ethz-asl/kalibr
- **入库日期：** 2026-11-08
- **一句话说明：** 相机和 IMU 的内参、外参与时间标定。
- **核对范围：** 已通读官方 README（能力、安装/支持范围、引用与限制），官方 Camera-IMU calibration 和 multiple-camera-calibration Wiki 全篇，以及 kalibr_calibrate_cameras、kalibr_calibrate_imu_camera CLI 入口、CameraCalibrator.py、IccCalibrator.py 的主要初始化与求解路径；主线源码快照 1f60227，未运行实测。
- **通读：** 已完成

## 直接一手来源（同一项目不同入口）

- [Kalibr](https://github.com/ethz-asl/kalibr) **（本归档主类型）**
- [Kalibr Camera-IMU calibration](https://github.com/ethz-asl/kalibr/wiki/Camera-IMU-calibration)

## 深读摘录（官方 README / CLI）

- 官方支持多相机投影+畸变、Camera–IMU 时空标定、IMU–IMU 标定、rolling-shutter 标定；此归档先聚焦多相机和 Camera–IMU。
- 主入口 `aslam_offline_calibration/kalibr/python/kalibr_calibrate_cameras` 使用 `kalibr_common.BagImageDatasetReader`、`extractCornersFromDataset`、`CameraGeometry.initGeometryFromObservations`、`MulticamCalibrationGraph` 完成观测建图与优化。
- `kalibr_calibrate_imu_camera` 读取相机链/IMU/target 配置后调用 `IccCalibrator.buildProblem` 构建 6 阶姿态样条与误差项，再由 `IccCalibrator.optimize` 使用 Optimizer2/LM 优化。
- 原文没有承诺支持任意相机与 IMU 的即插即用精度，必须逐项核对时间戳、靶板、观测连通性和 IMU 模型。

## 方法与边界

相机和 IMU 的内参、外参与时间标定。已核对官方 README、两份完整使用指南和可运行命令入口；尚未对真实数据进行运行验证，不代填重复投影误差或标定精度。

## 对 wiki 的映射与待核验事项

- 来源汇总：[视觉三维专题 90 条来源索引](../sites/vision3d_research_primary_sources_2026_11_08.md)。
- 在升格独立 wiki 实体页前，先按论文 ID / 项目页 / 源码查重，避免与既有节点及不同项目同仓实现冲突。
- [x] 完整阅读官方 README + 两份官方使用指南；该工具节点非单篇论文深读。
- [x] 核对两个 CLI 与对应 CameraCalibrator/IccCalibrator 主要代码路径；源码时序图位于 wiki。
- [x] 已升格 [Kalibr wiki](../../wiki/entities/kalibr.md)，收录两类 Mermaid 图、局限和实操；健康度待 CI 验证。
