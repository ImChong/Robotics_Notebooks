# Kalibr — 一手资料归档

> 来源层（作者项目/数据集/官方技术文档）；视觉与三维重建一手资料专题，2026-11-08。

- **标题：** Kalibr
- **类型：** site
- **链接：** https://github.com/ethz-asl/kalibr/wiki/Camera-IMU-calibration
- **入库日期：** 2026-11-08
- **一句话说明：** 相机和 IMU 的内参、外参与时间标定。
- **核对范围：** 已完整阅读官方 Camera-IMU-calibration Wiki（采集、IMU 噪声、时间戳、命令、输出、限制）与 multiple-camera-calibration Wiki（相机连通、观测筛选、命令、YAML/报告输出），对照官方 README 版本（截至 2026-11-08）。
- **通读：** 已完成

## 直接一手来源（同一项目不同入口）

- [Kalibr](https://github.com/ethz-asl/kalibr)
- [Kalibr Camera-IMU calibration](https://github.com/ethz-asl/kalibr/wiki/Camera-IMU-calibration) **（本归档主类型）**

## 官方案例与约束核对

- **多相机：** 相机固定、标定板移动，约 4 Hz 图像率用于减少冗余；相邻相机视野需有目标观测关联；由 `kalibr_calibrate_cameras` 生成 `camchain.yaml`。
- **Camera–IMU：** 标定板固定、相机 IMU rig 移动激励各惯性轴；文档示例 20 Hz 相机、200 Hz IMU；`kalibr_calibrate_imu_camera` 默认估计时间偏移。
- **诊断：** 输出 YAML/TXT/PDF；应检查 timestamp 的间歇批量上传、IMU noise density/bias random walk，以及 Aprilgrid/棋盘格目标对称性的影响。
- 原文链接：[Camera–IMU 官方教程](https://github.com/ethz-asl/kalibr/wiki/Camera-IMU-calibration)、[多相机教程](https://github.com/ethz-asl/kalibr/wiki/multiple-camera-calibration)。

## 方法与边界

相机和 IMU 的内参、外参与时间标定。官方使用指南已完整核对，提供采集模式、CLI、YAML/PDF 报告与时间戳故障边界；未做真实标定实验。

## 对 wiki 的映射与待核验事项

- 来源汇总：[视觉三维专题 90 条来源索引](../sites/vision3d_research_primary_sources_2026_11_08.md)。
- 在升格独立 wiki 实体页前，先按论文 ID / 项目页 / 源码查重，避免与既有节点及不同项目同仓实现冲突。
- [x] 通读两篇官方 Wiki 教程与 README（完整正文）。
- [x] 核对官方 CLI 及对应源码，时序图见 wiki。
- [x] 升格 [Kalibr](../../wiki/entities/kalibr.md)，标定流程/代码时序图和局限齐全；健康度待 CI 验证。
