# Calibrex

> 来源归档：官方代码仓库

- **链接：** <https://github.com/rsasaki0109/Calibrex>
- **类型：** 开源多传感器标定工具，Python 3.10+
- **许可证：** Apache-2.0；状态为 alpha。
- **核查日期：** 2026-10-27
- **文档：** <https://rsasaki0109.github.io/Calibrex/>
- **一句话说明：** 基于录制数据估计、检查和追踪传感器外参，并报告各轴可观测性。

## 开源核查与入口

官方仓库公开源码与 CLI，提供 `calibrex estimate`、`check`、`drift` 和浏览器工具。README 区分发布版 v0.5.1 与 main 分支：ROS 1、部分 MCAP 压缩支持、camera-LiDAR 检查等功能在该发布版之后，使用前应核对版本。支持的 bag 格式和传感器对详见官方文档。

## 工程边界

检查使用 bag 的 `/tf_static` 或外部标定文件作为候选；结果按轴提供 pass、warn、fail 或 inconclusive。camera-LiDAR 当前只检查旋转；平面行驶不能观测车辆对的 roll。KITTI 示例属于开发数据和注入误差实验，不代表所有轴均得到独立验证。

## 对 wiki 的映射

- [Calibrex 多传感器标定](../../wiki/entities/calibrex.md)
