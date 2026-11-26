# ROS REP 103 与 Humble sensor_msgs/Image

> 一手资料归档 · 2026-11-26

- [REP 103 原规范](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst)
- [Image.msg 原消息定义](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/Image.msg)
- **核对范围：** REP 103 全文与 Humble 官方 Image.msg 全文：单位、机身及光学参考系、旋转、协方差和采集时间。
- **通读：** 已完成

## 核对摘记

ROS 常用 SI 与右手坐标，光学帧轴与机体不同。Image.msg 规定采集时间和光学帧，另外定义步长及数据像素编码。

## 对 wiki 的映射

- [机器人传感器接口与集成](../../wiki/methods/robot-sensor-integration.md)
