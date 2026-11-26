# ROS 2 Design · Quality of Service Policies

> 一手资料归档 · 2026-11-26

- [ROS 2 QoS 官方设计说明](https://design.ros2.org/articles/qos)
- **核对范围：** 设计说明全文：请求提供模型、主要策略、默认用例、兼容性和未来工作；不将设计稿视为现有各中间件实测。
- **通读：** 已完成

## 核对摘记

发布订阅的请求与提供策略必须兼容；传感器示例优先及时更新，而非以无限重传追求完整性。

## 对 wiki 的映射

- [机器人软件系统集成](../../wiki/methods/robot-software-integration.md)
