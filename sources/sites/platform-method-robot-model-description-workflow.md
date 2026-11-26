# ROS 2 Humble URDF Adding Physical and Collision Properties

> 一手资料归档 · 2026-11-26

- [官方教程源文档](https://github.com/ros2/ros2_documentation/blob/humble/source/Tutorials/Intermediate/URDF/Adding-Physical-and-Collision-Properties-to-a-URDF-Model.rst)
- **核对范围：** 官方 Humble 教程全文：Collision、Physical Properties（含 Inertia、Contact Coefficients）、Joint Dynamics 和 Other Tags；未假设所有引擎等价实现扩展字段。
- **通读：** 已完成

## 核对摘记

碰撞几何可以比视觉几何更简单；惯量由质量和分布决定，并可相对连杆坐标另有参考原点。

## 对 wiki 的映射

- [机器人模型描述与资产接入](../../wiki/methods/robot-model-description-workflow.md)
