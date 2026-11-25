# Xbotics 第2讲：机器人系统架构：硬件、软件与 ROS2 / Xbotics Lecture 2: Robot System Architecture: Hardware, Software and ROS 2

- **类型：** 官方教程章节 / course
- **维护者：** Xbotics 具身智能社区
- **官方章节：** https://xbotics-embodied-ai-club.github.io/Xbotics-Embodied-AI-Handbook/part1-system-basics/02-ros2-architecture.html
- **固定版本 Markdown：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/docs/part1-system-basics/02-ros2-architecture.md
- **核对范围：** 官方章节完整正文及参考资料列表；不把课程所引第三方工作计为独立通读。
- **通读：** 已完成
- **代码入口：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/code/lecture02/simulation/robot_ws/src/robot_demo/launch/minimal_loop.launch.py
- **验证边界：** 文稿核验与源码静态走读，不代表本次重训、复现实验或真机运行。

## 编译线索

以六关节 mock 机械臂把任务、策略、控制、机器人状态和日志拆成 ROS 2 节点；对比 Topic、Service、Action 与 Launch 的适用场景。

mock 路径不等于硬件驱动；实际接入 SO101 时需核验厂商接口、单位、限位和异常停机。 讲义 §2.7 仍写完整包未发布，但同一提交中的配套目录已包含可指认的 launch 与六节点源码。

## 对 wiki 的映射

- [第 2 讲独立知识节点](../../wiki/entities/xbotics-handbook-lecture-02.md)
- [课程主节点](../../wiki/entities/xbotics-embodied-ai-handbook.md)
