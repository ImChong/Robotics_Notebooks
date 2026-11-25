# Xbotics 第4讲：传感器与感知基础 / Xbotics Lecture 4: Sensors and Perception Fundamentals

- **类型：** 官方教程章节 / course
- **维护者：** Xbotics 具身智能社区
- **官方章节：** https://xbotics-embodied-ai-club.github.io/Xbotics-Embodied-AI-Handbook/part1-system-basics/04-sensors-coordinates.html
- **固定版本 Markdown：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/docs/part1-system-basics/04-sensors-coordinates.md
- **核对范围：** 官方章节完整正文及参考资料列表；不把课程所引第三方工作计为独立通读。
- **通读：** 已完成
- **代码入口：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/code/lecture04/simulation/sensors_coordinates_demo.ipynb
- **验证边界：** 文稿核验与源码静态走读，不代表本次重训、复现实验或真机运行。

## 编译线索

沿 RGB-D 的一条完整链路讲内外参、时间戳、TF 树和传感器标定，目标是把像素转换成机器人可用的空间目标。

TUM 数据没有本练习的逐点真值，且 notebook 是离线几何练习，不验证实时 ROS TF 或真实手眼标定。

## 对 wiki 的映射

- [第 4 讲独立知识节点](../../wiki/entities/xbotics-handbook-lecture-04.md)
- [课程主节点](../../wiki/entities/xbotics-embodied-ai-handbook.md)
