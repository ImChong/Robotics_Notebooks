# Xbotics 第5讲：仿真环境与操作任务搭建 / Xbotics Lecture 5: Simulation Environments and Task Setup: From Real Tasks to Reproducible Loops

- **类型：** 官方教程章节 / course
- **维护者：** Xbotics 具身智能社区
- **官方章节：** https://xbotics-embodied-ai-club.github.io/Xbotics-Embodied-AI-Handbook/part2-vision-manipulation/05-simulation.html
- **固定版本 Markdown：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/docs/part2-vision-manipulation/05-simulation.md
- **核对范围：** 官方章节完整正文及参考资料列表；不把课程所引第三方工作计为独立通读。
- **通读：** 已完成
- **代码入口：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/code/lecture05/simulation/reach.py
- **验证边界：** 文稿核验与源码静态走读，不代表本次重训、复现实验或真机运行。

## 编译线索

先为操作任务写契约，再用 MuJoCo SO-101 到达与抓放兑现它，扩展到 Isaac Lab Franka Reach、G1 速度跟踪和 Sim2Real 扰动。

本地仿真真值不代表真机可观测；接触、执行器、延迟和视觉可能各自引起域差，单一成功样例不足以证明迁移。

## 对 wiki 的映射

- [第 5 讲独立知识节点](../../wiki/entities/xbotics-handbook-lecture-05.md)
- [课程主节点](../../wiki/entities/xbotics-embodied-ai-handbook.md)
