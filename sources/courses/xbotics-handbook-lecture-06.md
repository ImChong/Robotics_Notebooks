# Xbotics 第6讲：机器人感知与位姿估计 / Xbotics Lecture 6: Robot Perception and Pose Estimation: From Pixels to Grasp Poses

- **类型：** 官方教程章节 / course
- **维护者：** Xbotics 具身智能社区
- **官方章节：** https://xbotics-embodied-ai-club.github.io/Xbotics-Embodied-AI-Handbook/part2-vision-manipulation/06-pose-estimation.html
- **固定版本 Markdown：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/docs/part2-vision-manipulation/06-pose-estimation.md
- **核对范围：** 官方章节完整正文及参考资料列表；不把课程所引第三方工作计为独立通读。
- **通读：** 已完成
- **代码入口：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/code/lecture06/Visual_Perception/pipeline_from_syllabus.py
- **验证边界：** 文稿核验与源码静态走读，不代表本次重训、复现实验或真机运行。

## 编译线索

检测框、掩膜、三维中心、物体位姿和抓取位姿各回答不同问题。课程把这些表征连成 RGB-D 观测到机械臂执行的链路，并说明哪些环节可以被直接抓取预测替代。

点云课堂入口假定已有掩膜，不是检测、分割、6D 定姿或执行的一体化实现。真实遮挡、反光和手眼外参偏差仍需单独评测；高置信度框不保证可抓。 教材中的阈值和成功率量级不能外推为平台安全门槛。

## 对 wiki 的映射

- [第 6 讲独立知识节点](../../wiki/entities/xbotics-handbook-lecture-06.md)
- [课程主节点](../../wiki/entities/xbotics-embodied-ai-handbook.md)
