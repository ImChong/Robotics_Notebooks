# Xbotics 第9讲：操作数据闭环 —— 采一份能训练的数据集 / Xbotics Lecture 9: Robot Manipulation Data Loop: Collecting a Trainable Dataset

- **类型：** 官方教程章节 / course
- **维护者：** Xbotics 具身智能社区
- **官方章节：** https://xbotics-embodied-ai-club.github.io/Xbotics-Embodied-AI-Handbook/part3-end-to-end/09-%E6%93%8D%E4%BD%9C%E6%95%B0%E6%8D%AE%E9%97%AD%E7%8E%AF.html
- **固定版本 Markdown：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/docs/part3-end-to-end/09-%E6%93%8D%E4%BD%9C%E6%95%B0%E6%8D%AE%E9%97%AD%E7%8E%AF.md
- **核对范围：** 官方章节完整正文及参考资料列表；不把课程所引第三方工作计为独立通读。
- **通读：** 已完成
- **代码入口：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/code/vla/2_data_collection/2_3_teleop_record/record.sh
- **验证边界：** 文稿核验与源码静态走读，不代表本次重训、复现实验或真机运行。

## 编译线索

比较原机遥操、人类代理、通用接口与无机器人采集；以 SO-101 Leader/Follower 演示从标定、遥操作到 LeRobot 数据集录制和回看。

课中真机例为 30 Hz、640×480 绝对关节目标；键盘仿真例为 20 Hz、128×128 归一化关节增量。六维动作形状和容器相同并不意味语义兼容，合训前须明确转换与验收；不要靠采更多来修复标定或时间戳错误。

## 对 wiki 的映射

- [第 9 讲独立知识节点](../../wiki/entities/xbotics-handbook-lecture-09.md)
- [课程主节点](../../wiki/entities/xbotics-embodied-ai-handbook.md)
