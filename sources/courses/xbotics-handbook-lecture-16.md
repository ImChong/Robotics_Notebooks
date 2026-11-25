# Xbotics 第16讲：Off-policy 强化学习 —— 从仿真提速到真机落地 / Xbotics Lecture 16: Off-Policy Reinforcement Learning: From Faster Simulation to Real-Robot Deployment

- **类型：** 官方教程章节 / course
- **维护者：** Xbotics 具身智能社区
- **官方章节：** https://xbotics-embodied-ai-club.github.io/Xbotics-Embodied-AI-Handbook/part4-reinforcement-learning/16-%E7%9C%9F%E6%9C%BA%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0.html
- **固定版本 Markdown：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/docs/part4-reinforcement-learning/16-%E7%9C%9F%E6%9C%BA%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0.md
- **核对范围：** 官方章节完整正文及参考资料列表；不把课程所引第三方工作计为独立通读。
- **通读：** 已完成
- **代码入口：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/code/rl/1_rl_basics/1_1_g1_walk_rl/train_v4_sac.py
- **验证边界：** 文稿核验与源码静态走读，不代表本次重训、复现实验或真机运行。

## 编译线索

课程从真机采样成本出发，依次讲解表格 Q-learning、DQN、DDPG、TD3、SAC 与像素策略 squint，并回到 G1 仿真和 SO-101 真机 HIL-SERL 的不同验证口径。

课上 SAC 的简单仿真结果、FastTD3/FastSAC 的报道及 HIL-SERL 论文真机数字不是同一个硬件/基准；SO-101 真机章节提供复现路径而不是课程作者已完整跑出的同样战绩。

## 对 wiki 的映射

- [第 16 讲独立知识节点](../../wiki/entities/xbotics-handbook-lecture-16.md)
- [课程主节点](../../wiki/entities/xbotics-embodied-ai-handbook.md)
