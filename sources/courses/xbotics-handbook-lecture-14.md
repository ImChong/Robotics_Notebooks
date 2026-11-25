# Xbotics 第14讲：强化学习入门 —— 从策略梯度到 PPO / Xbotics Lecture 14: Introduction to Reinforcement Learning: From Policy Gradients to PPO

- **类型：** 官方教程章节 / course
- **维护者：** Xbotics 具身智能社区
- **官方章节：** https://xbotics-embodied-ai-club.github.io/Xbotics-Embodied-AI-Handbook/part4-reinforcement-learning/14-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E5%85%A5%E9%97%A8.html
- **固定版本 Markdown：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/docs/part4-reinforcement-learning/14-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E5%85%A5%E9%97%A8.md
- **核对范围：** 官方章节完整正文及参考资料列表；不把课程所引第三方工作计为独立通读。
- **通读：** 已完成
- **代码入口：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/code/rl/1_rl_basics/1_1_g1_walk_rl/train_v3_ppo.py
- **验证边界：** 文稿核验与源码静态走读，不代表本次重训、复现实验或真机运行。

## 编译线索

在仿真 G1 速度跟踪的相同任务和预算下对比 REINFORCE、A2C 和 PPO，再把算法迁到参考动作跟踪。

课程的速度跟踪对比统一预算；动作跟随中的原版 PPO/BeyondMimic 权重训练预算不同，只可作能力参照。控制策略与奖励曲线均不直接证明真机安全。

## 对 wiki 的映射

- [第 14 讲独立知识节点](../../wiki/entities/xbotics-handbook-lecture-14.md)
- [课程主节点](../../wiki/entities/xbotics-embodied-ai-handbook.md)
