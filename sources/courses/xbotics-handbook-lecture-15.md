# Xbotics 第15讲：GRPO 后训练 —— 让 VLM 学会数数、让 VLA 自我提升 / Xbotics Lecture 15: GRPO Post-Training: Counting with VLMs and Improving VLAs

- **类型：** 官方教程章节 / course
- **维护者：** Xbotics 具身智能社区
- **官方章节：** https://xbotics-embodied-ai-club.github.io/Xbotics-Embodied-AI-Handbook/part4-reinforcement-learning/15-GRPO%E5%90%8E%E8%AE%AD%E7%BB%83.html
- **固定版本 Markdown：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/docs/part4-reinforcement-learning/15-GRPO%E5%90%8E%E8%AE%AD%E7%BB%83.md
- **核对范围：** 官方章节完整正文及参考资料列表；不把课程所引第三方工作计为独立通读。
- **通读：** 已完成
- **代码入口：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/code/rl/2_grpo_posttraining/2_1_grpo_vlm_counting/train_grpo_qwen2_vl.py
- **验证边界：** 文稿核验与源码静态走读，不代表本次重训、复现实验或真机运行。

## 编译线索

先以可判分的视觉计数演示组内相对策略更新，再对照 SimpleVLA-RL、RLinf 和从真实经验学习的 RECAP。

全成功或全失败的组缺少相对奖励信号。没有可可靠计算的动作 log 概率或无法廉价重置的真机任务，不能原样照搬计数案例；RECAP 不是同一个 GRPO 实现。

## 对 wiki 的映射

- [第 15 讲独立知识节点](../../wiki/entities/xbotics-handbook-lecture-15.md)
- [课程主节点](../../wiki/entities/xbotics-embodied-ai-handbook.md)
