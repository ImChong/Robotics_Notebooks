# Robot Learning Overview：强化学习与控制的一手课程章节

- **强化学习：** OpenAI Spinning Up，[Part 1: Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- **控制：** Russ Tedrake 的 MIT Underactuated Robotics，[Ch. 1: Fully-actuated vs Underactuated Systems](https://underactuated.mit.edu/intro.html)
- **核对范围：** 通读上述两个独立章节的正文，含状态与观测、动作策略、轨迹、回报、价值函数和 Bellman 关系，以及欠驱动定义、实例、输入与状态约束；不宣称通读整套教材或嵌入视频。
- **通读：** 已完成

Spinning Up 区分环境内部状态与策略可获得的观测，并将强化学习目标表述为策略诱导的轨迹下的期望回报。这个定义不把训练奖励当作独立业务指标，也不要求把策略动作直接解释为关节力矩。

MIT 课程从机械系统的动力学、可施加输入和约束解释欠驱动。即使每个关节都配有执行器，浮动基座或输入限幅也会改变即时可实现的运动。课程章节并不为任何特定学习策略提供稳定性或安全保证。

原文支持概览的策略、反馈与物理约束区分。页面的三层架构和部署检查是结合知识库已有工程资料整理的阅读框架，并非两章节提出的新算法。

**对应 wiki：** [机器人学习概览](../../wiki/overview/robot-learning-overview.md)
