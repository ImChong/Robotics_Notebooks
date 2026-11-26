# 机器人策略学习：一手方法来源

- **通读：** 已完成
- **核对范围：** 2026-11-26，Diffusion Policy arXiv v5 的条件扩散、滚动时域、评测与局限，Octo arXiv v2 的微调及评估、附录 F，OpenAI Spinning Up 的 RL 问题定义、轨迹与回报相关小节。此处核对方法和协议，未复现训练或真机结果。

## 一手来源

- [Diffusion Policy，arXiv:2303.04137v5](https://arxiv.org/html/2303.04137v5)：分别定义观测历史、动作预测、动作实际执行三个时域；执行动作块的前缀后用新观测重新预测。训练去噪损失不是闭环任务成功率，论文分别做仿真与真实任务评测。
- [Octo，arXiv:2405.12213v2](https://arxiv.org/html/2405.12213v2)：将预训练分布内的多本体开箱评估、少量目标演示微调、新接口接入分开报道。Berkeley Bimanual 任务的 64 步预测、12 步执行是特定真机实验配置，不是通用动作块默认值。
- [OpenAI Spinning Up：Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)：按观测、动作、状态转移、轨迹和回报说明强化学习的目标；对奖励目标的优化与实际验收标准应在工程上分别定义。
