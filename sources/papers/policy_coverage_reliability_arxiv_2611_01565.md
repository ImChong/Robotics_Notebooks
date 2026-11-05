# Beyond Validation Loss: Coverage and Reliability Dynamics in Robot Policy Pretraining / 超越验证损失：机器人策略预训练的覆盖度与可靠性动态

- **论文：** [arXiv:2611.01565v1](https://arxiv.org/pdf/2611.01565v1)
- **作者：** Zhanyi Sun、Max Simchowitz、Shuran Song
- **机构：** 斯坦福大学（Stanford University）；卡内基梅隆大学（Carnegie Mellon University）
- **项目页：** https://zhanyisun.github.io/robot-policy-learning-dynamics/
- **核对：** 2026-11-03，v1 正文方法、实验和相关实现附录；其余附录待进一步通读。
- **对 wiki 的映射：** [覆盖度与可靠性](../../wiki/entities/paper-policy-coverage-reliability.md)

## 方法与实验摘录

- 固定初始配置独立采样 k 次闭环轨迹，统计至少一次成功的概率并对配置平均；pass@1 是平均成功率，较大 k 反映有限次尝试下的覆盖度。
- 在验证集观测下采样动作，计算各观测的动作误差分位数并求平均；归一化后的低分位和高分位风险取最大值，选取使之最小的 checkpoint。
- JointAug 从演示训练观测—动作联合生成器，训练策略时各使用一半真实与生成样本；RoboCasa 使用计算成本较低的动作扰动变体。
- Push-T 真机比较使用 20 个固定初始状态、每状态五次运行；所选 checkpoint 的 pass@1 / pass@5 为 0.62 / 0.95，最终 checkpoint 为 0.52 / 0.75。
