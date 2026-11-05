# FAITH: Feasibility-Aware Safety-Filtered RL for High-Dimensional Systems

> 论文来源归档；基于 arXiv v1 正文。

- **作者：** Songyuan Zhang、Baljeet Singh、Sarthak Ranjeet Kaingade、Chuchu Fan、Bryan Trinh
- **机构：** 麻省理工学院（MIT）、Amazon
- **版本：** arXiv:2610.12432v1，2026-10-08
- **原文：** https://arxiv.org/html/2610.12432v1
- **论文入口：** https://arxiv.org/abs/2610.12432
- **知识页：** [FAITH](../../wiki/entities/paper-faith-safety-filtered-rl.md)

## 原文要点

- §II–III：以轨迹最大伤害定义可行性；安全 Q 网络及 backup actor 估计后续最小峰值伤害，残差网络以动作偏差与安全惩罚训练过滤器；PPO 在过滤后的交互中只优化任务回报。
- §III：完美过滤器与全局策略优化下的等价结论不直接保证实际学习控制器的安全；折扣、近似及有限惩罚均带来误差。
- §V：双积分器和 Safety Gym 各训练 3 个随机种子；人形实验在 Isaac Lab 中采用 29 自由度 G1，保护区为无限高度平面区域。Walking-Avoid 与 Push-Avoid 各评估 4096 条轨迹。
- §V-C：Walking-Avoid 安全率 99.95%，回报 2365±345；无过滤回报 2428±288。Push-Avoid 安全率 85.45%，共同失败子集峰值侵入 0.153±0.184；共同失败不代表物理不可行。
- §V-D：1025 动作数值采样在双积分器与 Car-Circle 中慢 18 倍；过滤动力学训练、backup replay 与惩罚权重均有消融。
- §VI–VII：G1 真机实验为定性演示；10 Hz LiDAR、50 Hz 策略、1000 Hz PD。未给出真机安全统计、感知噪声及延迟影响的量化结果。
