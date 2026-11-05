# ODE Policy（arXiv:2611.01729）

> 来源归档（paper）

- **标题：** ODE Policy: Robot Learning via Action Dynamics, Not Actions
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.01729>
- **PDF：** <https://arxiv.org/pdf/2611.01729>
- **项目页：** <https://ntumars.github.io/ODE/>
- **代码：** <https://github.com/NTUMARS/ODE-POLICY>
- **入库日期：** 2026-11-04
- **线索来源：** [机器人做错了，奖励模型为什么还在加分？｜附代码与评测入口](../blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)（具身智能小站，2026-11-04）
- **一句话说明：** 动作块边界的跳变会妨碍平稳控制。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-04）：仓库可达（git ls-remote）。项目页 HTTP 200。

## 核心摘录（公众号转述，未核验原文表格）

- 让动作沿连续动力学自然衔接。ODE Policy直接学习动作变化的向量场，从上一动作积分得到下一动作；覆盖模仿学习、动作模型和强化学习，适合关注平滑性与推理效率的团队。
- 动作块边界的跳变会妨碍平稳控制。
- 指标：公众号转述：平滑性要连同推理条件比较：作者在14项真机任务中测试，并在单RTX 4090的药片操作实验中报告动作块生成10.3 ms，对比FM-DiT的36.3 ms；这不是所有平台通用的加速倍数。
- §1、§2.1–2.2、§4.3、§5、§6.1.1–6.1.3、图 4–6：用于实体页「与其他工作对比」（2026-11-05 补）。

## 对 wiki 的映射

- [paper-ode-policy-action-dynamics](../../wiki/entities/paper-ode-policy-action-dynamics.md)
- [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-04-57-papers-technology-map.md)
