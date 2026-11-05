# PolicyIK（arXiv:2611.05372）

> 来源归档（paper）

- **标题：** PolicyIK: High-Precision Inverse Kinematics for Robot Arms with Reinforcement Learning
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.05372>
- **PDF：** <https://arxiv.org/pdf/2611.05372>
- **项目页：** <https://tinaliu0123.github.io/PolicyIK/>
- **代码：** <https://github.com/Tinaliu0123/PolicyIK>
- **入库日期：** 2026-11-05
- **线索来源：** [人能稳稳提起的罐子，换一双机器手怎么提？](../blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)（具身智能小站，2026-11-05）
- **一句话说明：** 机械臂够到目标后还可能慢慢漂开。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-05）：仓库可达（git ls-remote）。项目页 HTTP 200。

## 核心摘录（公众号转述，未核验原文表格）

- 机械臂到位后，还能稳稳停住吗。PolicyIK 让策略记住关节指令与实际位置的差，再按剩余误差调整动作幅度。
- 机械臂够到目标后还可能慢慢漂开。
- 指标：公众号转述：评测取最后一秒的平均误差，把“停得稳”也算进成绩。

## 对 wiki 的映射

- [paper-policyik](../../wiki/entities/paper-policyik.md)
- [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-05-43-papers-technology-map.md)
