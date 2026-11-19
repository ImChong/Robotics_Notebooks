# Heave-Ho: Decentralized Long-Horizon Multi-Humanoid Collaboration｜去中心化长程多人形协作

- **类型：** paper
- **论文：** https://arxiv.org/abs/2611.22275
- **原文全文：** https://arxiv.org/html/2611.22275v1
- **原文版本：** arXiv v1；核对日期 2026-11-19
- **核对范围：** arXiv v1/v2 的任务设置、架构与评测以及官方项目页和 GitHub README 的代码发布声明
- **通读：** 已完成
- **项目：** https://heaveho-humanoid.github.io/
- **代码：** https://github.com/heaveho-humanoid/Heave-Ho
- **wiki：** [唯一论文详情](../../wiki/entities/paper-heave-ho-2611-22275.md)

## 核心摘录与知识映射

1. 低层：从冻结的人形全身跟踪基础模型出发，使用多智能体残差后训练处理协作任务中的共享物体反作用力；各机器人以局部观测独立执行策略。 → [方法说明](../../wiki/entities/paper-heave-ho-2611-22275.md)。
2. 高层：领队 VLM 从视觉选择团队技能并用语音广播；跟随机器人识别语音，将技能映射为角色与参考动作，通过共享技能调度时隙同步。 → [方法说明](../../wiki/entities/paper-heave-ho-2611-22275.md)。
3. 训练只使用两台同构人形，论文在仿真展示最多十台及异构团队；HH-Bench 提供 12 类协作任务。 → [方法说明](../../wiki/entities/paper-heave-ho-2611-22275.md)。

## 论文实验事实（非本仓复现）

- HH-Bench 总计 12 项任务，队伍为 2–10 台机器人，包含异构组合。
- 论文/项目展示两台相同人形训练并扩展至最多 10 台，以及真实机器人的协同实验；这些是作者报告。

## 论文局限

- “无网络通信”仅表示不用机器人间消息网络；系统仍依赖语音这个信息通道和感知/时序条件。
- 2026-11-19 官方 GitHub README 明言代码计划 2026 年 12 月底前发布，目前仓库主要是 README 和资产，没有可运行源码，未见明确 LICENSE。
- 仿真规模扩展和实机协作成功不能等同于任意复杂任务上的零样本泛化。
