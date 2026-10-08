# FreeSpeed: Training-Free Speed Control for Generative Robot Policies（arXiv:2610.05734）

> 来源归档（ingest）

- **标题：** FreeSpeed: Training-Free Speed Control for Generative Robot Policies
- **arXiv：** <https://arxiv.org/abs/2610.05734>
- **提交日期：** 2026-10-05（v1）
- **作者：** Yuxuan Hu、Shilin Shan、Qiheng Wang、Jinghan Yang、Junqiao Fan、Hao Wan、Jianfei Yang
- **机构：** MARS Lab, Nanyang Technological University；ROKAE Robotics
- **项目页：** <https://yuxuanhu9.github.io/FreeSpeed/>
- **代码：** 项目页截至 2026-10-08 标注 “Code soon”，未链接公开源码仓库；状态归档见 [sources/repos/freespeed.md](../repos/freespeed.md)
- **一句话说明：** 对冻结的生成式机器人策略输出动作块做推理时速度控制，利用动作方向不一致度降低接触关键阶段的调速幅度。

## 摘录：问题与方法

模仿学习策略通常沿用示范的时间尺度；测试时对整段动作作均匀重采样，可能让观测状态偏离训练分布并损害任务成功率。FreeSpeed 不更新基座策略，而在每次重规划时对其动作块进行时间重采样，再依据相邻平移增量的方向不一致度缩放平移和旋转增量，夹爪动作保持不变。

动作块中的方向变化被用作任务阶段关键性的信号：直线、自由运动阶段可更接近用户请求的速度；高方向不一致的抓取和放置阶段则将动作步长拉回基座策略预测的尺度。

## 摘录：评测

- 仿真涵盖三类策略与 50 个任务：π₀.₅、Fast-WAM 和 task-specific flow matching。
- 在各任务仍保持其 1× 基线成功率的设置中，实际执行速率范围为 0.22×–2.53×。
- 四项真机操作任务中，在六种非参考速度命令上平均成功率为 94.0%，冻结策略 1× 参考为 93.8%；逐任务速率为 0.38×–1.97×。
- 真机系统为 ROKAE Helios 系列人形机器人，动作执行频率 30 Hz。

## 当前提炼状态

- [x] 论文元信息和方法摘要
- [x] 项目页与源码开放状态核查
- [x] 映射到单一实体页：[FreeSpeed](../../wiki/entities/paper-freespeed.md)