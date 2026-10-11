# RoboDuet: Learning a Cooperative Policy for Whole-body Legged Loco-Manipulation（arXiv:2403.17367）

- **类型：** 论文 / legged loco-manipulation / whole-body control
- **arXiv：** https://arxiv.org/abs/2403.17367
- **PDF：** https://arxiv.org/pdf/2403.17367
- **期刊：** IEEE Robotics and Automation Letters, 10(5):4564–4571, 2025
- **DOI：** https://doi.org/10.1109/LRA.2025.3551230
- **项目页：** https://locomanip-duet.github.io/
- **官方训练代码：** https://github.com/locomanip-duet/RoboDuet
- **官方部署代码：** https://github.com/locomanip-duet/RoboDuet_Deployment
- **作者：** Guoping Pan、Qingwei Ben、Zhecheng Yuan、Guangqi Jiang、Yandong Ji、Shoujie Li、Jiangmiao Pang、Houde Liu、Huazhe Xu
- **机构：** 清华大学、上海期智研究院、上海人工智能实验室、江淮前沿技术中心、加州大学圣迭戈分校（按项目页列出的作者机构）
- **项目页标注：** Guoping Pan 与 Qingwei Ben 共同贡献；项目页称论文被 RA-L 2025 接收。
- **入库日期：** 2026-10-11

## 论文要点

- RoboDuet 面向配备机械臂的四足机器人，目标是在移动过程中实现对末端位姿的精确跟踪和全身协调。
- 方法由 locomotion policy 与 arm policy 组成。前者跟踪移动命令并响应后者的指引，后者跟踪末端 6D 位姿；机械臂关节偏移与姿态命令共同构成其动作设计。
- 训练分两阶段：先训练稳健盲行走，再训练 locomotion 与 manipulation 的协同控制。
- 项目页报告：Go1+ARX5 上训练的策略零样本部署到 Go2+ARX5；Go2 底盘重量增加 14.7%；困难移动操作任务的成功率较 AFB+ID 基线提升 23%。
- 官方项目页演示坡面、碎石、100 cm 高门把手开门，以及在 0/20/60/100 cm 高度间搬运物体等任务。

## 摘录与映射

1. 论文摘要提出“two collaborative policies”共同实现 locomotion 与 manipulation，并以相互作用形成 whole-body control。
   - **对 wiki 的映射：** 解释 RoboDuet 的问题设定与双策略协作架构，映射到 [RoboDuet 实体页](../../wiki/entities/paper-roboduet-loco-manipulation.md) 的「为什么重要」「核心方法」。

2. 项目页说明 loco policy 跟踪目标命令并根据 arm policy 的指引调整动作；arm policy 的目标是追踪末端 6-DoF 位姿。
   - **对 wiki 的映射：** 对照两策略输入、输出和协作关系，映射到实体页「核心方法」与流程图。

3. 项目页称训练采用 two-stage strategy，第一阶段获得稳健 locomotion，第二阶段同时启用 arm policy 与机械臂关节学习移动操作；并报告 23% 成功率提升和 Go1 到 Go2 的零样本迁移。
   - **对 wiki 的映射：** 映射到「两阶段训练」「实验与评测」「结论」，并限定数字与迁移证据的适用范围。

## 对 wiki 的映射

- 项目实体：[RoboDuet](../../wiki/entities/paper-roboduet-loco-manipulation.md)
- 任务入口：[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)
- 官方项目页来源：[项目页归档](../sites/roboduet.md)
- 官方代码来源：[训练仓库](../repos/roboduet.md)、[真机部署仓库](../repos/roboduet_deployment.md)
