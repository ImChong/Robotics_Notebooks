# Minute-scale training for microrobot navigation

> 来源归档（ingest）

- **标题：** Minute-scale training for microrobot navigation
- **类型：** paper / microrobot / navigation / deep reinforcement learning / sim-to-real
- **正式发表：** *Nature Machine Intelligence*, 2026-09-28
- **DOI：** <https://doi.org/10.1038/s42256-026-01305-w>
- **论文原文：** <https://www.nature.com/articles/s42256-026-01305-w>
- **预印本：** <https://arxiv.org/abs/2608.00854>
- **官方代码：** <https://github.com/yinghansun/mr-nav>（归档见 [`sources/repos/yinghansun-mr-nav.md`](../repos/yinghansun-mr-nav.md)）
- **作者：** Yinghan Sun, Aoji Zhu, Xiang Ji, Yamei Li, Jiachi Zhao, Yun Wang, Li Zhang, Huijun Gao, Lidong Yang
- **机构：** 香港理工大学、香港中文大学、哈尔滨工业大学
- **入库日期：** 2026-10-21
- **一句话说明：** 通过完全向量化的血管环境模拟器和 task-shaping-regularization（TSR）奖励，把微型机器人导航的深度强化学习训练压缩到 10 分钟以内；论文报告超过 10,000 个仿真血管环境、约 190,000 transitions/s，并展示跨微机器人形态与未见环境的零样本迁移。

## 摘要级要点

- **问题：** 微型机器人导航的深度强化学习往往训练数小时至数天，拖慢参数优化和实际部署。
- **并行仿真：** 向量化环境并行处理机器人动力学、受 LiDAR 启发的感知和可行性检查；论文摘要称其覆盖超过 10,000 个人工血管环境，吞吐约 190,000 transitions/s。
- **奖励设计：** TSR（task-shaping-regularization）将任务目标、逐步进度 shaping 与轨迹正则化结合。论文报告动作变化至少降低 33.7%，障碍物间隙至少增加 2.1%，并加快收敛。
- **迁移验证：** 策略在未针对具体任务微调的情况下，迁移到不同微机器人形态和导航场景；实验包含人工血管网络、含障碍的三维场景及脑动脉模型通道。
- **开源实现：** 官方仓库提供训练、评估脚本、配置与示例数据；README 给出的典型训练时间约 8–10 分钟（RTX 3060），训练建议使用至少 8 GB 显存 GPU。

## 对 wiki 的映射

- 沉淀实体页：[Minute-scale training for microrobot navigation](../../wiki/entities/paper-minute-scale-microrobot-navigation.md)
- 补强方法页：[强化学习](../../wiki/methods/reinforcement-learning.md)（连续导航中的大规模并行采样、奖励塑形与仿真到实机迁移）

## 当前提炼状态

- [x] 论文元数据、摘要贡献与评测结论摘录
- [x] 核实官方代码仓并归档其训练入口
- [x] wiki 实体页与强化学习方法页交叉链接
