# From Hitter to Player: Separating Motor Execution and Strategy in Competitive Humanoid Table Tennis

> 来源归档（ingest · arXiv 预印本）

- **类型：** paper / humanoid robotics / table tennis / motor control / competitive self-play
- **版本：** arXiv:2610.14067v1，2026-10-07
- **作者：** Fanrong Dong, Ziyi Sun, Ziqi Han, Junhan Sun, Lei Ye, Zetong Jing, Hao Zhao, Shaoxun Liu, Rongrong Wang
- **单位：** 浙江大学、中国海洋大学、同济大学、上海创智学院、清华大学、RoboParty Lab（按提交者提供信息）
- **论文：** [arXiv abstract](https://arxiv.org/abs/2610.14067) · [HTML](https://arxiv.org/html/2610.14067v1) · [PDF](https://arxiv.org/pdf/2610.14067v1)
- **项目页：** [From Hitter to Player](https://mufanmufan.github.io/hitter-to-player/)
- **代码状态：** 未核实。arXiv 列有项目页，但本次检查无法访问该页面；未确认官方代码仓库或权重链接。
- **提交日期：** 2026-10-07
- **入库日期：** 2026-10-23
- **一句话说明：** 将人形乒乓球的动作执行与落点决策拆开：先训练以目标落点为条件的全身击球策略，再冻结动作栈、单独通过竞争式自博弈学习选点策略。

## 核心摘录（策展）

论文把竞技乒乓球拆成两个问题：机器人能否可靠地执行预期回球，以及应该把球打向哪里。球的落点目标是二者之间的接口。

1. **先学可执行击球。** 基于模型的击球规划器与学习控制器将期望落点转换为人形机器人的全身击球动作；运动策略以落点目标为条件。
2. **闭环动作自博弈。** 通过与另一位球员生成的来球进行连续回合自博弈，提高落点准确度和合法回球率。此阶段的落点仍随机采样，因此改进的是动作执行能力，而非对手策略。
3. **冻结动作栈后学战术。** 固定整个运动执行栈，只训练逐拍选择落点的策略，并通过竞争式自博弈优化得分表现。

论文摘要报告，学习到的落点策略相对“打向中间”“随机落点”和“远离对手”基线，分别赢得 89%、82% 和 73% 的分数。作者还在真实硬件上与休闲人类对手完成 11 分制比赛，并展示包含正手与反手的多拍回合。

## 对 wiki 的映射

- [From Hitter to Player：人形乒乓球中的动作执行与策略分离](../../wiki/entities/paper-from-hitter-to-player-humanoid-table-tennis.md)
- [Table Tennis Strategy & Skill Learning（PhysicsPingPong）](../../wiki/methods/table-tennis-strategy-skill-learning.md) — 与分层技能/落点策略工作对照

## 原始入口

- arXiv:2610.14067v1：标题、作者、提交日期、摘要、项目页链接与论文正文。
- 项目页：arXiv 指向该站点；本次检查无法访问，代码和数据发布状态待核实。
