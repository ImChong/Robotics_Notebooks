# RoboStriker: Hierarchical Decision-Making for Autonomous Humanoid Boxing

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** RoboStriker: Hierarchical Decision-Making for Autonomous Humanoid Boxing
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/RoboStriker__Hierarchical_Decision-Making_for_Autonomous_Humanoid_Boxing/RoboStriker__Hierarchical_Decision-Making_for_Autonomous_Humanoid_Boxing.html>
- **分类：** 04_Loco-Manipulation_and_WBC
- **arXiv：** <https://arxiv.org/abs/2601.22517>
- **入库日期：** 2026-06-07
- **一句话说明：** RoboStriker 把"两个人形机器人互殴"建成 两玩家零和马尔可夫博弈，先用单智能体追真人拳击 MoCap 训出 运动跟踪器（46 段、约 14 分钟 Xsens 数据，经 GMR 重定向到 Unitree G1），再把这些技能蒸馏成一个 投到单位超球面的潜空间动作流形，最后在这个潜空间上跑 Latent-Space Neural Fictitious Self-Play (LS-NFSP)，让两个智能体只挑"高层动作意图"而不直接挑电机指令——动作天然物理可行又像人，多智能体训练也稳定收敛。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-robostriker](../../wiki/entities/paper-notebook-robostriker.md).

## 对 wiki 的映射

- [paper-notebook-robostriker](../../wiki/entities/paper-notebook-robostriker.md)
- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../../wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/RoboStriker__Hierarchical_Decision-Making_for_Autonomous_Humanoid_Boxing/RoboStriker__Hierarchical_Decision-Making_for_Autonomous_Humanoid_Boxing.html>
- 论文：<https://arxiv.org/abs/2601.22517>
