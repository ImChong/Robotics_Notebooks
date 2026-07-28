# Humanoid Parkour Learning 官方项目页

> 来源归档（项目页核查）

- **标题：** Humanoid Parkour Learning
- **类型：** site / project-page
- **官方入口：** <https://humanoid4parkour.github.io/>
- **论文：** <https://arxiv.org/abs/2406.10759>
- **机构：** 上海创智学院、上海科技大学、清华大学
- **入库日期：** 2026-07-28
- **一句话说明：** 展示 Unitree H1 单一视觉全身策略完成十类地形、0.42 m 跳台、0.8 m 跨沟与 1.8 m/s 野外跑。
- **开源状态（2026-07-28 核查）：** **未开源**；项目页只提供论文和视频，没有本论文 GitHub、权重或数据链接。

## 重要辨析

作者的 <https://github.com/ZiwenZhuang/parkour> 是前作 **Robot Parkour Learning** 的四足 A1/Go2 代码，不是本文 Unitree H1 的 Humanoid Parkour Learning 实现，不应据此标记本文“已开源”。

## 页面公开信息

- 单一策略覆盖 jump up、stairs、tilted ramp、jump down、leap、slope 与 robust walking。
- 无动作参考，先训练特权 scandots oracle，再以 DAgger 蒸馏深度图学生。
- 可覆盖手臂策略输出，把下肢跑酷控制接到移动操作任务。

## 对 wiki 的映射

- 论文归档：[humanoid_pnb_humanoid-parkour-learning.md](../papers/humanoid_pnb_humanoid-parkour-learning.md)
- 实体页：[paper-notebook-humanoid-parkour-learning.md](../../wiki/entities/paper-notebook-humanoid-parkour-learning.md)
