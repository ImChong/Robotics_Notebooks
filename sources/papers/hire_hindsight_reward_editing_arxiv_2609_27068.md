# HiRE（arXiv:2609.27068）

> 来源归档（paper）

- **标题：** HiRE: Hindsight Reward Editing for Policy Finetuning
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.27068>
- **PDF：** <https://arxiv.org/pdf/2609.27068>
- **项目页：** <https://hire-project.github.io/>
- **代码：** <https://github.com/HiRE-Project/HiRE>
- **入库日期：** 2026-10-18
- **线索来源：** [机器人会写控制程序之后，能否自己造训练数据？](../blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)（具身智能小站，2026-10-18）
- **一句话说明：** 用成功与失败轨迹对照，压低“看似接近目标却会失败”的状态奖励，给策略微调补上控制相关反馈。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-18）：仓库与项目页均可达，项目页链接 GitHub 仓库。

## 核心摘录（公众号转述，未核验原文表格）

- 视觉相似不等于任务做对；基于视觉相似度的奖励会把接近但会失败的状态打高分，误导微调。
- 事后（hindsight）对照成功与失败轨迹，编辑视觉奖励：压低失败轨迹中高相似度状态的奖励，使奖励与控制相关。
- 指标：公众号称在仿真和真机上比较；未给具体数字，本页不引用。

## 对 wiki 的映射

- [paper-hire-hindsight-reward-editing](../../wiki/entities/paper-hire-hindsight-reward-editing.md)
- [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md)
