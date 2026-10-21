# Humanoid SRL: Augmenting Humanoids with Supernumerary Limbs for Coordinated Loco-Manipulation

> 来源归档（ingest）

- **标题：** Humanoid SRL: Augmenting Humanoids with Supernumerary Limbs for Coordinated Loco-Manipulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.25628>
- **机构：** Georgia Institute of Technology
- **作者：** Fukang Liu, Feiyang Wu, Paul Barsa, Jaehwi Jang, Akshat Jain, Zhigen Zhao, Yipu Chen, Zhaoyuan Gu, Nikhil Vemuri, Anshul Chandaliya, Kaijie Zhu, Sean Lublinsky, Matthew Zhu, Tony G. Chen, Ye Zhao
- **提交日期：** 2026-10-19
- **入库日期：** 2026-10-21
- **一句话说明：** 在人形机器人上集成超冗余机械臂，结合模型式轨迹优化和目标条件强化学习控制器，使新增肢体与全身运动、平衡协调；论文报告仿真零样本迁移至硬件，并展示基于硬件示范训练的扩散策略完成视觉引导自主执行。

## 摘要要点

- **研究问题：** 人形机器人双臂同时被占用或需要超出双臂可达范围的额外接触时，原有形态会限制移动操作能力。
- **形态扩展：** 在人形平台集成超冗余机械臂（supernumerary robotic limb, SRL），并通过两种配置的定量分析与硬件实验评估其对全身交互能力的影响。
- **控制方法：** 使用模型式轨迹优化（trajectory optimization, TO）生成动态可行运动，再用于目标条件强化学习（reinforcement learning, RL）控制器。
- **迁移与感知：** 控制器在仿真中训练后零样本迁移到硬件；部分任务还使用从硬件示范训练的扩散策略进行视觉引导自主执行。
- **任务重点：** 协调三臂全身移动操作、操控与支撑，以及接触丰富场景下的身体平衡协调。

## 来源与状态

- **论文版本：** arXiv:2610.25628v1，2026-10-19 提交。
- **代码 / 数据：** arXiv 摘要未提供官方代码或数据仓库链接；后续如有官方发布再补录。
- **项目页：** 暂未发现与论文对应的官方项目页。

## 对 wiki 的映射

- [paper-humanoid-srl](../../wiki/entities/paper-humanoid-srl.md) — 论文实体页，概述形态扩展、控制流程和验证边界。

## 参考来源（原始）

- arXiv 摘要与作者信息：<https://arxiv.org/abs/2610.25628>
