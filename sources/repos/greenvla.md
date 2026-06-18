# GreenVLA

> 来源归档

- **标题：** GreenVLA — Staged Vision–Language–Action for Generalist Robot Manipulation
- **类型：** repo
- **组织：** Sber Robotics Center
- **代码：** <https://github.com/greenvla/GreenVLA>
- **项目页：** <https://greenvla.github.io/>
- **论文：** <https://arxiv.org/abs/2602.00919>
- **入库日期：** 2026-06-18
- **一句话说明：** Green-VLA 官方实现：五阶段 VLA 训练、统一多本体动作接口、DataQA 数据管线与 Green 人形部署相关代码与配置入口。
- **沉淀到 wiki：** [Green-VLA（论文实体）](../../wiki/entities/paper-greenvla-staged-vla-humanoid.md)

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [VLA](../../wiki/methods/vla.md) | **L0–R2 分阶段** flow-matching VLA；与 π₀ / GR00T / EO-1 等同赛道对照 |
| [Manipulation](../../wiki/tasks/manipulation.md) | 双臂/人形/单臂统一策略与电商货架 **JPM 引导** 真机任务 |
| [Behavior Cloning](../../wiki/methods/behavior-cloning.md) | R0–R1 以掩码 BC 为主；R2 用 IQL + 噪声分布 RL **突破 BC 饱和** |

## 为何值得保留

- 论文强调 **质量对齐 + 动作统一 + RL 精炼** 而非纯堆数据；开源仓是复现 **DataQA / $\mathcal{A}_u$ / R2** 的直接入口。
- 与 [Humanoid_Robot_Learning_Paper_Notebooks](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks) 单篇深读互补：本库负责 **跨主题 VLA 训练配方** 索引。
