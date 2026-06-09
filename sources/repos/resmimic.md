# ResMimic（Amazon FAR 人形 GMT→Loco-Manipulation 残差学习）

> 来源归档

- **标题：** ResMimic
- **类型：** repo
- **来源：** Amazon FAR（Frontier AI & Robotics）
- **链接：** <https://github.com/amazon-far/ResMimic>
- **入库日期：** 2026-06-09
- **一句话说明：** ResMimic 论文配套仓库：**GPU 加速仿真基础设施**、**sim-to-sim 评测原型** 与 **运动数据**（论文承诺发布）；实现 **GMT 预训练 + 任务残差策略** 的人形全身 loco-manipulation 管线。
- **沉淀到 wiki：** [`wiki/entities/paper-resmimic.md`](../../wiki/entities/paper-resmimic.md)

---

## 核心定位

**ResMimic** 是 [arXiv:2510.05070](https://arxiv.org/abs/2510.05070) 的官方代码入口，与 [项目页](https://resmimic.github.io/) 配套。方法上在 **大规模 GMT 先验** 之上学习 **物体条件残差**，面向 **Unitree G1** 等 29 DoF 人形的 **跪姿抬箱、背载、蹲起托举、搬椅** 等任务。

---

## 论文承诺的发布内容（README / 项目页）

| 组件 | 说明 |
|------|------|
| 仿真基础设施 | GPU 加速训练与 rollout（论文训练于 **IsaacGym**） |
| sim-to-sim 评测 | 向 **MuJoCo** 迁移评测原型（论文 Table I 主表） |
| 运动数据 | 人–物 MoCap 参考与处理脚本（与 OptiTrack 采集对应） |

> 截至入库日，仓库为论文公开入口；具体目录结构与安装步骤以 GitHub `README` 为准。

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [holosoma](./holosoma.md) | 同机构 **人形 RL + 重定向** 框架；OmniRetarget 侧重 **交互保留参考生成**，ResMimic 侧重 **GMT→物体交互残差** |
| [TWIST](../papers/humanoid_rl_stack_09_twist_teleoperated_whole_body_imitation_system.md) | GMT 奖励与域随机化配方来源 |
| [OmniRetarget](../papers/omniretarget_arxiv_2509_26633.md) | 互补的 loco-manipulation **数据与重定向** 路线 |

## 对 wiki 的映射

- 实体页：[ResMimic（论文）](../../wiki/entities/paper-resmimic.md)
- 任务页：[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)
- 流水线：[Whole-Body Tracking Pipeline](../../wiki/concepts/whole-body-tracking-pipeline.md)
