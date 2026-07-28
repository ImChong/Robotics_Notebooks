# krishanrana.github.io/reskill（ReSkill 项目页）

- **标题：** Residual Skill Policies: Learning an Adaptable Skill-based Action Space for Reinforcement Learning for Robotics
- **类型：** site / project-page
- **URL：** <https://krishanrana.github.io/reskill/>
- **配套论文：** [arXiv:2211.02231](https://arxiv.org/abs/2211.02231)，CoRL 2022
- **代码：** <https://github.com/krishanrana/reskill>（MIT License）— 归档见 [`sources/repos/reskill.md`](../repos/reskill.md)
- **入库日期：** 2026-07-28

## 一句话摘要

Rana, Xu, Tidd, Milford, Sünderhauf（QUT Centre for Robotics / CSIRO Data61）的 ReSkill 官方项目页：在已有技能空间之上加**低层残差策略**，配合 normalizing flows 状态条件技能先验加速探索，使预训练技能能适应新任务变体（障碍、物体变化、摩擦变化）；页面声明 Code 与视频可用。

## 公开信息要点（截至入库日）

- **机构：** QUT Centre for Robotics（Queensland University of Technology）；Data61 Robotics and Autonomous Systems Group, CSIRO。
- **页面内容：** 方法图（skill prior + residual policy 双层结构）、四个 Fetch 臂下游任务视频、代码链接。
- **任务设定：** 下游任务改编自 Silver et al. RPL 环境族（Slippery-Push、CleanUp、Pyramid-Stack、Complex-Hook），均带技能提取时**未见过的**物理/动力学变体。
- **代码开放度：** **已开源**（PyTorch 官方实现，MIT）。

## 为何值得保留

ReSkill 把 Residual 思想从「单任务控制器打底」提升到「**技能空间打底**」：残差策略恢复对原子动作空间的细粒度访问，缓解 skill-based RL 的 generality–sample-efficiency 权衡；是分层/技能学习方向理解残差机制的关键节点。

## 对 wiki 的映射

- 实体页：[paper-reskill-residual-skill-policies](../../wiki/entities/paper-reskill-residual-skill-policies.md)
- 方法页：[residual-policy-learning](../../wiki/methods/residual-policy-learning.md)
