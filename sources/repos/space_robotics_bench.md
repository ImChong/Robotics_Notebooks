# AndrejOrsula/space_robotics_bench

> 来源归档

- **标题：** Space Robotics Bench（SRB）
- **类型：** repo
- **代码：** <https://github.com/AndrejOrsula/space_robotics_bench>
- **项目页：** <https://AndrejOrsula.github.io/space_robotics_bench>
- **论文：** [arXiv:2608.23452](https://arxiv.org/abs/2608.23452) — 归档见 [`sources/papers/reward_free_continual_adaptation_space_arxiv_2608_23452.md`](../papers/reward_free_continual_adaptation_space_arxiv_2608_23452.md)
- **入库日期：** 2026-08-26
- **一句话说明：** Isaac Lab 上的地外任务套件（并行仿真、程序化生成、域随机化、Gymnasium、ROS 2）；无奖励持续适应论文把 DreamerV3 配方挂在本仓。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [无奖励持续适应](../../wiki/entities/paper-reward-free-continual-adaptation-space.md) | 实体归纳页 |
| [DreamerV3](../../wiki/entities/paper-shenlan-wm-13-dreamerv3.md) | RSSM 想象 RL 骨干 |
| [Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md) | 仿真宿主 |

## 复现入口（README 摘要）

- 文档：<https://AndrejOrsula.github.io/space_robotics_bench>
- 安装脚本：`scripts/install_isaacsim.bash`、`scripts/install_isaaclab.bash`
- DreamerV3 超参：`scripts/dreamerv3.yaml`
- 许可：MIT 或 Apache-2.0；资产 CC0（第三方见 attributions）

## 开源状态

**已开源** — 完整 Bench + CI（Rust/Python/Docker/Docs）。论文方法是该 Bench 上的世界模型适应流程，不是单独算法包。
