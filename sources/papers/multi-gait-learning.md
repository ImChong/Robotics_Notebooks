# multi-gait-learning

> 来源归档（ingest）

- **标题：** Multi-Gait Learning for Humanoid Robots Using Reinforcement Learning with Selective Adversarial Motion Priority
- **类型：** paper
- **来源：** arXiv:2604.19102 (Hypothetical/Found)
- **入库日期：** 2026-05-03
- **最后更新：** 2026-05-03
- **一句话说明：** A paper about learning multiple gaits using RL and Selective AMP where AMP is applied to periodic gaits but omitted for highly dynamic ones.

## 核心论文摘录（MVP）

### 1) Multi-Gait Learning for Humanoid Robots Using Reinforcement Learning with Selective Adversarial Motion Priority
- **链接：** <https://arxiv.org/abs/2604.19102>
- **核心贡献：** 提出了一种 multi-gait learning 框架，使人形机器人能够掌握五种不同的步态（walking, goose-stepping, running, stair climbing, and jumping）。核心贡献是选择性对抗运动先验（Selective Adversarial Motion Priority / AMP）策略：对周期性、注重稳定性的步态（如 walking, goose-stepping, stair climbing）应用 AMP 以加速收敛和抑制不规律动作；对于高动态步态（如 running, jumping），则故意省略 AMP，因为其正则化会过度约束运动。该策略在统一的策略结构、动作空间和奖励公式下训练，并通过 zero-shot sim-to-real transfer 部署到真实机器人上。
- **对 wiki 的映射：**
  - [AMP Reward](../../wiki/methods/amp-reward.md) -> 更新，说明 selective AMP
  - [Locomotion](../../wiki/tasks/locomotion.md) -> 更新多步态学习 (multi-gait learning)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [ ] 关联 wiki 页面的参考来源段落已添加 ingest 链接
