# AME-2 官方最小代码仓库（ACP 复用的前作）

- **标题：** Minimal AME-2 extension of IsaacLab
- **类型：** repo（官方 AME-2 实现）
- **组织：** Legged Robotics / ETH Zurich RSL
- **代码：** <https://github.com/leggedrobotics/ame2_minimal>
- **项目页：** <https://sites.google.com/leggedrobotics.com/ame-2>
- **论文：** [AME-2（arXiv:2601.08485）](../papers/humanoid_pnb_ame-2-agile-and-generalized-legged-locomotion-vi.md)
- **关联项目：** ACP 说明复用 AME-2 神经地图；见 [ACP 项目页归档](../sites/acp-attention-based-contacts.md)
- **许可：** GPL-3.0（仓库 README）
- **核查日期：** 2026-10-23
- **一句话说明：** AME-2 的官方 IsaacLab 最小扩展；它是 ACP 所引用前作的代码入口，不能视为 ACP 全套接触规划与跟踪实现。

## 开源边界

仓库 README 将自身描述为 AME-2 官方重实现。README 同时明确指出，Mid-360 神经地图模型不包含在仓库中；G1 student training 依赖该模型，亦暂未提供。神经地图训练代码和 checkpoint 列为后续发布计划。因此该仓库可以用于检查 AME-2 的实现边界，但不能单独复现 ACP 项目页所述的神经地图感知、接触规划器或全身跟踪系统。

## 对 wiki 的映射

- [ACP：注意力式接触规划人形移动](../../wiki/entities/paper-acp-humanoid-mobility.md)
- [AME-2：注意力式神经地图编码](../../wiki/entities/paper-notebook-ame-2-agile-and-generalized-legged-locomotion-vi.md)
