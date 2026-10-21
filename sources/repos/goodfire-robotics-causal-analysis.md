# goodfire-ai/robotics_causal_analysis

- **类型：** 官方代码仓库
- **仓库：** <https://github.com/goodfire-ai/robotics_causal_analysis>
- **论文：** [arXiv:2610.20750](https://arxiv.org/abs/2610.20750)
- **许可：** MIT；仓库中派生的 `data/robot/franka_droid_kinematics.xml` 标注 Apache-2.0
- **源码状态：** 已公开（核验日期：2026-10-17）
- **知识节点：** [VLA 因果分析论文](../../wiki/entities/paper-robot-foundation-model-causal-analysis.md)

## 仓库要点

仓库归档 activation patching 代码与相关数据，目录包括 `vla_patching/`、`data/`、`docs/` 和 `figures/`。README 摘要给出核心复现线索：在供体与接收任务之间交换视觉语言骨干层到动作专家的激活，观察动作变化；将 MolmoBot 的动作条件化连接收窄到 36 条中的 3 条后，计算量降低约 33%，仿真成功率接近原模型。

## 关联资料

- [论文摘录](../papers/robot-foundation-model-causal-analysis_arxiv_2610_20750.md)
