# RPL-CS-UCL/SDS

> 来源归档

- **标题：** SDS — See it, Do it, Sorted（官方实现）
- **类型：** repo
- **组织：** Robot Perception Lab, UCL
- **代码：** <https://github.com/RPL-CS-UCL/SDS>
- **论文：** <https://arxiv.org/abs/2410.11571>（摘录见 [`sources/papers/sds_quadruped_arxiv_2410_11571.md`](../papers/sds_quadruped_arxiv_2410_11571.md)）
- **项目页：** <https://rpl-cs-ucl.github.io/SDSweb/>
- **入库日期：** 2026-05-17
- **一句话说明：** **四足** 单视频 → VLM 生成奖励 → IsaacGym PPO + 闭环进化的参考实现入口；与 **E-SDS（arXiv:2512.16446）** 同属 UCL 路线，后者扩展到 **人形 + 地形统计条件的感知奖励**。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [E-SDS 论文实体](../../wiki/entities/paper-e-sds-environment-aware-humanoid-locomotion-rl.md) | 方法谱系上游：**共享 SUS / 网格视频编码 / 闭环奖励进化** 叙事 |
| [Locomotion 任务页](../../wiki/tasks/locomotion.md) | 腿足运动学习中 **奖励工程自动化** 的代码侧参照 |

## 对 wiki 的映射

- 沉淀交叉引用于 **[`wiki/entities/paper-e-sds-environment-aware-humanoid-locomotion-rl.md`](../../wiki/entities/paper-e-sds-environment-aware-humanoid-locomotion-rl.md)**；SDS 论文原文见 [`sources/papers/sds_quadruped_arxiv_2410_11571.md`](../papers/sds_quadruped_arxiv_2410_11571.md)。
