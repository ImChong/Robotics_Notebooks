# T-Rex（ZhuoyangLiu2005/T-Rex）

> 来源归档

- **标题：** T-Rex — Tactile-Reactive Dexterous Manipulation
- **类型：** repo
- **组织：** UC Berkeley / NVIDIA 等（见论文作者）
- **代码：** <https://github.com/ZhuoyangLiu2005/T-Rex>
- **项目页：** <https://tactile-reactive-dexterous.github.io/>
- **论文：** <https://arxiv.org/abs/2606.17055>
- **入库日期：** 2026-07-01
- **一句话说明：** T-Rex 官方实现入口：触觉反应式灵巧操作策略、**T-Rex Dataset** 相关工具与 **12 任务** 评测/部署代码（以仓库 README 为准）。
- **沉淀到 wiki：** [T-Rex（论文实体）](../../wiki/entities/paper-trex-tactile-reactive-dexterous-manipulation.md)

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [VLA](../../wiki/methods/vla.md) | **变频率 MoT + flow matching** 在 VLA 骨干上叠加 **高频触觉专家** |
| [EgoScale](../../wiki/methods/egoscale.md) | 共享 **人 egocentric 预训练 + 机端 mid-training** 叙事；T-Rex 把 mid-training 换成 **触觉同步 play** |
| [接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md) | **12 任务** 基准覆盖力控、形变与双手协调 |
| [视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md) | **时序力 VQ-VAE + 形变图** 双通路触觉 token 化 |

## 为何值得保留

- 论文同时开源 **数据集与模型**；仓库是复现 **异步级联 flow matching** 与 **触觉 mid-training** 的直接入口。
- 与 [Humanoid_Robot_Learning_Paper_Notebooks](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks) 单篇深读互补：本库负责 **触觉反应式 VLA × 双手灵巧** 跨主题索引。
