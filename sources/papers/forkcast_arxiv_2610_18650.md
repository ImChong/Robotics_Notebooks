# ForkCast: Predictive Recovery for Vision-Language-Action Policies via Counterfactual Futures

> 来源归档（paper）

- **标题：** ForkCast: Predictive Recovery for Vision-Language-Action Policies via Counterfactual Futures
- **中文标题：** ForkCast：通过反事实未来预测实现视觉语言动作策略的预测式恢复
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.18650>
- **入库日期：** 2026-10-21
- **线索来源：** [公众号综述（33 篇 Manipulation）](../raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md)（具身运控lab，2026-10-20）
- **一句话说明：** 比较 Commit / Reobserve / Replan 三种候选的反事实未来，决定冻结 VLA 是否及如何干预。
- **沉淀到 wiki：** [`wiki/entities/paper-forkcast-counterfactual-recovery.md`](../../wiki/entities/paper-forkcast-counterfactual-recovery.md)

## 摘要要点（arXiv 摘要页）

- 物体滑落与遮挡会打断 VLA 执行；现有恢复多在训练时学习或失败后触发。
- 方法：候选条件世界模型预测各候选的潜在结果，共享任务进度头打分。

## 开源状态

- **未核查代码**：arXiv 摘要页未列代码链接。
