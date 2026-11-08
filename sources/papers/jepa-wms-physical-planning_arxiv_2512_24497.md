# What Drives Success in Physical Planning with Joint-Embedding Predictive World Models? / JEPA 世界模型物理规划成功因素

- **类型：** 论文原文归档（非转载）
- **主论文：** <https://arxiv.org/abs/2512.24497>
- **HTML 原文：** <https://arxiv.org/html/2512.24497>
- **官方代码：** <https://github.com/facebookresearch/jepa-wms>
- **入库日期：** 2026-11-08
- **核对范围：** 全文背景/设计空间/全部主实验与结论（§1–6），附录 A–G 的训练细节、误差传播、规划器、机器人数据和失败案例；核验 app/main.py 与官方 README。
- **通读：** 已完成

## 核心方法

冻结视觉编码器，与可选本体状态编码器、动作编码器和 Transformer 预测器组成 JEPA-WM；预测未来联合嵌入，采用 MSE 与多步 Rollout 训练，测试时由 CEM 等规划器搜索动作。

## 实验与边界

官方仓库区分 DROID/RoboCasa 上使用 DINOv3 ViT-L/16 的模型，与 Metaworld/Push-T/PointMaze/Wall 使用 DINOv2 的模型；论文主要讨论规划优化器、上下文、模型大小及 proprioception 的设计取舍。

不是所有 JEPA-WM 模型都使用 DINOv3；测试任务和规划成本具有环境依赖，不能把论文整体称为 DINOv3 专用模型。

## 对 wiki 的映射

- [paper-jepa-wms-physical-planning.md](../../wiki/entities/paper-jepa-wms-physical-planning.md)

## 2026-11-08 一手资料精读笔记

- **模型架构：** 冻结视觉编码器 + 动作编码器 + 可选本体状态输入 + causal JEPA predictor，以未来潜空间目标误差进行 CEM/NG/梯度规划。**DROID/RoboCasa 使用 DINOv3 ViT-L/16；其他常用仿真任务的最佳模型使用 DINOv2**。
- **实验口径：** CEM+L2 通常表现最稳；Metaworld 梯度方法有竞争力；训练多步 loss 的最优步数随任务变化。论文 DROID/RoboCasa 上仍有 Pick 与末端标定失误，规划成功率不能用单步预测误差代替。
- **局限风险：** 真实机器人域偏移、夹爪位置、标定及动作尺度影响目标达成；CEM 候选数量和迭代开销显著；不可把所有 JEPA-WM checkpoint 混成一个 DINOv3 模型。
