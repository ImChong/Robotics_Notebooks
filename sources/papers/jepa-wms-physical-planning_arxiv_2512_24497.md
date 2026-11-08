# What Drives Success in Physical Planning with Joint-Embedding Predictive World Models? / JEPA 世界模型物理规划成功因素

- **类型：** 论文原文归档（非转载）
- **主论文：** <https://arxiv.org/abs/2512.24497>
- **HTML 原文：** <https://arxiv.org/html/2512.24497>
- **官方代码：** <https://github.com/facebookresearch/jepa-wms>
- **入库日期：** 2026-11-08
- **核对范围：** 阅读 HTML §3–§5 的方法、对比与部分结果以及官方 README；完整附录、源码调用链尚未逐项核查
- **通读：** 未完成

## 核心方法

冻结视觉编码器，与可选本体状态编码器、动作编码器和 Transformer 预测器组成 JEPA-WM；预测未来联合嵌入，采用 MSE 与多步 Rollout 训练，测试时由 CEM 等规划器搜索动作。

## 实验与边界

官方仓库区分 DROID/RoboCasa 上使用 DINOv3 ViT-L/16 的模型，与 Metaworld/Push-T/PointMaze/Wall 使用 DINOv2 的模型；论文主要讨论规划优化器、上下文、模型大小及 proprioception 的设计取舍。

不是所有 JEPA-WM 模型都使用 DINOv3；测试任务和规划成本具有环境依赖，不能把论文整体称为 DINOv3 专用模型。

## 对 wiki 的映射

- [paper-jepa-wms-physical-planning.md](../../wiki/entities/paper-jepa-wms-physical-planning.md)
