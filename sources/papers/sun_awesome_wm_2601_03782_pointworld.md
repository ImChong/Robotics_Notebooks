# PointWorld

> 来源归档（sun254667 Awesome 策展索引级）

- **列表：** [Awesome World Models](https://github.com/sun254667/awesome-world-models)
- **分组：** 812 Manipulation
- **编号：** 324/571
- **入库日期：** 2026-08-10
- **arXiv：** 2601.03782
- **出处：** arXiv 2026
- **论文：** <https://arxiv.org/abs/2601.03782>
- **代码：** <https://github.com/NVlabs/PointWorld>（2026-09-02 核查，清单原未标注）
- **项目页：** <https://point-world.github.io>
- **Highlights（清单）：** NVIDIA's large-scale pre-trained 3D world model unifying state and action as 3D point flows.
- **沉淀到 wiki：** [`wiki/entities/paper-sa-2601-03782-pointworld.md`](../../wiki/entities/paper-sa-2601-03782-pointworld.md)

---

## 开源边界（步骤 2.5）

| 已发布 | 备注 |
|--------|------|
| 清单条目元数据 | 本 source 为策展摘录，非全文转存 |
| 代码/权重 | **已开源** [NVlabs/PointWorld](https://github.com/NVlabs/PointWorld)（2026-09-02 核查） |

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-sa-2601-03782-pointworld.md`](../../wiki/entities/paper-sa-2601-03782-pointworld.md)
- 列表实体：[`wiki/entities/awesome-world-models.md`](../../wiki/entities/awesome-world-models.md)

## 2026-11-08 一手资料核对

- **原文：** <https://arxiv.org/html/2601.03782>
- **核对范围：** 全文 1–6 章（3D 预测、训练数据、实验与真实机器人规划）与附录 A.1–A.5 中关键技术、局限、点特征和实验协议，交叉阅读 train.py / eval.py 及 README。
- **通读：** 已完成
- **官方代码核验：** <https://github.com/NVlabs/PointWorld>；以仓库 README / 可见文件为准，不等同于复现成功。

## 2026-11-08 一手资料精读笔记

- **模型架构：** PointWorld 把 DINOv3 ViT-L/16 多层二维特征经 camera intrinsics/extrinsics 投影到 3D 场景点，将机器人表面点未来流作为动作，学习 action-conditioned 3D 点流预测；实际规划用 **MPPI** 优化末端 SE(3) 目标轨迹。
- **实验口径：** 在 DROID 与 BEHAVIOR 大规模混合数据上预训练；作者使用移动 Franka 与单台 RealSense D435 测试刚体推、柔性/铰接物体及工具交互，预测 10-step 点流。约 0.1s 模型预测不等于全 MPPI 优化时间。
- **局限风险：** 论文附录：只见初始单帧，默认初始静态；不预测光度变化；以刚体 URDF 前向运动学处理机器人，细小物体受深度/外参误差影响；不代表学到了因果物理机制。
