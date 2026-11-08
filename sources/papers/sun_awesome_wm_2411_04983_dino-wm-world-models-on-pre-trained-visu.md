# DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning

> 来源归档（sun254667 Awesome 策展索引级）

- **列表：** [Awesome World Models](https://github.com/sun254667/awesome-world-models)
- **分组：** 44 Latent Space World Models
- **编号：** 110/571
- **入库日期：** 2026-08-10
- **arXiv：** 2411.04983
- **出处：** arXiv 2024
- **论文：** <https://arxiv.org/abs/2411.04983>
- **代码：** 未在清单中标注
- **项目页：** <https://dino-wm.github.io/>
- **Highlights（清单）：** World models on pre-trained visual features enable zero-shot planning.
- **沉淀到 wiki：** [`wiki/entities/paper-sa-2411-04983-dino-wm-world-models-on-pre-trained-visual-featu.md`](../../wiki/entities/paper-sa-2411-04983-dino-wm-world-models-on-pre-trained-visual-featu.md)

---

## 开源边界（步骤 2.5）

| 已发布 | 备注 |
|--------|------|
| 清单条目元数据 | 本 source 为策展摘录，非全文转存 |
| 代码/权重 | 以项目页 / GitHub 实际链接为准；清单标注见上 |

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-sa-2411-04983-dino-wm-world-models-on-pre-trained-visual-featu.md`](../../wiki/entities/paper-sa-2411-04983-dino-wm-world-models-on-pre-trained-visual-featu.md)
- 列表实体：[`wiki/entities/awesome-world-models.md`](../../wiki/entities/awesome-world-models.md)

## 2026-11-08 一手资料核对

- **原文：** <https://arxiv.org/html/2411.04983>
- **核对范围：** 完整阅读正文 1–5、附录 A 的数据/模型/规划与消融，核查官方 train.py 和 plan.py。
- **通读：** 已完成
- **官方代码核验：** <https://github.com/gaoyuezhou/dino_wm>；以仓库 README / 可见文件为准，不等同于复现成功。

- **版本辨析：** 本文使用 DINOv2，不是 DINOv3；作为后续 DINOv3 世界模型的历史基线收录。

## 2026-11-08 论文逐段核验要点

- **方法：** 使用 **DINOv2**（不是 DINOv3）的冻结 patch 表征，学习 action-conditioned causal Transformer 预测器，使用视觉目标嵌入差异指导 CEM 零样本规划。
- **实验：** 在 PointMaze、Wall、PushT、Reach、Rope、Granular 任务测试；CEM 动作序列搜索开销显著，不等于单次 WM forward 时延。
- **局限：** 需要覆盖足够的离线状态-动作轨迹，训练仍依赖真实动作标签；原项目只能作为 DINOv3 世界模型的历史基线。
