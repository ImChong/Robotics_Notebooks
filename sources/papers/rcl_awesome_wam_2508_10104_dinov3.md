# DINOv3

> 来源归档（RCL Awesome World-Action Models · ``docs/PAPERS.md`` / ``data/papers.json``）

- **列表：** [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- **PAPERS.md：** <https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md>
- **分组：** Components of WAMs
- **编号：** 070/564
- **入库日期：** 2026-09-25
- **arXiv：** 2508.10104
- **出处：** Transactions on Machine Learning Research
- **提交日：** —
- **论文：** <https://openreview.net/forum?id=2NlGyqNjns>
- **代码：** <https://github.com/facebookresearch/dinov3>

- **子类 / 象限：** 视觉编码器与表征 · 不适用
- **Contribution（清单）：** RCL Awesome WAM 清单收录（Components of WAMs）；细节以原文 PDF / 项目页为准。
- **沉淀到 wiki：** [`wiki/entities/paper-rcl-2508-10104-dinov3.md`](../../wiki/entities/paper-rcl-2508-10104-dinov3.md)

---

## 开源边界（步骤 2.5）

| 已发布 | 备注 |
|--------|------|
| 清单条目元数据 | 本 source 为策展摘录，非全文转存 |
| 代码/权重 | 以项目页 / GitHub 实际链接为准；清单标注见上 |

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-rcl-2508-10104-dinov3.md`](../../wiki/entities/paper-rcl-2508-10104-dinov3.md)
- 列表实体：[`wiki/entities/paper-rcl-wam-robot-learning-control-survey.md`](../../wiki/entities/paper-rcl-wam-robot-learning-control-survey.md)

## 2026-11-08 一手资料核对

- **原文：** <https://arxiv.org/html/2508.10104>
- **核对范围：** 原文第 1–10 章（训练方法/Gram Anchoring/蒸馏/稠密评测/遥感与局限），并核对附录 A、C、D 与官方 hubconf.py、模型 README。
- **通读：** 已完成
- **官方代码核验：** <https://github.com/facebookresearch/dinov3>；以仓库 README / 可见文件为准，不等同于复现成功。

## 2026-11-08 一手资料精读笔记

- **模型架构：** **DINOv3 是视觉编码器而非世界模型**。训练以 DINO/iBOT/KoLeo 与 Gram anchoring 保持稠密 patch 特征，7B ViT 教师经蒸馏提供 ViT-S/B/L/H+；下游世界模型各自决定冻结层和预测目标。
- **实验口径：** 论文 ViT-7B/16 的 DAVIS-L J&F=83.3，对比 DINOv2-g/14 的 76.6；ADE20k / 深度等视觉评测不能解释为动作规划成功率。
- **局限风险：** 权重独立许可、输入分辨率与 patch token 影响显存；论文本身没有 action-conditioned transition、动力学或 MPC。
