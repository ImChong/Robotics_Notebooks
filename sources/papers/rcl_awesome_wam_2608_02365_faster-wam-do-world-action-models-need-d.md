# Faster-WAM: Do World Action Models Need Deep Action Modules?

> 来源归档（RCL Awesome World-Action Models · ``docs/PAPERS.md`` / ``data/papers.json``）

- **列表：** [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- **PAPERS.md：** <https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md>
- **分组：** WAMs
- **编号：** 351/564
- **入库日期：** 2026-09-25
- **arXiv：** 2608.02365
- **出处：** 见清单
- **提交日：** 2026-08-03
- **论文：** <https://arxiv.org/abs/2608.02365>
- **代码：** 未在清单中标注

- **子类 / 象限：** 高效推理与实时控制 · 四象限外
- **Contribution（清单）：** To address this limitation, we introduce Dock of Transformer (DoT), a video-centric design principle that treats a pretrained video Transformer as a representation hub and connects lightweight output-heads through docking interfaces. Without additional embodied pretraining, Faster-WAM achieves competitive performanc...
- **沉淀到 wiki：** [`wiki/entities/paper-rcl-2608-02365-faster-wam-do-world-action-models-need-deep-acti.md`](../../wiki/entities/paper-rcl-2608-02365-faster-wam-do-world-action-models-need-deep-acti.md)

---

## 开源边界（步骤 2.5）

| 已发布 | 备注 |
|--------|------|
| 清单条目元数据 | 本 source 为策展摘录，非全文转存 |
| 代码/权重 | 以项目页 / GitHub 实际链接为准；清单标注见上 |

## 原文 v2 补充核对（2026-11-14）

- **原文：** <https://arxiv.org/html/2608.02365>（v2，2026-10-06）；项目 <https://teamfasterwam.github.io/Faster-WAM/>
- **核对范围：** arXiv HTML v2 §1–5、附录结构与主要训练/评测设置，表 1–4；对照官方代码 README、scripts/train.sh、scripts/train.py、scripts/eval.sh。
- **通读：** 已完成
- **官方代码：** <https://github.com/TeamFasterwam/FasterWAM> 已开放完整训练/基准评测入口，LICENSE 为 PolyForm Noncommercial 1.0.0；主页权重 Coming soon。
- **关键机制：** DoT / Lite KV-Fusion 融合视频多层 KV，world-model-only conditioning，retracted 1D-RoPE；单层 action expert。
- **论文实验：** LIBERO-Plus 78.3%（Fast-WAM 51.5%）；Franka 4 任务平均成功 81.3%（Fast-WAM 80.0%），本地部署 chunk 推理 129 vs 502 ms；RoboTwin 2.0 平均 90.5%。
- **对 wiki 的映射：** [升级后的 Faster-WAM 实体](../../wiki/entities/paper-rcl-2608-02365-faster-wam-do-world-action-models-need-deep-acti.md)；原索引归档保留以便回溯。

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-rcl-2608-02365-faster-wam-do-world-action-models-need-deep-acti.md`](../../wiki/entities/paper-rcl-2608-02365-faster-wam-do-world-action-models-need-deep-acti.md)
- 列表实体：[`wiki/entities/paper-rcl-wam-robot-learning-control-survey.md`](../../wiki/entities/paper-rcl-wam-robot-learning-control-survey.md)
