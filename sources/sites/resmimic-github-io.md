# resmimic.github.io（ResMimic 项目页）

- **标题：** ResMimic — From General Motion Tracking to Humanoid Whole-Body Loco-Manipulation via Residual Learning
- **类型：** site / project-page
- **URL：** <https://resmimic.github.io/>
- **配套论文：** [ResMimic（arXiv:2510.05070）](https://arxiv.org/abs/2510.05070) — 归档见 [`sources/papers/resmimic_arxiv_2510_05070.md`](../papers/resmimic_arxiv_2510_05070.md)
- **代码：** <https://github.com/amazon-far/ResMimic> — 归档见 [`sources/repos/resmimic.md`](../repos/resmimic.md)
- **入库日期：** 2026-06-09

## 一句话摘要

Amazon FAR 等团队的 **ResMimic** 官方站点：展示 **GMT 预训练 + 残差后训练** 在 Unitree G1 上完成 **跪姿抬箱、背箱、蹲起托举、搬椅** 等全身 loco-manipulation；强调 **5.5 kg 级载荷**、**随机物体初姿**、**连续 MoCap 驱动** 与 **外力扰动恢复**，并提供与从头训练/微调基线的并排对比与 **关节残差动作可视化**。

## 公开信息要点（截至入库日）

- **机构：** Amazon FAR；USC；Stanford；UC Berkeley；CMU（§ 实习；† FAR Co-Lead）。
- **方法卖点（页首）：** 无需任务定制设计即可 **全身 loco-manipulation**；**跨姿态泛化**；**反应式行为**；载荷至 **5.5 kg**。
- **演示板块：**
  - **Expressive Whole-Body Loco-Manipulation** — Carry Box onto Back；Kneel on One Knee & Lift Box
  - **Heavy Object with Whole-Body Contact** — Squat & Lift（4.5 kg）；Lift Chair（4.5–5.5 kg，多实例）
  - **General Object Interaction** — Sit on Chair
  - **Continuous Execution with MoCap Input** — 随机初姿抬箱；自主连续抬箱
  - **Reactive Behavior** — 外力扰动
  - **Comparison with Baselines** — vs Base / Train from Scratch / Finetune
  - **Policy Visualization** — 预训练 vs 残差 **关节动作空间 delta**（腕部残差更显著）
  - **Ablation Results in Simulation**
- **BibTeX：** 页面提供 `@misc{zhao2025resmimic,...}` 引用块。

## 为何值得保留

- **非 PDF 证据：** 真机载荷标注、连续执行与扰动视频比表格更直观呈现 **残差修正** 与 **全身接触** 收益。
- **与 arXiv 三角互证：** 项目页强调 5.5 kg 与「无任务定制」，便于核对论文实验设定。
- **下游锚点：** 与 [holosoma](../repos/holosoma.md)、[OmniRetarget](../papers/omniretarget_arxiv_2509_26633.md) 同属 Amazon FAR 人形 loco-manipulation 技术线。

## 关联资料

- 论文归档：[`sources/papers/resmimic_arxiv_2510_05070.md`](../papers/resmimic_arxiv_2510_05070.md)
- 代码仓库：[`sources/repos/resmimic.md`](../repos/resmimic.md)
