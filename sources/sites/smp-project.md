# SMP 项目页（yxmu.foo/smp-page）

- **标题：** SMP — Reusable Score-Matching Motion Priors
- **类型：** site / project-page
- **URL：** <https://yxmu.foo/smp-page/>
- **配套论文：** [SMP（arXiv:2512.03028）](https://arxiv.org/abs/2512.03028) — 归档见 [`sources/papers/smp.md`](../papers/smp.md)
- **官方代码：** <https://github.com/xbpeng/MimicKit>（`docs/README_SMP.md`）— 归档见 [`sources/repos/mimickit.md`](../repos/mimickit.md)
- **G1 复现代码：** <https://github.com/senlanke/mimic> — 归档见 [`sources/repos/senlanke_mimic.md`](../repos/senlanke_mimic.md)（2026-08-29 起同仓还挂 CMoE 移植与未完成 AME）
- **入库日期：** 2026-08-25
- **复核日期：** 2026-08-29

## 一句话摘要

SFU / NVIDIA 等团队的 **Score-Matching Motion Priors (SMP)** 官方项目页：展示冻结扩散模型 + SDS 作为可复用运动先验，在仿真人形多任务（转向、落点、躲避球、搬运、楼梯）与 **Unitree G1 真机** 上的结果；强调 **Modular / Reusable / Composable** 三卖点。

## 公开信息要点（截至入库日）

- **机构：** Simon Fraser University、Sony Interactive Entertainment、Stanford University、Snap Inc.、National Research Council Canada、NVIDIA（共同一作 Yuxuan Mu、Ziyu Zhang、Yi Shi、Dun Yang）。
- **页首卖点：**
  - **Reusable** — 单一冻结扩散模型跨 locomotion / steering / dodgeball / zombie-walk 等多任务作奖励
  - **Modular** — 先验与策略解耦训练，下游 RL **无需访问原始 MoCap**
  - **Composable** — 100STYLE 条件先验经 classifier-free guidance 与 per-body-part mixing 组合新风格
- **方法板块：** DDPM 预训练 → 冻结 ε-预测器 → SDS 奖励；**Ensemble Score-Matching (ESM)**、**Adaptive Normalization**、**Generative State Initialization (GSI)**
- **任务演示：** Steering、Target Location、Dodgeball、Object Carry、Stair Traversal、3 秒数据技能涌现、G1 真机
- **视频：** 完整 walkthrough（YouTube）
- **论文 PDF：** 链向 arXiv
- **BibTeX：** `@article{mu2026smp,...}`（ACM TOG / SIGGRAPH 2026）

## 源码开放核查（步骤 2.5）

| 入口 | 状态 | 说明 |
|------|------|------|
| 项目页 | **已开源（官方）** | 论文方法实现见 **MimicKit** `docs/README_SMP.md` |
| 项目页 | **已开源（G1 复现）** | 用户指定仓库 **senlanke/mimic**（mjlab；SMP 完整 + CMoE 移植完成 + AME 未验证；与 SUZ-tsinghua/smp 同系） |
| 预训练权重 | **部分** | MimicKit 需自训 prior；senlanke/mimic 内置三套 G1 SMP prior；CMoE/AME 无可靠预置权重 |

## 关联资料

- 论文摘录：[`sources/papers/smp.md`](../papers/smp.md)
- 官方实现：[`sources/repos/mimickit.md`](../repos/mimickit.md)
- G1 复现：[`sources/repos/senlanke_mimic.md`](../repos/senlanke_mimic.md)、[`sources/repos/smp_suz_tsinghua.md`](../repos/smp_suz_tsinghua.md)
