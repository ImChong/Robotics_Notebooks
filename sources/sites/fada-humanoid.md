# FADA 项目主页与公开资源（CoRL 2026）

> 项目一手资源归档；核查日期：2026-11-23。

- **项目：** FADA: Few-Shot Domain Adaptation via Dynamics Alignment for Humanoid Control
- **主页：** <https://lecar-lab.github.io/FADA-humanoid/>
- **论文：** <https://arxiv.org/abs/2606.28476v2>（[论文归档](../papers/fada_arxiv_2606_28476.md)）
- **代码：** <https://github.com/LeCAR-Lab/FADA>（[源码核查](../repos/fada-humanoid.md)）
- **Oracle 权重：** <https://huggingface.co/AngchenXie/fada-checkpoints>
- **研究机构：** 卡内基梅隆大学（CMU）LeCAR Lab
- **发表状态：** 官方代码仓库标注 CoRL 2026；arXiv v2 日期为 2026-10-09。
- **通读：** 已完成
- **核对范围：** 核对官方项目主页及其项目仓库 README 的论文、视频、源码与公开状态；源码入口另核对官方仓库的训练、评估、微调与推理文件。

## 资源开放结论（2026-11-23）

官方仓库 README 的 Status 清单显示：oracle PPO + DAgger 训练、IDM LoRA 微调、MuJoCo sim2sim、真机部署与数据采集代码以及 G1/T1 oracle checkpoint **已开放**；**whole-body tracking 代码仍未勾选**。这意味着有可执行的 locomotion 复现链路，但不能把论文所有全身跟踪实验视为已经完整开源。

## 对 wiki 的映射

- [FADA 论文独立实体](../../wiki/entities/paper-fada-humanoid.md) — Planner–IDM、少样本适应与真实运行入口
- [Sim2Real](../../wiki/concepts/sim2real.md) — 动力学失配与执行层适应
