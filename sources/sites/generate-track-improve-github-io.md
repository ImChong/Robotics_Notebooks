# generate-track-improve.github.io（Generate, Track, Improve 项目页）

> 来源归档（ingest）

- **标题：** Generate, Track, Improve: Perceptive Multi-Skill Humanoid Locomotion with RL-Fine-Tuned Motion Generators
- **类型：** site / project-page
- **URL：** <https://zolkin1.github.io/generate-track-improve/>
- **入库日期：** 2026-09-29
- **配套论文：** [Generate, Track, Improve（arXiv:2609.31577）](https://arxiv.org/abs/2609.31577) — 归档见 [`sources/papers/generate_track_improve_arxiv_2609_31577.md`](../papers/generate_track_improve_arxiv_2609_31577.md)
- **机构：** 加州理工学院（Caltech）控制与动力系统系（CDS）— Zachary Olkin, William D. Compton, Aaron D. Ames（AMBER Lab）

## 一句话摘要

Caltech AMBER 官方站点：展示 **双层感知 locomotion**（flow matching 全身轨迹生成 + CLF-RL 跟踪）、**AWR 离线 RL 微调生成器** 的真机/仿真视频，以及双相机、速度自调节等消融；页内 **Paper / arXiv / Video** 齐全，**尚无 Code 按钮**。

## 公开信息要点（截至入库日）

- **PDF 镜像：** `paper/generate-track-improve.pdf`
- **补充视频：** [YouTube](https://youtu.be/U81SjJIKUFY)
- **导航分区：** Abstract · Method · Hardware · Skills in simulation · RL fine-tuning · Ablations · Citation
- **硬件：** Unitree G1；ZED X（前向）+ ZED X Mini（下视）；生成器/跟踪器在 Jetson Thor，深度在 Jetson Orin
- **真机技能：** 15 级楼梯、箱跳上/下、户外走跑、建筑入口与停车结构等
- **HTML 注释（源码）：** `<!-- TODO(code): add a Code button here when the code is released. -->`

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| GitHub / HF / Zenodo | **无** — 页眉仅 Paper、arXiv、Video |
| 开放程度 | **待发布** — 站点明确预留 Code 按钮位，截至 **2026-10-01** 无官方可运行仓库（复核查 HTML `<!-- TODO(code): ... -->` 仍在；页眉仍仅 Paper / arXiv / Video） |
| 数据集 | 未单独发布；依赖 BONES-SEED + 自研优化 clip 库（论文描述） |
| 部署栈 | 真机 demo 完整；复现需自建 Isaac Lab + flow matching + AWR 环 |

## 为何值得保留

- **非 PDF 证据：** 户外楼梯/箱跳/跑等多技能单策略对视频，便于与 [RPL](../../wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md)、[ETH 扩散+跟踪](../../wiki/entities/paper-hrl-stack-27-learning_whole_body_humanoid_locomot.md) 对照。
- **开源状态锚点：** HTML TODO 便于 lint 跟进代码发布。

## 对 wiki 的映射

- [`wiki/entities/paper-generate-track-improve.md`](../../wiki/entities/paper-generate-track-improve.md)
- [`wiki/methods/chasing-autonomy-pipeline.md`](../../wiki/methods/chasing-autonomy-pipeline.md) — 同组 Caltech 动态优化 + CLF-RL 谱系
