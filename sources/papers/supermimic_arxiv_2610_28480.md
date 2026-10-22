# SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model（arXiv:2610.28480）

> 来源归档（ingest）

- **标题：** SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.28480>（v1，2026-10-20，cs.RO）
- **会议：** Conference on Robot Learning（CoRL）2026，Oral
- **作者：** Qiayuan Liao、Yiyang Shao、Junfeng Long、Kevin Zakka、Roberto Horowitz、Pieter Abbeel、Koushil Sreenath
- **机构：** 加州大学伯克利分校
- **项目页：** <https://supermimic.github.io/>（[sources/sites/supermimic-project.md](../sites/supermimic-project.md)）
- **官方代码：** 项目页未列论文作者维护的代码、数据或权重链接（2026-10-22 核查）；社区独立复现见 [Uni-SuperMimic](../repos/supermimic-unilab.md) 和 [supermimic_vibed](../repos/supermimic-vibed.md)，二者不属于官方实现
- **入库日期：** 2026-10-22
- **一句话说明：** 从 MuJoCo Warp 记录的状态转移中学习可微世界模型，并与人形动作跟踪策略联合训练，以短时想象 rollout 和解析策略梯度降低大规模全身动作跟踪的计算成本。

## 核心摘录（论文摘要与官方项目页）

- **方法：** 世界模型学习标准刚体仿真器中的共享动力学；世界模型与策略在同一训练循环中迭代更新。策略更新时对短时想象 rollout 做反向传播，不需要对仿真器求导。
- **计算：** 官方项目页称单张 NVIDIA H100 用 3 小时超过 SONIC 已发布 checkpoint 的归一化回报；SONIC 对照为 128 张 L40、7 天。论文摘要报告 GPU-hours 降低超过 1000 倍。双卡训练 9.8 小时后，基准动作失败数为 0。
- **覆盖：** 在 129,514 个 BONES-SEED 训练动作上，SONIC checkpoint 失败 179 个；SuperMimic 单卡 3 小时失败 25 个、双卡 9.8 小时失败 0 个。该评测反映训练动作覆盖率，不代表未见动作泛化。
- **真机：** 论文与项目页报告同一策略迁移到 Unitree G1，展示多种动态动作；项目页说明视频没有按动作单独调参。

## 对 wiki 的映射

- [paper-supermimic](../../wiki/entities/paper-supermimic.md) — 论文、项目页及独立复现统一映射到同一研究实体
- [SONIC](../../wiki/entities/nvidia-sonic.md) — 相关的大规模人形动作跟踪基线
