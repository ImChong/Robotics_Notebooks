# SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model（arXiv:2610.28480）

> 来源归档（ingest）

- **标题：** SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.28480>（v1，2026-10-20，cs.RO）
- **会议：** Conference on Robot Learning（CoRL）2026，Oral
- **作者：** Qiayuan Liao、Yiyang Shao、Junfeng Long、Kevin Zakka、Roberto Horowitz、Pieter Abbeel、Koushil Sreenath
- **机构：** 加州大学伯克利分校
- **项目页：** <https://supermimic.github.io/>（[sources/sites/supermimic-project.md](../sites/supermimic-project.md)）
- **代码状态（2026-11-11 更新）：** [HybridRobotics/supermimic](../repos/supermimic-hybridrobotics.md) 已公开 mjlab 训练、回放实现；论文官网目前未直接链接该仓库，论文全部资产/实验配置是否对应尚需另行核对。旧版归档于 2026-10-22 尚未查见该入口。社区独立复现见 [Uni-SuperMimic](../repos/supermimic-unilab.md) 和 [supermimic_vibed](../repos/supermimic-vibed.md)
- **入库日期：** 2026-10-22
- **再核日期：** 2026-11-11
- **核对范围：** arXiv v1 PDF 正文与附录（共 19 页），重点核对方法图、训练伪代码、实验表、部署评估边界和局限；并对照公开仓库 README 与关键训练/回放源码。未经独立实验复现。
- **通读：** 已完成
- **一句话说明：** 从 MuJoCo Warp 记录的状态转移中学习可微世界模型，并与人形动作跟踪策略联合训练，以短时想象 rollout 和解析策略梯度降低大规模全身动作跟踪的计算成本。

## 核心摘录（论文摘要与官方项目页）

- **方法：** 世界模型学习标准刚体仿真器中的共享动力学；世界模型与策略在同一训练循环中迭代更新。策略更新时对短时想象 rollout 做反向传播，不需要对仿真器求导。
- **计算：** 官方项目页称单张 NVIDIA H100 用 3 小时超过 SONIC 已发布 checkpoint 的归一化回报；SONIC 对照为 128 张 L40、7 天。论文摘要报告 GPU-hours 降低超过 1000 倍。双卡训练 9.8 小时后，基准动作失败数为 0。
- **覆盖：** 在 129,514 个 BONES-SEED 训练动作上，SONIC checkpoint 失败 179 个；SuperMimic 单卡 3 小时失败 25 个、双卡 9.8 小时失败 0 个。该评测反映训练动作覆盖率，不代表未见动作泛化。
- **真机：** 论文与项目页报告同一策略迁移到 Unitree G1，展示多种动态动作；项目页说明视频没有按动作单独调参。

## 对 wiki 的映射

- [paper-supermimic](../../wiki/entities/paper-supermimic.md) — 论文、项目页及独立复现统一映射到同一研究实体
- [SONIC](../../wiki/entities/nvidia-sonic.md) — 相关的大规模人形动作跟踪基线

## 全文补读及开源核对（2026-11-11）

- 论文的方法本质是从 MuJoCo Warp 采集真实仿真转移训练可微动力学，在短时想象轨迹内执行策略梯度优化；仿真器本身不参与求导。论文附录给出训练流程伪代码。
- 对运动覆盖与 SONIC 比较，应保留不同硬件、数据集规模、训练与 checkpoint 评估口径的限制，129,514 是训练运动覆盖集合而非未见动作泛化测试。
- 论文方法与 HybridRobotics 公开仓库的训练流程一致：`train.py` 调用 `SuperMimicRunner`，后者从缓存交替更新世界模型和 BPTT/Actor-Critic 策略。但**尚无证据表明该仓库已经公开论文所有实验所需 checkpoint、数据或真实机器人部署代码**。
- [CoRL 2026 官方 Program](../sites/corl-2026-program.md) 提供大会直播导航，目前没有核验到 SuperMimic 单场视频直链。
