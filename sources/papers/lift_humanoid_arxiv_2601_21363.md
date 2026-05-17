# LIFT：Towards Bridging the Gap between Large-Scale Pretraining and Efficient Finetuning for Humanoid Control（arXiv:2601.21363）

> 论文来源归档（ingest）

- **标题：** Towards Bridging the Gap between Large-Scale Pretraining and Efficient Finetuning for Humanoid Control
- **缩写：** **LIFT**（Large-scale pretraIning and efficient FineTuning）
- **类型：** paper / humanoid RL + model-based finetuning
- **arXiv：** <https://arxiv.org/abs/2601.21363>（HTML 版便于检索公式与附录：<https://arxiv.org/html/2601.21363v3>）
- **PDF：** <https://arxiv.org/pdf/2601.21363.pdf>
- **机构：** 北京通用人工智能研究院（BIGAI）等
- **会议标注（以官方为准）：** 项目站 / 仓库 README 写作 **ICLR 2026**；录用状态以官方更新为准
- **入库日期：** 2026-05-17
- **一句话说明：** 用 **大规模并行 JAX SAC（大 batch、较高 UTD）** 做人形 locomotion **仿真预训练** 与部分 **零样本 sim-to-real**；再用 **物理知情（拉格朗日 + 残差接触/耗散）世界模型** 做 **模型内随机探索**，真机/目标环境侧 **只执行确定性均值动作** 采集，以提升 **微调安全性与样本效率**。

## 摘要级要点（与 abs 一致）

- **问题：** PPO 等 on-policy 路线在 GPU 大规模仿真里墙钟快，但 **样本效率** 与 **新环境安全适应** 受限；纯 off-policy / model-based 在人形上要么 **大规模并行 + sim2real 证据不足**，要么 **真机随机探索风险高**。
- **阶段 (i)：** **SAC + 非对称 actor–critic**（actor 本体感知、critic 特权状态），**MuJoCo Playground** 上千并行环境；**JAX** 固定形状张量、**大 replay + 高 UTD**；报告 **约 1×RTX 4090 墙钟量级** 的 Booster T1 任务收敛叙事（含 **Optuna** 超参搜索前后对比，以论文为准）。
- **阶段 (ii)：** 预训练期 **全量落盘 transition**，策略收敛后 **离线** 训练世界模型：**Brax 可微刚体（M/C/G）+ 半隐式欧拉 + PD 扭矩映射**，残差网络预测 **等效外扰扭矩 + 预测方差**；特权状态显式包含 **基座高度**（论文强调人形若省略高度则 rollout 不稳）。
- **阶段 (iii)：** 新环境中 **确定性策略采集**；**世界模型与策略交替微调**；**随机性仅出现在模型内 SAC rollout**（与真机随机探索解耦）。

## 对 wiki 的映射

- [`wiki/entities/lift-humanoid.md`](../../wiki/entities/lift-humanoid.md) — 方法栈、仿真链路与安全–效率折中的归纳页
- [`sources/sites/lift-humanoid-github-io.md`](../sites/lift-humanoid-github-io.md) — 项目页对实验场景、视频与补充结果的归档
- [`sources/repos/bigai-lift-humanoid.md`](../repos/bigai-lift-humanoid.md) — 开源仓库入口与工程映射
