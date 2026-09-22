# Multi-Expert Distillation for Legged Locomotion — 一手资料合集（ingest 策展）

> 来源归档（ingest · 方法谱系一手资料索引）
>
> 本文件 **不替代** 各论文原文；为「多专家蒸馏」方法 ingest 的 **一手来源导航**，供 [`wiki/methods/multi-expert-distillation.md`](../../wiki/methods/multi-expert-distillation.md) 编译。

- **类型：** survey-index / locomotion / distillation / dagger / teacher-student
- **入库日期：** 2026-09-22
- **一句话说明：** 足式/人形 **分任务 RL 专家 → DAgger（或 BC 聚合）→ 可部署单策略** 的主线一手资料索引；含 **RL 微调**、**转移/切换数据** 与 **感知模态迁移** 变体。

## 理论锚点

| 资料 | 链接 | 角色 |
|------|------|------|
| Ross et al., DAgger | [sources/papers/ross_dagger_aistats_2011.md](./ross_dagger_aistats_2011.md) · [wiki/entities/paper-ross-dagger.md](../../wiki/entities/paper-ross-dagger.md) | 在线聚合专家标注、缓解 covariate shift 的 **算法原论文** |
| Privileged training 概念 | [sources/papers/privileged_training.md](./privileged_training.md) | 专家用特权感知、学生用可部署传感的 **问题设定** |

## 四足 / 跑酷：显式「multi-expert distillation」或同构管线

| # | 论文 | 一手归档 | 实体页 | 专家→学生要点 |
|---|------|----------|--------|---------------|
| 1 | **Parkour in the Wild**（IJRR 2026 / arXiv:2505.11164） | [parkour_in_the_wild_arxiv_2505_11164.md](./parkour_in_the_wild_arxiv_2505_11164.md) | [paper-parkour-in-the-wild.md](../../wiki/entities/paper-parkour-in-the-wild.md) | **9 地形 RL 专家**（高程图）→ **DAgger + 4 深度 LSTM** → **RLFT** + 3D 扫描增广；ANYmal D |
| 2 | **Robot Parkour Learning**（CoRL 2023 / arXiv:2309.05665） | [hmi_p130_robot-parkour-learning.md](./hmi_p130_robot-parkour-learning.md) | [paper-robot-parkour-learning.md](../../wiki/entities/paper-robot-parkour-learning.md) | **5 跑酷技能**（软→硬动力学课程）→ **DAgger** 循环 **深度视觉** 单策略；A1/Go1；**已开源** |
| 3 | **ANYmal Parkour**（SciRob 2023） | 项目 PDF [robot-parkour.github.io](https://robot-parkour.github.io/resources/Robot_Parkour_Learning.pdf) 同系 | （分层对照，无蒸馏实体页） | **分层选技能 vs 蒸馏** 的方法论对照；PITW Related work 明确引用 |

## 人形 / 全身：多专家 DAgger + 感知迁移

| # | 论文 | 一手归档 | 实体页 | 专家→学生要点 |
|---|------|----------|--------|---------------|
| 4 | **Light-Loco-Parkour**（Light Origins 2026） | [light_loco_parkour_light_origins_2026.md](./light_loco_parkour_light_origins_2026.md) | [paper-light-loco-parkour.md](../../wiki/entities/paper-light-loco-parkour.md) | loco + 多 skill **teacher** → **多专家 DAgger**（无技能标签 height-scan）→ **transition RL** → **GRU 深度**；Lightbot 0 |
| 5 | **RPL**（arXiv:2602.03002） | [rpl_arxiv_2602_03002.md](./rpl_arxiv_2602_03002.md) | [paper-rpl-robust-humanoid-perceptive-locomotion.md](../../wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md) | **分地形高程专家** → **DAgger** 多视角深度 Transformer；DFSV/RSM |
| 6 | **LadderMan**（arXiv:2606.05873） | [ladderman_arxiv_2606_05873.md](./ladderman_arxiv_2606_05873.md) | [paper-ladderman-humanoid-perceptive-ladder-climbing.md](../../wiki/entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md) | 多几何攀爬专家 → **DAgger+PPO+KL** 深度 visuomotor |
| 7 | **PHP**（arXiv:2602.15827） | [php_parkour_arxiv_2602_15827.md](./php_parkour_arxiv_2602_15827.md) | [paper-hrl-stack-22-perceptive_humanoid_parkour.md](../../wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md) | 跑酷 **teacher** → **DAgger+PPO** 学生；纯 DAgger 攀爬不足 |

## 全身跟踪 / scaling：专家蒸馏（非地形专家，但同范式）

| # | 论文 | 备注 |
|---|------|------|
| 8 | **Humanoid-GPT** | [sources/sites/humanoid-gpt-qizekun-github-io.md](../sites/humanoid-gpt-qizekun-github-io.md) — **数百 RL expert → DAgger 合并** |
| 9 | **Athena-WBC** | [athena_wbc_arxiv_2607_04837.md](./athena_wbc_arxiv_2607_04837.md) — **能力对齐专家 → 单 student**；RLFT 参照 PITW |
| 10 | **EAGLE** | [humanoid_pnb_embodiment-aware-generalist-specialist-distillat.md](./humanoid_pnb_embodiment-aware-generalist-specialist-distillat.md) — 多本体 **specialist → DAgger generalist** |

## 常见工程变体（一手资料中的反复出现）

1. **感知模态切换：** 专家 privileged height map / scan → 学生 onboard depth（PITW、RPL、Robot Parkour、LightLP）。
2. **蒸馏后 RL：** 纯 DAgger 在多模态地形上 **平均化**；加 **PPO fine-tune** 恢复任务奖励与泛化（PITW、LadderMan、LightLP）。
3. **切换/转移数据：** 仅蒸馏不够 loco↔技能边界 → **transition group RL** 或稀疏切换奖励（LightLP Table VII：无 transition **0%**）。
4. **动作噪声：** 蒸馏 rollout 加噪 → 利于后续 RL 探索（PITW Algorithm 1）。
5. **与 MoE/路由对照：** 运行时 **门控选 expert**（TeleGate、CMoE）vs **蒸馏进单网络**；见 [humanoid-policy-network-architecture.md](../../wiki/concepts/humanoid-policy-network-architecture.md)。

## 对 wiki 的映射

- 方法总页：[`wiki/methods/multi-expert-distillation.md`](../../wiki/methods/multi-expert-distillation.md)
- 算法底座：[`wiki/methods/dagger.md`](../../wiki/methods/dagger.md)、[`wiki/methods/teacher-student-dagger-training.md`](../../wiki/methods/teacher-student-dagger-training.md)
- 路线：[`roadmap/depth-perceptive-locomotion.md`](../../roadmap/depth-perceptive-locomotion.md)
