---
title: 人形运动跟踪方法选型指南
type: query
status: complete
created: 2026-05-21
updated: 2026-06-26
summary: 在人形 RL 运动控制栈中，如何按任务阶段在 DeepMimic / BeyondMimic / AMP 家族 / 通用 tracker / 生成式动作先验之间选型。
sources:
  - ../../sources/papers/deepmimic.md
  - ../../sources/papers/amp.md
  - ../../sources/papers/smp.md
  - ../../sources/papers/heracles_humanoid_diffusion_arxiv_2603_27756.md
  - ../../sources/papers/phygile_arxiv_2603_19305.md
  - ../../sources/papers/unified_walk_run_recovery_sdamp_arxiv_2605_18611.md
  - ../../sources/papers/sprint_arxiv_2605_28549.md
  - ../../sources/papers/any2any_arxiv_2605_23733.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md
---

> **Query 产物**：本页由以下问题触发：「人形运动跟踪与风格先验方法这么多，工程上怎么选、怎么组合？」
> 综合来源：[DeepMimic](../methods/deepmimic.md)、[BeyondMimic](../methods/beyondmimic.md)、[AMP & HumanX](../methods/amp-reward.md)、[Locomotion](../tasks/locomotion.md)、[人形 AMP 先验综述](../overview/humanoid-amp-motion-prior-survey.md)

# 人形运动跟踪方法选型指南

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBT | Whole-Body Tracking | 参考动作跟踪类方法总称 |
| AMP | Adversarial Motion Prior | 分布约束式运动先验路线 |
| RL | Reinforcement Learning | 任务奖励与先验联合优化 |
| MoCap | Motion Capture | 参考动作与风格数据来源 |
| Sim2Real | Simulation to Real | 跟踪策略上真机的迁移考量 |

## TL;DR 决策路径

```mermaid
flowchart TD
  A[有高质量参考轨迹?] -->|否| B[生成式先验 / 扩散动作]
  A -->|是| C[首要目标是逐帧贴合?]
  C -->|是| D[DeepMimic / BeyondMimic 显式跟踪]
  C -->|否| E[首要目标是自然步态/风格?]
  E -->|是| F[AMP / ADD / SMP 运动先验]
  F --> G[需要实时全身 tracker?]
  G -->|是| H[Any2Track / AMS / MotionBricks]
  G -->|接触柔顺| I[GentleHumanoid]
```

| 阶段目标 | 优先方法族 | 典型入口 |
|----------|------------|----------|
| 证明「能跟参考跑起来」 | 显式 tracking reward | [DeepMimic](../methods/deepmimic.md)、[BeyondMimic](../methods/beyondmimic.md) |
| 任务完成后仍像「人」 | 对抗式 motion prior | [AMP](../methods/amp-reward.md)、[ADD](../methods/add.md)、[SMP](../methods/smp.md) |
| 多动作通用 tracker | 规模化 tracking policy | [Any2Track](../methods/any2track.md)、[AMS](../methods/ams.md)、[MotionBricks](../methods/motionbricks.md)、[EGM](../methods/egm-efficient-general-mimic.md)、[SONIC](../methods/sonic-motion-tracking.md)、[Humanoid-GPT](../entities/paper-humanoid-gpt.md) |
| 数据稀缺、要合成参考 | 生成式动作 | [ASE](../methods/ase.md)、[GenMo](../methods/genmo.md)、[扩散动作生成](../methods/diffusion-motion-generation.md) |

---

## 分阶段选型说明

### 1. 显式跟踪：先解决「跟得上」

[DeepMimic](../methods/deepmimic.md) 用多 term 跟踪奖励 + RSI，适合作为**第一条可复现基线**。[BeyondMimic](../methods/beyondmimic.md) 在同类框架上面向人形与更复杂参考，适合在 DeepMimic 已跑通后升级。

**常见误判**：把 tracking MSE 当成最终目标——高频抖动往往说明需要进入 motion prior 阶段，而不是继续堆 tracking 权重。

### 2. Motion prior：再解决「像不像」

当任务奖励已满足，仍出现步态不自然时，引入 [AMP](../methods/amp-reward.md) 判别器先验。[ADD](../methods/add.md) 用对抗差分减轻多目标手调；[SMP](../methods/smp.md) 走 **冻结扩散 + SDS** 路线（非判别器），先验预训练后可**丢弃原始 MoCap**、在多任务多策略间复用，代价是两阶段训练、同采样量 wall-clock 约为 AMP 的 ~1.8×（论文报告 600M samples：SMP ~11.5h vs AMP ~6.2h）。

**选型轴**：每任务都要重训先验 / 必须保留数据集 → AMP/ADD；先验一次训好跨任务复用、不愿在 RL 阶段保留 MoCap → SMP。三者对比见 [AMP / ADD / SMP 运动先验变体对比](../comparisons/amp-add-smp-motion-prior-variants.md)。

### 3. 通用 tracker 与实时原语

[MotionBricks](../methods/motionbricks.md) 强调实时 smart primitives + 全身控制；[Any2Track](../methods/any2track.md)、[AMS](../methods/ams.md) 面向**多参考、抗扰、负载变化**的通用跟踪器，常作为「身体基础模型」层。

当瓶颈不在网络结构而在**数据不平衡与高动态精度**时，看 [EGM](../methods/egm-efficient-general-mimic.md)：它用 **bin 级误差驱动的跨动作采样课程** + **上下身分组 CDMoE**，论证「小而高质量的精选 MoCap 子集优于大规则筛集」，把选型轴从「堆更多小时数据」转向「数据策展 + 采样调度」。

若已有 **AMASS 级大库** 且 tracker 已选定（如 Any2Track / TWIST2），优先评估 **[LIMMT / GQS](../methods/limmt-gqs-motion-curation.md)**：**离线** 三阶段策展（仿真可行性 → HME 多样性 → 复杂度加权 FPS）可在 **≈3% 数据** 上击败全量训练，且 **plug-and-play** 不改动算法——适合作为 WBT **阶段 3 前置数据模块**。

### 4. 接触柔顺与生成式补充

[GentleHumanoid](../methods/gentlehumanoid-motion-tracking.md) 把力/柔顺约束写进跟踪目标，适合接触丰富场景。参考不足时，[ASE](../methods/ase.md)、[GenMo](../methods/genmo.md)、[扩散动作生成](../methods/diffusion-motion-generation.md) 用于扩充或平滑参考分布。

当入口是 **自然语言** 且目标是 **机器人可执行的高动态全身**（而非人体 SMPL 再 retarget）时，优先评估 **[PhyGile](../entities/paper-phygile.md)**：**262D robot-native 扩散 + physics-prefix + GMT 验证/微调闭环**；与 [Harmon](../entities/paper-loco-manip-161-097-harmon.md) 同族但强调 **物理前缀与跟踪器共训**，避免人体 T2M 先验的推理期重定向鸿沟。

### 5. 大扰动：跟踪 vs 生成中间件 vs 统一 AMP

| 目标 | 优先路线 | 入口 |
|------|----------|------|
| 保持 tracker，只在 OOD 改参考 | **状态条件生成 middleware + tracker** | [Heracles](../entities/paper-heracles-humanoid-diffusion.md) |
| 单策略 RL，训练期分离 recovery/loco 先验 | **SD-AMP 双判别器** | [SD-AMP](../entities/paper-unified-walk-run-recovery-sdamp.md) |
| 工程复现统一 walk+recovery（mjlab） | **AMP_mjlab 统一判别器** | [AMP_mjlab](../entities/amp-mjlab.md) |

**常见误判**：把 Heracles 当作「又一个 tracking 论文」——其贡献在 **中间层改参考命令**，底层仍是高频 tracking MDP。

### 6. 竞技冲刺：参考极少 + 连续全速域

当目标是 **6 m/s 级冲刺**、**走–跑–冲无缝变速**，且 **人形可用 MoCap 极少** 时：

| 目标 | 优先路线 | 入口 |
|------|----------|------|
| 频域外推 + 单策略全速域 | **频率自适应频谱先验 + 冻结先验 + 残差 PPO** | [SPRINT](../entities/paper-sprint-humanoid-athletic-sprints.md) |
| 单演示扩展周期参考库 + 控制引导 | **动态重定向 + goal-conditioned RL** | [Chasing Autonomy](../methods/chasing-autonomy-pipeline.md) |
| 对抗风格 + 跌倒/起身统一 | **SD-AMP 双判别器** | [SD-AMP](../entities/paper-unified-walk-run-recovery-sdamp.md) |

**常见误判**：在冲刺段继续堆 AMP 参考——论文指出高动态下 AMP 易不稳定；SPRINT 用 **5 条 LAFAN1 单周期 + 频谱生成** 外推，与「多 clip 对抗先验」是不同数据假设。

### 7. 跨具身：已有 WBT 专家迁到新硬件

当 **源机上已有大规模 WBT 专家**（如 [SONIC](../methods/sonic-motion-tracking.md) / Gear-SONIC on G1），而目标机 DoF、观测布局与动力学不同时：

| 目标 | 优先路线 | 入口 |
|------|----------|------|
| 少数据、少算力迁到新机型 | **运动学对齐 + 局部 LoRA 动力学适配** | [Any2Any](../entities/paper-any2any-cross-embodiment-wbt.md) |
| 从零获得单平台最强 tracker | **继续 scaling 预训练** | [SONIC](../methods/sonic-motion-tracking.md)、[Humanoid-GPT](../entities/paper-humanoid-gpt.md)（2B 帧 + Transformer 蒸馏，CVPR 2026） |
| 多机统一 generalist | **多具身联合预训练 / 统一动作空间** | 见 [BFM](../entities/paper-behavior-foundation-model-humanoid.md) 等 |

**常见误判**：把 Any2Any 当作「再训一个 SONIC」——其设定是 **冻结单源专家 + 后训练**，与亿级帧从头预训练的算力预算不同；运动学对齐层必须覆盖 **髋轴、闭链** 等结构差异，不能只做关节 index 重排。

> **三路径展开**：单具身重训 / Any2Any / 多具身联合训练的「算力 × 数据 × 泛化」决策树与故障模式，见 [跨具身策略迁移选型指南](./cross-embodiment-transfer-strategy.md)。

---

## 推荐组合 pipeline

| Pipeline | 组合 | 适用 |
|----------|------|------|
| **经典 mimic** | DeepMimic → BeyondMimic | 单动作高保真、论文复现 |
| **AMP 增强** | BeyondMimic + AMP/ADD/SMP | 行走/舞蹈等需自然风格 |
| **通用 tracker** | GMR/NMR 重定向 → Any2Track/AMS | 多动作库、遥操作闭环 |
| **跨具身 WBT** | 源机 Sonic/Oli-WBT → Any2Any 对齐+LoRA | 新机少量数据、保留源先验 |
| **接触任务** | GentleHumanoid + 下游操作/搬运 | 推、扶、柔顺交互 |
| **竞技冲刺** | LAFAN1→GMR 五周期 → 频谱先验 → 残差 PPO | G1 零样本 0–6 m/s（[SPRINT](../entities/paper-sprint-humanoid-athletic-sprints.md)） |

---

## 常见误区

1. **AMP ≠ 更好 tracking**：AMP 约束的是**状态转移分布**，不能替代任务奖励与稳定跟踪基线。
2. **生成式先验不能跳过仿真验证**：扩散/ASE 产物仍需进物理仿真做 feasibility 检查。
3. **tracker 与 prior 混在同一 reward**：建议分阶段训练或明确权重 schedule，避免梯度互相掩盖。

---

## 参考来源

- [DeepMimic 论文摘要](../../sources/papers/deepmimic.md)
- [AMP 论文摘要](../../sources/papers/amp.md)
- [具身智能研究室：人形 AMP 先验综述](../../sources/blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)
- [Heracles（arXiv:2603.27756）](../../sources/papers/heracles_humanoid_diffusion_arxiv_2603_27756.md)、[PhyGile（arXiv:2603.19305）](../../sources/papers/phygile_arxiv_2603_19305.md)、[SD-AMP（arXiv:2605.18611）](../../sources/papers/unified_walk_run_recovery_sdamp_arxiv_2605_18611.md)、[SPRINT（arXiv:2605.28549）](../../sources/papers/sprint_arxiv_2605_28549.md)
- [Any2Any（arXiv:2605.23733）](../../sources/papers/any2any_arxiv_2605_23733.md)

## 关联页面

- [DeepMimic](../methods/deepmimic.md)、[BeyondMimic](../methods/beyondmimic.md)
- [AMP & HumanX](../methods/amp-reward.md)、[ADD](../methods/add.md)、[SMP](../methods/smp.md)
- [MotionBricks](../methods/motionbricks.md)、[Any2Track](../methods/any2track.md)、[AMS](../methods/ams.md)、[EGM](../methods/egm-efficient-general-mimic.md)
- [GentleHumanoid](../methods/gentlehumanoid-motion-tracking.md)
- [ASE](../methods/ase.md)、[GenMo](../methods/genmo.md)、[扩散动作生成](../methods/diffusion-motion-generation.md)
- [AMP / ADD / SMP 对比](../comparisons/amp-add-smp-motion-prior-variants.md)
- [SONIC vs BeyondMimic vs SD-AMP vs Heracles 对比](../comparisons/sonic-vs-beyondmimic-vs-sdamp-vs-heracles.md)
- [人形 RL 运动控制身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
- [人形 RL Cookbook](./humanoid-rl-cookbook.md)
- [Heracles](../entities/paper-heracles-humanoid-diffusion.md)、[PhyGile](../entities/paper-phygile.md)、[SD-AMP](../entities/paper-unified-walk-run-recovery-sdamp.md)、[SPRINT](../entities/paper-sprint-humanoid-athletic-sprints.md)
- [Any2Any](../entities/paper-any2any-cross-embodiment-wbt.md)

## 一句话记忆

> **先 DeepMimic 证明能跟，再 AMP 家族修风格，最后 Any2Track/AMS 做通用 tracker；接触与生成式是两条侧向增强线。**
