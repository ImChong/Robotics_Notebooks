# Stubborn（arXiv:2606.12814）

> 来源归档（ingest · 项目页 + arXiv 全文）

- **标题：** Stubborn: A Streamlined and Unified Reinforcement Learning Framework for Robust Motion Tracking and Fall Recovery for Humanoids
- **缩写：** **Stubborn**
- **类型：** paper / humanoid motion tracking / fall recovery / unified RL
- **arXiv：** <https://arxiv.org/abs/2606.12814>（HTML：<https://arxiv.org/html/2606.12814v1>）
- **PDF：** <https://arxiv.org/pdf/2606.12814>
- **项目页：** <https://aislab-sustech.github.io/Stubborn/>
- **代码：** Coming Soon（项目页标注）
- **作者：** Xiao Ren*, Yuhui Yang*, Zongbiao Weng, Zhijie Liu, He Kong†（*equal contribution；†corresponding）
- **机构：** ACT Lab，南方科技大学（Southern University of Science and Technology, SUSTech）
- **入库日期：** 2026-06-23
- **一句话说明：** 用 **单一 RL 策略** 同时学 **鲁棒全身运动跟踪** 与 **跌倒恢复**：**yaw-aligned 跟踪表征** + **非对称 Actor-Critic** + **Bernoulli 概率终止（PT）** 与 **跟踪误差驱动自适应采样（AdpS）**，避免多阶段训练与独立恢复策略；仿真与 **Unitree G1（29-DoF）** 真机验证。

## 核心摘录（策展，非全文）

1. **问题设定：** 现有 RL 人形跟踪常把 **motion tracking** 与 **fall recovery** 当不同任务，需 **多阶段训练**、**专用恢复奖励** 或 **独立恢复策略**；且严重跟踪失败时 **硬终止 episode**，限制在不稳定/倒地状态下的恢复探索。
   - **对 wiki 的映射：** [paper-motion-cerebellum-stubborn](../../wiki/entities/paper-motion-cerebellum-stubborn.md)、[SD-AMP 统一走跑起身](../../wiki/entities/paper-unified-walk-run-recovery-sdamp.md)

2. **Yaw-aligned 跟踪表征：** 在 **yaw 对齐根坐标系** 下表示身体跟踪误差，削弱对 **全局平移漂移与航向扰动** 的敏感度，同时保留 **重力相关俯仰/横滚** 平衡信息；策略仅用本体感知，Critic 可用仿真特权信息（非对称 AC）。
   - **对 wiki 的映射：** [Whole-Body Control](../../wiki/concepts/whole-body-control.md)、[RGMT](../../wiki/entities/paper-hrl-stack-14-robust_and_generalized_humanoid_moti.md)

3. **Bernoulli 概率终止（PT）：** 当根部高度误差 $|e_{root,z}^p|>\theta_{pos}$ 或姿态误差 $e_{root}^q>\theta_{quat}$ 时，**不以概率 1 终止**，而以概率 $p_{term}$ 终止（条件 Bernoulli）。实验取 $p_{term}=0.005$、$\theta_{pos}=0.25$ m、$\theta_{quat}=\pi/2$，使期望存活步数 $\mathbb{E}[\mathcal{T}]=1/p_{term}=200$（50 Hz × 4 s 恢复窗），为倒地后恢复行为留出探索空间。
   - **对 wiki 的映射：** [paper-motion-cerebellum-stubborn](../../wiki/entities/paper-motion-cerebellum-stubborn.md)

4. **跟踪误差驱动自适应采样（AdpS）：** PT 下 episode 终止不再可靠反映片段难度；Stubborn 用段内 **平均关键点跟踪误差** $\bar{e}$ 在线更新参考帧采样权重——高误差段增权、已成功跟踪段衰减，使训练聚焦难片段与不稳定状态。
   - **对 wiki 的映射：** [BeyondMimic](../../wiki/methods/beyondmimic.md)、[LIMMT](../../wiki/methods/limmt-gqs-motion-curation.md)

5. **实验：** IsaacLab/MuJoCo 上 **完整 LAFAN1** 跟踪；对比 HoloMotion、Any2Track、BFM-Zero、From-scratch multi-motion RL。Stubborn 在 MPBPE/MPJPE/MPJVE 上最优（如 MPBPE 48.85 mm vs From-scratch 62.68）；5 m/s 强扰动下 PT 消融使恢复成功率 **100%**（w/o PT 为 77.5%–85%）。真机 **G1** 演示跟踪、抗扰与倒地恢复（含翻转/杂技类动作）。
   - **对 wiki 的映射：** [LaFAN1](../../wiki/entities/lafan1-dataset.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[HoloMotion](../../wiki/entities/holomotion.md)、[Any2Track](../../wiki/methods/any2track.md)

## 对 wiki 的映射

- 沉淀实体页：[paper-motion-cerebellum-stubborn](../../wiki/entities/paper-motion-cerebellum-stubborn.md)
- 分类 hub：[motion-cerebellum-category-04-wbt-base](../../wiki/overview/motion-cerebellum-category-04-wbt-base.md)
- 姊妹策展：[motion_cerebellum_survey_34_stubborn.md](./motion_cerebellum_survey_34_stubborn.md)

## 参考来源（原始）

- 项目页：<https://aislab-sustech.github.io/Stubborn/>
- arXiv：<https://arxiv.org/abs/2606.12814>
