# CMoE: Contrastive Mixture of Experts for Motion Control and Terrain Adaptation of Humanoid Robots（ICRA 2026）

> 来源归档（ingest · arXiv + 项目页 + 官方代码仓）

- **标题：** CMoE: Contrastive Mixture of Experts for Motion Control and Terrain Adaptation of Humanoid Robots
- **简称（本库）：** **CMoE**（Contrastive Mixture of Experts）
- **类型：** paper / humanoid / perceptive-locomotion / elevation-map / mixture-of-experts / contrastive-learning / single-stage-rl / unitree-g1
- **arXiv：** [2603.03067](https://arxiv.org/abs/2603.03067)
- **PDF：** <https://arxiv.org/pdf/2603.03067>
- **项目页：** <https://hoshi-no-ai.github.io/CMoE/>
- **代码：** <https://github.com/Hoshi-No-Ai/CMoE>（BSD-3-Clause；`Fudan-MAGIC-Lab/CMoE` 截至 2026-08-29 为空占位，不是可用镜像）
- **mjlab 移植：** <https://github.com/senlanke/mimic> 任务 `CMoE-G1` — [`sources/repos/senlanke_mimic.md`](../repos/senlanke_mimic.md)
- **视频：** <https://www.youtube.com/watch?v=Q95Ssg1FP7A>
- **作者：** Shihao Ma、Hongjin Chen、Zijun Xu、Yi Zhao、Ke Wu、Ruichen Yang、Leyao Zou、Zhongxue Gan、Wenchao Ding（复旦大学智能机器人与先进制造学院）
- **会议：** ICRA 2026
- **入库日期：** 2026-08-23
- **一句话说明：** 单阶段 PPO + 高程图感知人形 RL：Vanilla MoE 门控在多地形的专家激活趋于均匀；CMoE 用 **SwAV 式地形对比学习** 约束门控与高程 latent 的聚类分配，使专家按地形分化；G1 真机连续 20 cm 台阶、80 cm 沟、30 cm 栏与混合地形。

## 开源状态（步骤 2.5）

- **核查日：** 2026-08-29。项目页 Footer / README 均链至 `Hoshi-No-Ai/CMoE`；仓库含 `legged_gym`（task `g1cmoe`）与定制 `rsl_rl`（`cmoe_ppo`、`cmoe_actor_critic`、双 estimator）。
- **已发布：** 训练脚本 `train.py`、可视化 `play.py`、环境配置、对比损失与 estimator 模块；BSD-3-Clause。README「Deployment References」指向 `elevation_mapping_humanoid` 与 `rl_sar`。
- **未发布 / 勿误用：** 无预训练 checkpoint；`Fudan-MAGIC-Lab/CMoE` 为空占位；真机节点不在官方仓内。
- **社区移植：** [senlanke/mimic](https://github.com/senlanke/mimic) 任务 `CMoE-G1`（mjlab，结构对齐，无官方数字）。
- **结论：** **已开源**（可运行仿真训练栈）；权重与真机感知/部署需接社区仓。

## 摘录 1：问题与 Vanilla MoE 失效（摘要 + §I）

人形需在砾石→坡道、踏石→软面等**突变混合地形**上连续行走。两阶段「单地形预训 + 蒸馏」虽能减轻灾难性遗忘，但训练成本高且易过拟合。Vanilla MoE 虽可并行建模多地形，实测 **门控在各地形上专家激活近乎均匀**（Fig. 2 t-SNE），无法按环境特征激活技能，限制表达力与穿越极限。

CMoE 提出 **单阶段 RL**：MoE actor-critic + **对比学习**——**同地形内**最大化专家激活分布一致性，**跨地形**最小化相似性，促使专家专精不同地表类型。

**对 wiki 的映射：** 新建 [`wiki/entities/paper-cmoe.md`](../../wiki/entities/paper-cmoe.md)；对照 [MoRE](../../wiki/entities/paper-amp-survey-08-more.md)（两阶段 + gait MoE）、[TRAMP](../../wiki/entities/paper-tramp-vision-assisted-bipedal-locomotion.md)（单阶段 MoE + AMP，无对比防塌缩）、[Hiking in the Wild](../../wiki/entities/paper-hiking-in-the-wild.md)（单阶段 MoE + 深度，非高程图对比门控）。

## 摘录 2：方法栈（§III）

1. **信息编码：** 本体历史经 **β-VAE** 估计体速与隐状态；高程图经 **AE** 自预测提取地形特征（式 3–5）。
2. **MoE 策略：** actor/critic 各含多组 expert；**共享门控** softmax 加权专家输出（式 6）；PPO 优化。
3. **地形对比学习（§III-E）：** 门控输出与高程 latent 经 MLP 对齐维度，用 **可学习 prototype + SwAV + Sinkhorn-Knopp** 做聚类分配互预测（式 7–8）；同轨迹内 \(\langle g^z, e^z \rangle\) 为正样本，否则为负。
4. **训练：** Isaac Gym，4096 并行，20k epoch，**5 experts**，高程图 0.7 m × 1.1 m，prototype=32，τ=0.2；八类地形（坡/楼梯/沟/栏/离散/两种混合）+ 课程学习。

**对 wiki 的映射：** [地形适应](../../wiki/concepts/terrain-adaptation.md)、[Privileged Training](../../wiki/concepts/privileged-training.md)、[强化学习](../../wiki/methods/reinforcement-learning.md)。

## 摘录 3：仿真与真机结果（§V）

**仿真 benchmark（3 m × 18 m，0.8 m/s，20 s）：** CMoE 在八类地形成功率与平均行进距离均优于 Vanilla MoE 与等参 Base。典型：gap 成功率 **0.974 vs 0.818**（Vanilla MoE）；mix1 **0.767 vs 0.605**；楼梯上行 **0.886 vs 0.798**。

**对比学习消融：** t-SNE 显示 CMoE 专家激活按地形聚类（上/下楼阶可分层）；Vanilla MoE 激活几乎不随地形变化。Expert 1 专精**上行/抬腿**地形——移除后上楼失败、下楼仍可走。

**真机（Unitree G1 + 雷达点云→高程图）：** 最大沟宽 **80 cm**、连续台阶 **20 cm**、栏高 **30 cm**、17° 坡；混合地形（台阶+沟+坡+栏）与未训练户外台阶、拖拽/碰撞扰动下仍稳定。

**对 wiki 的映射：** [楼梯/障碍感知 locomotion](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[Humanoid Locomotion](../../wiki/tasks/humanoid-locomotion.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)。

## 对 wiki 的映射（汇总）

- 新建实体页：[`wiki/entities/paper-cmoe.md`](../../wiki/entities/paper-cmoe.md)
- 交叉更新：[`wiki/tasks/stair-obstacle-perceptive-locomotion.md`](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[`wiki/entities/paper-tramp-vision-assisted-bipedal-locomotion.md`](../../wiki/entities/paper-tramp-vision-assisted-bipedal-locomotion.md)、[`wiki/entities/smp-g1-mjlab.md`](../../wiki/entities/smp-g1-mjlab.md)

## 参考来源（原始）

- arXiv：<https://arxiv.org/abs/2603.03067>
- 项目页：<https://hoshi-no-ai.github.io/CMoE/>
- 代码：<https://github.com/Hoshi-No-Ai/CMoE>
- 视频：<https://www.youtube.com/watch?v=Q95Ssg1FP7A>
