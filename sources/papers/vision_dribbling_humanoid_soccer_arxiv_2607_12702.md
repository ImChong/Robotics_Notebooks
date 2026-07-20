# Vision-Based Dribbling for Humanoid Soccer via Privileged Representation Learning（arXiv:2607.12702）

> 来源归档（ingest）

- **标题：** Vision-Based Dribbling for Humanoid Soccer via Privileged Representation Learning
- **类型：** paper / humanoid soccer / loco-manipulation / vision / privileged distillation / RMA-style
- **arXiv abs：** <https://arxiv.org/abs/2607.12702>
- **arXiv HTML：** <https://arxiv.org/html/2607.12702v1>
- **PDF：** <https://arxiv.org/pdf/2607.12702>
- **项目页：** <https://lab-rococo-sapienza.github.io/learning-to-dribble/>
- **机构：** 罗马大学 Sapienza、西班牙 CSIC-UPC（IRI）、洛桑联邦理工学院（EPFL）
- **作者：** Flavio Maiorana、Valerio Spagnoli、Eugenio Bugli、Flavio Volpi、Daniele Affinita、Vincenzo Suriani、Daniele Nardi、Luca Iocchi
- **硬件：** Booster T1（仿真）
- **仿真：** mjlab
- **发表日期：** 2026-07-15
- **入库日期：** 2026-07-20
- **一句话说明：** 借鉴 **RMA 特权蒸馏**，分两阶段：Phase 1 用特权编码器学 **对手感知运球策略**（四阶段课程：无障碍→静态障碍→动态抢球者）；Phase 2 冻结策略，用 **CNN+GRU 深度时序编码器** 从机载深度图重建同一 latent，实现 **无显式状态估计的视觉运球**。

## 摘要级要点

- **问题：** 人形足球运球需在平衡、控球与动态对手感知间闭环；传统 **YOLO+卡尔曼** 感知与控制解耦，表征未必为控制优化。
- **Phase 1 — 特权策略：** 特权 MLP 将球/最近障碍状态（机体系）映射为 latent $z_t$，与本体感知拼接喂给 PPO actor；四阶段课程逐步引入静态 blocker 与追球 dynamic opponent；Stage 1–3 用 **DAgger 正则 PPO** 防遗忘 Stage 0 基础运球。
- **Phase 2 — 视觉适应：** 冻结策略；深度图 $108\times192$ → CNN → GRU → 投影到同一 $z_t$；损失为 latent MSE + 球/障碍位置速度辅助头。
- **评测协议：** 三类场景（无障碍 / 单静态障碍 / ball attacker）；成功=球-目标距离 $\le0.75$ m；5 seed × 10 trial。
- **主结果：** 最终部署策略（深度编码器）— 无障碍 **100%** SR；静态障碍 **96%**；主动抢球者 **46%**（fall 52%、碰撞 68%）。感知误差跨场景稳定，难点在 **动态对手闭环控制** 而非感知崩溃。
- **课程消融：** 无课程时静态/动态场景 Stage 1 仅 24%/2% SR，完整四阶段后升至 88%/46%。

## 核心摘录（面向 wiki 编译）

### 与模块化感知路线对比

| 维度 | 本文 | 经典 RoboCup 感知栈 |
|------|------|---------------------|
| 感知目标 | **为控制优化的 task latent** | 检测精度 / 滤波平滑 |
| 部署传感 | **头载深度 + 时序编码** | RGB 检测 + 卡尔曼 |
| 对手建模 | 课程化静态→动态障碍 | 多模块手工拼接 |
| 局限 | 仅仿真；动态对手仍难 | 遮挡与高速球易失效 |

### 奖励结构备忘（节选）

- 强稀疏成功项：ball-target reached **50.0**
- 跟踪：ball velocity/speed/heading、ball-target progress
- 安全：robot/ball-obstacle collision **-10**；foot slip、self-collision 等正则

## 对 wiki 的映射

- 沉淀实体页：[视觉特权表征人形足球运球（arXiv:2607.12702）](../../wiki/entities/paper-vision-dribbling-humanoid-soccer-privileged-representation.md)
- 交叉补强：[Humanoid Soccer](../../wiki/tasks/humanoid-soccer.md)、[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)、[RMA](../../wiki/entities/paper-rma-rapid-motor-adaptation.md)、[Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)

## 当前提炼状态

- [x] 两阶段框架、课程、三类评测与数值摘录
- [x] wiki 实体页与 humanoid-soccer 交叉链接规划
- [ ] 项目页材料发布边界待后续 lint 跟进（截至入库日无独立 GitHub 仓）
