# Extreme Parkour with Legged Robots（arXiv:2309.14341）

> 来源归档（ingest）

- **标题：** Extreme Parkour with Legged Robots
- **类型：** paper / 四足 locomotion / 感知跑酷 / Teacher–Student 蒸馏
- **arXiv：** <https://arxiv.org/abs/2309.14341>（PDF：<https://arxiv.org/pdf/2309.14341>）
- **会议：** ICRA 2024；CoRL 2023 Generalist / Roboletics / Deployable Workshop（Oral）
- **项目页：** <https://extreme-parkour.github.io/>
- **代码：** <https://github.com/chengxuxin/extreme-parkour>
- **作者：** Xuxin Cheng*, Kexin Shi*, Ananye Agarwal, Deepak Pathak（CMU / Pathak Lab）
- **入库日期：** 2026-06-04
- **一句话说明：** 在 **Isaac Gym + legged_gym** 上用 **两阶段 RL + 双重蒸馏**（特权 scandots→深度、oracle 航向→自预测 yaw），训练 **Unitree Go1** 级低成本四足 **端到端深度跑酷** 策略；约 **20 小时** 仿真训练即可实机高跳 / 远跳 / 手倒立 / 斜坡变向。

## 摘要级要点

- **问题：** 传统跑酷依赖 **分模块高精度工程**（感知 / 执行 / 控制分别调参），难以泛化到新障碍组合；人类则通过练习在 **不改造硬件** 的前提下习得技能。
- **平台约束：** 小型低成本四足、**执行器不精确**、仅 **单目前向深度相机**（低频、抖动、伪影）——刻意贴近真实 onboard 限制。
- **主张：** **单神经网络** 从深度图直接输出关节控制；仿真 **大规模 RL** 克服感知与执行噪声，实机展现 **高精度** 动态动作。
- **能力（项目页 / 论文）：** **2× 身高高跳**、**2× 体长远跳**、手倒立、倾斜 ramp 上跑并 **自主变向**；对新障碍组合与不同物理属性 **零样本泛化**；CoRL 2023 **2 分钟不间断** 攀爬 / 远跳 / 跳下演示。
- **Phase 1 — 特权 scandots RL：** PPO 训练 base policy，输入 **本体 $x$ + scandots $m$ + oracle 航向 $\hat{d}$（terrain waypoint 计算）+ 行走标志 $W$ + 速度指令**；用 **ROA（Regularized Online Adaptation）** 从历史观测估计环境参数（RMA 思路）。
- **Phase 2 — 双重蒸馏：** Teacher 依赖的两类特权在部署时不可用——(1) scandots → **ConvNet–GRU 深度管线**（RMA 式）；(2) waypoint 航向 → **从深度自预测 yaw**。外感知与航向均用 **DAgger** 向 Phase 1 行为对齐。
- **统一奖励（inner-product）：** 航向不再随机采样，而由 **terrain 上 red-dot waypoint** 给出 $\hat{d}_w = (p-x)/\|p-x\|$，速度跟踪奖励与之对齐；避免「绕障碍转圈」 exploit。
- **Clearance 惩罚：** $r_{\mathrm{clearance}} = -\sum_{i=0}^{4} c_i \cdot M[p_i]$，足端接触点距地形边缘 **< 5 cm** 时惩罚——否则大 gap 上策略会 **贴边省能** 导致实机失败（项目页 ablation）。
- **Direction distillation ablation：** 无航向蒸馏时 ramp 地形 **摇杆遥控几乎不可控**（需自主选跳上角与即时变向，人类无法实时给 waypoint）。
- **训练成本（README）：** Base **10–15k iter（3090 约 8–10 h，建议 ≥15k）**；Distillation **5–10k iter（5–10 h，建议 ≥5k）**；合计 **< 20 h** 量级。

## 对 wiki 的映射

- 沉淀实体页：`wiki/entities/extreme-parkour.md`
- 交叉更新：`wiki/concepts/privileged-training.md`、`wiki/entities/legged-gym.md`、`wiki/tasks/locomotion.md`、`wiki/methods/dagger.md`
- 姊妹工作：Robot Parkour Learning（ZiwenZhuang / Unitree）、DreamWaQ++、PHP（人形跑酷）
