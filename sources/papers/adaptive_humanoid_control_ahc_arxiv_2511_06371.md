# Towards Adaptive Humanoid Control via Multi-Behavior Distillation and Reinforced Fine-Tuning（arXiv:2511.06371）

> 来源归档（ingest）

- **标题：** Towards Adaptive Humanoid Control via Multi-Behavior Distillation and Reinforced Fine-Tuning
- **类型：** paper / humanoid multi-behavior / distillation / multi-task RL / AMP / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2511.06371>
- **arXiv HTML：** <https://arxiv.org/html/2511.06371v3>
- **PDF：** <https://arxiv.org/pdf/2511.06371>
- **项目页：** <https://ahc-humanoid.github.io>
- **会议：** **AAAI 2026 Oral**
- **机构：** 哈尔滨工程大学、中国电信 TeleAI、中科大、上海科技大学、哈工大、西北工业大学深圳研究院等
- **硬件：** Unitree **G1**（50 Hz 策略 / 500 Hz PD）
- **入库日期：** 2026-06-25
- **一句话说明：** **AHC** 两阶段：**行为专精策略**（起身 $\pi^b_r$ + 平地走 $\pi^b_w$，均含 **AMP**）→ **DAgger + MoE 蒸馏** 得统一 $\pi^d$ → **多任务 PPO 微调**（行为专属 critic + **PCGrad** + 地形课程）得地形自适应 $\pi^{\mathrm{AHC}}$。

## 摘要级要点

- **Stage 1 — 行为专精：** $\pi^b_r$ 俯卧/仰卧起身 + 多 critic + AMP（参考起身 MoCap）；$\pi^b_w$ 速度跟踪平地走 + AMP（LAFAN1）；各训 ~10k iter。
- **蒸馏：** MoE-based $\pi^d$ 仅本体 69 维输入；按状态空间 $\mathcal{S}_r/\mathcal{S}_w$ 监督 mimic 专家动作；~4k iter；蒸馏后已能 **近跌倒恢复 + 更自然站立过渡**。
- **Stage 2 — 强化微调：** 初始化 $\pi^{ft}\leftarrow\pi^d$；walk / recovery **双任务** 并行 PPO（双 GPU）；共享 actor、**分行为 critic**；**PCGrad** 消解梯度冲突；地形：flat / slope(≤16.6°) / hurdle / discrete；继续 **AMP** 保人形性。
- **与 AMP 专题关系：** 策展强调多行为必须先 **压进统一身体系统**，AMP 是各行为专精阶段的 **风格正则**，而非单技能附加项。
- **仿真（Table 1 摘要）：** AHC 走跑 hurdle 成功率 **0.922**、discrete **0.969**；多行为 $\pi^d$ 仅 **0.756 / 0.702**；独立走策略 $\pi^b_w$ 在坡地 **0.000**；起身 recovery discrete **0.969** vs HoST **0.843**。
- **真机：** 跌倒后起身续走、楼梯/坡地/障碍；相对 HoST 关节加速度更平滑（AMP 引导自然起身）。

## 对 wiki 的映射

- 沉淀实体页：[paper-adaptive-humanoid-control.md](../../wiki/entities/paper-adaptive-humanoid-control.md)
- 交叉：[SD-AMP #10](../../wiki/entities/paper-unified-walk-run-recovery-sdamp.md)、[HAML #12](../../wiki/entities/paper-amp-survey-12-haml.md)、[HoST](../../wiki/entities/paper-host-humanoid-standingup.md)、[Balance Recovery](../../wiki/tasks/balance-recovery.md)
- 42 篇栈 #21：[humanoid_rl_stack_21_…](humanoid_rl_stack_21_towards_adaptive_humanoid_control_via_multi_beha.md)
- AMP 策展 #11：[humanoid_amp_survey_11_…](humanoid_amp_survey_11_towards_adaptive_humanoid_control_via_multi_beha.md)

## 参考来源（原始）

- arXiv:2511.06371 / AAAI 2026 Proceedings
- [wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)、[wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)
