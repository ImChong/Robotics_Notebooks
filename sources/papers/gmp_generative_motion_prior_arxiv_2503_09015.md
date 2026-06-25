# Natural Humanoid Robot Locomotion with Generative Motion Prior（arXiv:2503.09015）

> 来源归档（ingest）

- **标题：** Natural Humanoid Robot Locomotion with Generative Motion Prior
- **类型：** paper / humanoid locomotion / generative motion prior / CVAE / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2503.09015>
- **arXiv HTML：** <https://arxiv.org/html/2503.09015v1>
- **PDF：** <https://arxiv.org/pdf/2503.09015>
- **项目页：** <https://sites.google.com/view/humanoid-gmp>
- **机构：** 浙江大学；浙江人形机器人创新中心（Zhejiang Humanoid Robot Innovation Center）
- **硬件：** NAVIAI 全尺寸人形（1.65 m / 60 kg；控制 21 DoF）
- **入库日期：** 2026-06-25
- **一句话说明：** **GMP** 用离线训练的 **CVAE 生成式运动先验**（冻结）在线自回归合成全身参考轨迹，以关节角与关键点位置的 **稠密 motion guidance reward** 指导 PPO 策略，相对 AMP 标量风格分提供更细、更稳的自然走跑监督。

## 摘要级要点

- **问题：** 纯 RL 忽视运动自然性；AMP 等对抗先验给出 **模糊标量风格奖励**，联合训练不稳定、可解释性差。
- **管线：** (1) 全身 MoCap retarget → 机器人参考数据集；(2) **CVAE + command encoder** 离线训练生成先验；(3) RL 阶段 **冻结** 先验，按当前姿态与速度命令 **自回归预测** 未来参考 $\hat{\bm{m}}_{t+1}$；(4) 策略观测含参考运动，奖励含 $r_{\mathrm{guidance}}=r_{\mathrm{dof}}+r_{\mathrm{keypos}}$ + 任务/正则项。
- **CVAE：** 编码器 $f_\theta(\bm{m}_{t+1},\bm{m}_t)$ → 潜变量；解码器 $f_\phi(\bm{z}_{t+1},\bm{m}_t)$ 重建下一帧；command encoder $f_\psi(\bm{c}_t,\bm{m}_t)$ 使生成轨迹服从速度命令；scheduled sampling 支持长序列。
- **与 AMP 对照：** AMP 判别「像不像」；GMP 直接给 **轨迹级** 关节/关键点目标——策展导读称「不只告诉你不像，还告诉你该往哪靠近」。
- **数据：** XSenS MoCap，37 序列 / 47,7k 帧 @ 50 Hz；镜像增广。
- **训练：** Isaac Gym + PPO；单卡 RTX 4090 ~12 h；速度命令 $v_x\in[0,1.5]$ m/s 等。
- **指标：** JFID/KFID（关节/关键点分布）、JDTW/KDTW、MELV；相对 SaW、PBRS、HumanMimic 及 **+AMP** 基线显著更优（例：JFID **0.931 vs PBRS+AMP 2.088**）。

## 核心摘录（面向 wiki 编译）

### 与 AMP / 生成先验路线对照

| 维度 | GMP（本文） | AMP | SD-AMP / MoRE 等 |
|------|------------|-----|------------------|
| 先验形态 | **冻结 CVAE 生成参考轨迹** | 在线对抗判别器 | 多判别器 / 门控 AMP |
| 监督粒度 | 关节角 + 关键点 **逐帧** | 标量 style reward | 判别器 + 任务奖励 |
| 训练稳定性 | 先验离线、策略单独 PPO | 策略–判别器共训 | 各异 |
| 典型平台 | NAVIAI | 多平台 | G1 等 |

## 对 wiki 的映射

- 沉淀实体页（AMP 专题 #06 深读）：[paper-amp-survey-06-natural_humanoid_robot_locomotion_wi.md](../../wiki/entities/paper-amp-survey-06-natural_humanoid_robot_locomotion_wi.md)
- 交叉补强：[AMP & HumanX](../../wiki/methods/amp-reward.md)、[Locomotion](../../wiki/tasks/locomotion.md)、[MoRE #08](../../wiki/entities/paper-amp-survey-08-more.md)（对抗多判别器对照）、[humanoid-amp-motion-prior-survey](../../wiki/overview/humanoid-amp-motion-prior-survey.md)
- 策展索引：[humanoid_amp_survey_06_natural_humanoid_robot_locomotion_with_generativ.md](humanoid_amp_survey_06_natural_humanoid_robot_locomotion_with_generativ.md)

## 参考来源（原始）

- arXiv:2503.09015 — 论文正文
- [wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md) — AMP 19 篇微信公众号编译导读
