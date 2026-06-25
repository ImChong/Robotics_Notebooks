# HAML: Humanoid Adversarial Multi-Skill Learning via a Single Policy（MDPI Actuators 2026）

> 来源归档（ingest）

- **标题：** HAML: Humanoid Adversarial Multi-Skill Learning via a Single Policy
- **类型：** paper / humanoid multi-skill / conditional AMP / policy distillation / sim2real
- **期刊：** MDPI *Actuators* 2026, 15(4), 212
- **DOI：** <https://doi.org/10.3390/act15040212>
- **全文：** <https://www.mdpi.com/2076-0825/15/4/212>
- **项目页：** <https://vsislab.github.io/haml/>
- **机构：** 山东大学控制科学与工程学院
- **硬件：** Unitree **G1**（机载 100 Hz，延迟 15–25 ms）
- **入库日期：** 2026-06-25
- **一句话说明：** **两阶段可部署系统**：Stage I 用 **clip 级 one-hot 技能标签** 训练 **条件对抗多技能 teacher**（**错配 transition–label + condition-aware loss** 防条件坍缩）；Stage II **历史本体蒸馏** 为仅 onboard 观测的 student，单策略覆盖走、舞、挥手等多技能切换。

## 摘要级要点

- **问题：** 大规模 MoCap → 单策略多技能时，条件判别器常 **忽略技能 ID**（conditional collapse），导致「像人但不听命令」；真机又难用特权全局速度。
- **技能接口：** 每条 clip 自动赋 **粗粒度 one-hot 标签**（walk、dance、wave hello 等），可扩展库。
- **Stage I — Teacher：** PPO + **条件判别器** $D(s_{t-N+1:t}, c)$；注入 **(transition, 错误 label)** 错配对；**condition-aware** 辅助损失惩罚错误关联；短上下文 $N$ 步转移。
- **Stage II — Student：** 仅用 **堆叠历史本体** $\pi(a|o^{\mathrm{real}}_t, c)$ 模仿 teacher；减依赖全局状态估计。
- **工程指标：** G1 机载 **100 Hz**；延迟 **15–25 ms**；仿真与真机报告技能覆盖率、转移覆盖率、真实感与训练效率。
- **与 AMP 专题：** 多技能 AMP 核心是让先验 **理解当前技能条件**——走路/跑步/起身/跳跃的「像人」标准不同；HAML 用 **显式条件绑定** 防判别器只判真假不判技能。

## 对 wiki 的映射

- 沉淀实体页：[paper-amp-survey-12-haml.md](../../wiki/entities/paper-amp-survey-12-haml.md)
- 交叉：[AHC #11](../../wiki/entities/paper-adaptive-humanoid-control.md)、[MoRE #08](../../wiki/entities/paper-amp-survey-08-more.md)、[SD-AMP #10](../../wiki/entities/paper-unified-walk-run-recovery-sdamp.md)、[amp-reward.md](../../wiki/methods/amp-reward.md)
- 策展索引：[humanoid_amp_survey_12_haml_humanoid_adversarial_multi_skill_learning_v.md](humanoid_amp_survey_12_haml_humanoid_adversarial_multi_skill_learning_v.md)

## 参考来源（原始）

- MDPI Actuators 15(4):212 — 论文正文
- [wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md) — AMP 19 篇微信公众号编译导读
