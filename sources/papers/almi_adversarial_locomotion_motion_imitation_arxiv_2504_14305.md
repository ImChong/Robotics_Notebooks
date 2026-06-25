# Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning（arXiv:2504.14305）

> 来源归档（ingest）

- **标题：** Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning
- **类型：** paper / humanoid whole-body control / adversarial RL / loco-manipulation / dataset
- **arXiv abs：** <https://arxiv.org/abs/2504.14305>
- **arXiv HTML：** <https://arxiv.org/html/2504.14305v1>
- **PDF：** <https://arxiv.org/pdf/2504.14305>
- **项目页：** <https://almi-humanoid.github.io>
- **会议：** NeurIPS 2025
- **机构：** 中国电信人工智能研究院（TeleAI）、上海科技大学、中科大、西北工业大学、清华大学
- **硬件：** Unitree **H1-2**（主实验）；Unitree **G1**（泛化对比）
- **入库日期：** 2026-06-25
- **一句话说明：** **ALMI** 将人形控制拆为 **下半身 locomotion 策略 $\pi^l$** 与 **上半身 motion imitation 策略 $\pi^u$**，通过 **上下半身互相对抗的迭代 RL**（命令空间内环优化 + 双臂课程）达到走跑稳健与上肢跟踪精度的纳什均衡，并发布 **ALMI-X** 语言标注全身数据集。

## 摘要级要点

- **问题：** 全身一体 mimic 奖励复杂、易牺牲平衡；上下半身角色不同却被同一策略硬学。
- **对抗框架：** 学 $\pi^l$ 时上半身作扰动（max-min）；学 $\pi^u$ 时下半身速度命令作扰动；理论保证 $\epsilon$-近似纳什均衡；实践用 **固定一方、采样对抗命令/动作** 降计算。
- **下半身 $\pi^l$：** 12 DoF 腿；速度命令 + 步态相位；Table 1 丰富 locomotion 奖励；**双臂课程**：按存活时长排序 AMASS 动作难度 + motion scale $\alpha_s$ 渐进扰动。
- **上半身 $\pi^u$：** 9 DoF（肩肘 + 腰 yaw）；跟踪参考关节 $\bm{g}^u$；对抗速度课程随跟踪误差加大。
- **平台：** H1-2 共 27 DoF，策略控 21 DoF（腕除外）；Isaac Gym 4096 并行；三轮对抗迭代 ~17 h。
- **ALMI-X：** >80k 轨迹；语言模板「${movement mode} ${direction} ${velocity level} and ${motion}」；MuJoCo 采集；初步 Transformer 全身 foundation model。
- **主结果（CMU MoCap，Hard）：** ALMI 存活率 **0.9723** vs Exbody **0.8778**；上肢 JPE **0.2116 m** vs ALMI(whole) **0.7022 m**；相对 OmniH2O / Exbody2 在 G1 上显著更稳。

## 核心摘录（面向 wiki 编译）

### 与 AMP 专题语境

- **非典型 AMP 论文**，但体现「运动先验必须理解身体结构」：下半身 **locomotion prior**（速度/平衡）与上半身 **mimic prior**（姿态跟踪）应 **分部位、可对抗地共训**，而非单一判别器笼统约束全身。
- 与 [MoRE #08](../../wiki/entities/paper-amp-survey-08-more.md)（深度 + 多判别器步态）、[Hiking #09](../../wiki/entities/paper-hiking-in-the-wild.md)（感知落脚）同属 **02 人形走跑** 段互补视角。

## 对 wiki 的映射

- 沉淀实体页（AMP 专题 #07 深读）：[paper-amp-survey-07-adversarial_locomotion_and_motion_im.md](../../wiki/entities/paper-amp-survey-07-adversarial_locomotion_and_motion_im.md)
- 交叉补强：[Locomotion](../../wiki/tasks/locomotion.md)、[MoRE #08](../../wiki/entities/paper-amp-survey-08-more.md)、[humanoid-amp-motion-prior-survey](../../wiki/overview/humanoid-amp-motion-prior-survey.md)
- 策展索引：[humanoid_amp_survey_07_adversarial_locomotion_and_motion_imitation_for.md](humanoid_amp_survey_07_adversarial_locomotion_and_motion_imitation_for.md)

## 参考来源（原始）

- arXiv:2504.14305 — 论文正文
- [wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md) — AMP 19 篇微信公众号编译导读
