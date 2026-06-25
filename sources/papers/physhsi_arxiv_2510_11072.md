# PhysHSI: Towards a Real-World Generalizable and Natural Humanoid-Scene Interaction System（arXiv:2510.11072）

> 来源归档（ingest）

- **标题：** PhysHSI: Towards a Real-World Generalizable and Natural Humanoid-Scene Interaction System
- **类型：** paper / humanoid / HSI / AMP / sim2real / perception
- **arXiv abs：** <https://arxiv.org/abs/2510.11072>
- **arXiv HTML：** <https://arxiv.org/html/2510.11072>
- **PDF：** <https://arxiv.org/pdf/2510.11072>
- **项目页：** <https://why618188.github.io/physhsi>
- **机构：** 上海人工智能实验室（Shanghai AI Lab）、香港科技大学（HKUST）
- **硬件：** Unitree G1（29 DoF）+ Livox Mid-360 LiDAR + Intel RealSense D455；机载 Jetson Orin NX 全栈部署
- **入库日期：** 2026-06-25
- **一句话说明：** 仿真 **AMP + 混合 RSI** 学搬箱/坐/躺/站起四类 HSI；真机 **粗–细物体定位**（LiDAR 里程计 + AprilTag）闭环，室内外高成功率与自然动作。

## 摘要级要点

- **问题：** 真实 HSI 需同时满足 **泛化场景、自然动作、可靠感知**；纯 RL 塑形繁重，纯 MoCap 跟踪难泛化且缺 sim2real。
- **数据：** AMASS/SAMP retarget → $M_{\text{Robo}}$，再 **人工标注接触关键帧** 规则推断物体轨迹，得含物体位姿的增强数据集 $M$。
- **AMP 训练：** 判别观测 $\mathbf{o}^{\mathcal{D}}_t$ **含物体位姿** $\mathbf{p}^{o_t}_{b_t}$，使判别器隐式感知任务阶段（接近/抓取/搬运/放置）；风格奖励 $r^S=-\log(1-\mathcal{D}(\cdot))$；任务+正则+风格加权。
- **混合 RSI：** 从参考随机相位 $\phi$ 初始化并随机化 $(\phi,1]$ 场景；部分 episode 从默认站姿 + 全随机场景启动，防过拟合演示布局。
- **非对称 actor-critic：** 部署可观测量进 actor；训练 critic 见速度等特权；抓取后物体出视野时 **mask 物体观测** + 域随机化对齐真机。
- **粗–细定位：** 远距用手动 LiDAR 粗位姿 + FAST-LIO 传播；近距 AprilTag 精定位；静态椅 vs 动态箱不同传播策略。
- **任务：** Carry Box、Sit Down、Lie Down、Stand Up；另可学 **风格化行走**（恐龙步、高抬腿等）。
- **仿真 benchmark（Table I）：** 相对无 AMP/无 RSI 基线，PhysHSI 在成功率与自然度 $S_{\text{human}}$ 上全面领先（详见原文）。

## 核心摘录（面向 wiki 编译）

### 与相邻 HSI / AMP 路线对照

| 维度 | PhysHSI（本文） | TeamHOI（#17） | SplitAdapter 下游 |
|------|----------------|----------------|-------------------|
| 智能体 | **单机器人 HSI** | 2–8 人形协作搬桌 | 冻结 PhysHSI 类搬箱策略 |
| AMP 数据 | **单人–物体 MoCap 增强** | 单人 + masked AMP | 负载适配 FiLM |
| 感知 | **LiDAR+相机粗细分层** | 仿真理想观测 | 力/惯量感知 |
| 真机 | G1 全 onboard | 物理仿真为主 | sim2real 适配 |

### 策展导读（AMP 专题 #15）

- 把 AMP 从 **locomotion 正则** 推到 **可部署的真实场景 HSI 系统**；物体进判别观测是长时程阶段感知的要点。
- 与 [DIMOS](../../wiki/entities/paper-dimos-human-scene-motion-synthesis.md) 任务重叠但 **物理人形 + AMP**，非 SMPL 运动学合成。

## 对 wiki 的映射

- 沉淀实体页：[PhysHSI（AMP #15）](../../wiki/entities/paper-amp-survey-15-physhsi.md)
- 交叉：[amp-reward](../../wiki/methods/amp-reward.md)、[loco-manipulation](../../wiki/tasks/loco-manipulation.md)、[sim2real](../../wiki/concepts/sim2real.md)、[unitree-g1](../../wiki/entities/unitree-g1.md)、[paper-splitadapter-load-aware-loco-manipulation](../../wiki/entities/paper-splitadapter-load-aware-loco-manipulation.md)、[TeamHOI #17](../../wiki/entities/paper-amp-survey-17-teamhoi.md)

## 参考来源（原始）

- arXiv:2510.11072
- [humanoid_amp_survey_15_physhsi_towards_a_real_world_generalizable_and_n.md](humanoid_amp_survey_15_physhsi_towards_a_real_world_generalizable_and_n.md)
- [wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)
