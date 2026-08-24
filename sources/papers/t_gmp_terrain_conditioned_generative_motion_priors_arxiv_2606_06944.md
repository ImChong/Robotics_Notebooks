# T-GMP: Terrain-conditioned Generative Motion Priors for Versatile and Natural Humanoid Locomotion（arXiv:2606.06944）

> 来源归档（ingest · 2026-08-24）

- **标题：** T-GMP: Terrain-conditioned Generative Motion Priors for Versatile and Natural Humanoid Locomotion
- **类型：** paper / humanoid locomotion / generative motion prior / CVAE / terrain-conditioned AMP / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2606.06944>
- **arXiv HTML：** <https://arxiv.org/html/2606.06944v1>
- **PDF：** <https://arxiv.org/pdf/2606.06944>
- **项目页：** <https://t-gmp.github.io>
- **作者：** Junhong Guo、Hao Hu、Chen Chen、Haoxuan Han、Linao Gong、Xin Yang、Zhicheng He、Yao Su、Fenghua He（* 前两位等贡献；† Fenghua He 通讯作者）
- **机构：** 哈尔滨工业大学（HIT）；乐聚机器人（Leju Robotics）
- **年份：** 2026（arXiv preprint）
- **硬件：** 乐聚 **Kuavo** 全尺寸人形（1.66 m / ~55 kg / **28 DoF**）；头载 **Mid-360 LiDAR** 做点云感知与高程图重建
- **仿真：** Isaac Lab；PPO；单卡 **RTX 4090**；**2048** 并行环境
- **入库日期：** 2026-08-24
- **一句话说明：** 用 **地形条件 CVAE（T-GMP）** 从少量专家状态–高程示范学习潜运动流形，再以 **地形条件判别器（AMP 扩展）** 与 **Foothold Penalty** 联合训练统一感知 locomotion 策略，在 **8 类地形** 上同时提升穿越成功率与全身运动自然性。

## 摘要级要点

- **问题：** 固定、与地形无关的运动先验在复杂地形上会惩罚为稳定性必需的偏离，导致步态僵硬或失败；纯手工奖励难以同时优化任务与自然性。
- **T-GMP 模块：** 条件 **β-VAE / CVAE**；局部高程图 $h_t$ 经两层 CNN 得地形嵌入，解码器生成专家状态序列 $\hat{s}_{t:t+T}$，学习 **地形条件潜运动流形**。
- **统一对抗训练：** 扩展 AMP 为 **terrain-conditioned discriminator** $D(s_k,s_{k+1}|h_t^{\mathrm{emb,d}})$，在线用 T-GMP 解码器合成专家转移；风格奖励 $r_{\mathrm{amp}}=\max[0,1-(D-1)^2/4]$。
- **Foothold Penalty：** 趾端距离 $d_{\mathrm{toe}}$ 抑制踢台阶；足底距离 $d_{\mathrm{sole}}$ 抑制边缘支撑与打滑（下楼尤甚）。
- **专家数据：** 特权策略在楼梯/坡/梁等难地形采集 + 平地/沟/台 **MoCap（GMR retarget）**；合计约 **29.6 min（88.8k 帧）** 覆盖 **8 地形**。
- **感知：** 机器人前方 **$9\times7$ height scan**（局部 patch **0.8 m × 0.6 m**）；真机由 LiDAR 点云经 elevation mapping + 最近邻填 NaN 得高程图（非深度相机主路径）。
- **评测：** 相对 Baseline RL [Rudin et al.]、w/o Condition、w/o CVAE、w/o Foothold；**700-step** rollout；t-SNE 关节轨迹聚类 + 成功率表 + 力矩/加速度平滑度。
- **量化（论文）：** 全身平均关节力矩 **178.97 N·m**、加速度 **434.72 rad/s²**，相对 Baseline RL 降 **30.01% / 38.20%**；八地形单次穿越成功率全面领先（例：Gap **98.83%** vs 92.97%；Beam **96.88%** vs 79.69%）。
- **真机：** 平地摆臂、楼梯/坡降质心、沟/窄梁伸臂平衡；台阶高 **0.13 m**、踏面 **0.28 m**；沟宽 **0.4 m**、梁宽 **0.35 m**（脚间距 0.3 m）。

## 核心摘录（面向 wiki 编译）

### 三阶段管线

1. **数据采集：** 特权专家策略 + MoCap/GMR → $\mathcal{D}=\mathcal{D}_{\mathrm{priv}}\cup\mathcal{D}_{\mathrm{mocap}}$，每轨迹 $d=\{(s_t,h_t)\}$。
2. **离线 T-GMP：** 条件 CVAE 最小化重建 + $\beta$-KL；部署时仅用当前帧 $h_t$ 条件解码（缩短 horizon $T$ 抑制误差累积）。
3. **在线 RL：** Actor 拼接 CNN 地形嵌入与 5 帧堆叠本体观测；Critic 用特权项；总奖励 $r=w_1 r_{\mathrm{task}}+w_2 r_{\mathrm{reg}}+w_3 r_{\mathrm{amp}}+w_4 r_{\mathrm{foothold}}$。

### 与相关路线对照

| 维度 | T-GMP（本文） | GMP（浙大 arXiv:2503.09015） | TRAMP（SJTU RA-L 2026） | MoRE（TeleHuman G1） |
|------|--------------|------------------------------|-------------------------|---------------------|
| 先验形态 | **地形条件 CVAE 流形 + 地形条件 AMP** | 冻结 CVAE 生成参考轨迹 + 稠密 guidance | 平地/楼梯双示范 **判别式** terrain-related AMP | 多判别器 + 两阶段深度 MoE |
| 感知 | LiDAR 高程图 height scan | 未强调复杂地形感知 | 低成本深度 + 层次特征 | 深度图 |
| 平台 | **Kuavo（乐聚）** | NAVIAI | 未写明型号双足 | Unitree G1 |
| 全身协调 | 摆臂/降 CoM/伸臂平衡 | 走跑自然性 | 强调单阶段轻量 | 多步态切换 |

## 对 wiki 的映射

- 升格实体页：[paper-motion-cerebellum-t-gmp](../../wiki/entities/paper-motion-cerebellum-t-gmp.md)
- 运动小脑策展索引：[motion_cerebellum_survey_02_t_gmp.md](motion_cerebellum_survey_02_t_gmp.md)
- 交叉：[AMP 方法页](../../wiki/methods/amp-reward.md)、[GMP #06](../../wiki/entities/paper-amp-survey-06-natural_humanoid_robot_locomotion_wi.md)、[TRAMP](../../wiki/entities/paper-tramp-vision-assisted-bipedal-locomotion.md)、[MoRE #08](../../wiki/entities/paper-amp-survey-08-more.md)、[乐聚机器人](../../wiki/entities/leju-robotics.md)

## 参考来源（原始）

- arXiv:2606.06944 — 论文正文
- 项目页：https://t-gmp.github.io（入库日核查见 [`sources/sites/t-gmp.md`](../sites/t-gmp.md)）
- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md) — 运动小脑 64 篇策展导读
