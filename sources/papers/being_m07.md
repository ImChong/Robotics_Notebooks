# Being-M0.7

> 来源归档（ingest）

- **标题：** Being-M0.7: A Latent World-Action Model for Humanoid Robots
- **缩写：** **Being-M0.7**
- **类型：** paper / world-action-model / humanoid / loco-manipulation / latent-world-model
- **项目页：** <https://research.beingbeyond.com/being-m07>
- **PDF：** <https://research.beingbeyond.com/being-m07/being-m07.pdf>
- **作者：** Yue Junpeng, Li Boyuan, Wang Yuxuan, Wang Zepeng, Fu Yuhui, Xie Feiyang, Zhang Yu, Zhang Jing, Wang Jiangxing, Lu Zongqing 等（BeingBeyond Team）
- **机构：** 超越智能（BeingBeyond）
- **入库日期：** 2026-07-15
- **一句话说明：** 面向 **人形 loco-manipulation** 的 **潜空间 World–Action Model**：在 **>1 万小时** 人中心混合模态数据（egocentric 视频 / 视频–动作对 / 纯动作）上，用 **video-motion MoT + flow matching** 学 **DINO 视觉 latent + 紧凑全身 motion** 联合先验，再以 **future-conditioned action expert** 在少量 **宇树 G1** VR 全身遥操作轨迹上接地为可执行命令；推理上 **低频刷新世界计划、高频复用 prior KV cache** 输出 action chunk。

## 核心论文摘录（MVP）

### 1) 问题：人形演示贵、像素未来预测贵、上下身常割裂

- **链接：** <https://research.beingbeyond.com/being-m07/being-m07.pdf> §1
- **摘录要点：** 可扩展人形 loco-manipulation 需要同步 egocentric 视频、本体与全身可执行命令，遥操作难、安全与硬件约束使机器人示范难规模化；**像素级未来预测** 计算贵且容量易耗在与控制弱相关的外观上，快速 ego 运动还会引入视角抖动噪声；许多管线仍 **上身操纵与下身运动分开建模**，削弱全身协调。
- **对 wiki 的映射：**
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)
  - [World Action Models](../../wiki/concepts/world-action-models.md)
  - [Teleoperation](../../wiki/tasks/teleoperation.md)

### 2) 分解：prior latent WAM + future-conditioned action expert（式 1）

- **链接：** <https://research.beingbeyond.com/being-m07> Overview / PDF §3.1
- **摘录要点：** 将问题写为 **先验世界–动作模型** $P(Z_Q, M_Q \mid Z_C, M_C, I)$ 与 **动作专家** $P(a_q \mid Z_Q, M_Q, O_q, I)$ 的组合；$Z=E(V)$ 为 **冻结 DINO** 视觉 latent，$M$ 为 **粗粒度全身 motion 计划**（非直接可执行命令）；动作专家 **单向** 读取 prior 多层隐状态 + 当前观测 $O_q$（egocentric 图、本体、执行进度），输出 **action chunk**；分离 **低频世界规划** 与 **高频闭环控制**。
- **对 wiki 的映射：**
  - [Being-M0.7 论文实体](../../wiki/entities/paper-being-m07-humanoid-latent-wam.md)
  - [Being-H0.7 方法页](../../wiki/methods/being-h07.md) — 同族「潜空间 WAM」，H 系偏操作 VLA 式先验，M0.7 显式 **video-motion MoT + 人形全身接地**
  - [Action Chunking](../../wiki/methods/action-chunking.md)

### 3) 数据：三流混合预训练 + 统一 head-root 紧凑 motion

- **链接：** PDF §3.2 / Figure 3
- **摘录要点：** 原始 **>10,000 h** 混合模态（过滤前），三监督流：**配对 video–motion**、**仅视频**、**仅动作**；视觉一律 **DINO latent** 生成/预测，不做像素重建；动作侧将异构人动作规范为 **head-root** 坐标，保留 **头、双手、双脚** 紧凑表示，并可经 **正运动学** 从机器人轨迹重建，桥接人–机形态差；来源含 Ego4D、Xperience、Nymeria、Bones-SEED、SnapMoGen、HumanML3D、LAFAN1 及 Being-H0.5 / Being-M0.5 部分内部数据等。
- **对 wiki 的映射：**
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — DINO latent 与语义状态预测
  - [Motion Retargeting Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md)

### 4) 架构：video-motion MoT + flow matching + 几何辅助损失

- **链接：** PDF §3.3–3.4
- **摘录要点：** Prior 为 **Mixture-of-Transformers**：视觉/动作模态 **独立投影、LN、FFN**，**共享跨模态注意力**；指令经 **冻结 DistilBERT** 条件化；未来区间用 **chunk 并行去噪**（非因果 target 交互）；预训练 **flow matching**，配对样本同时监督 $L_V + L_M + L_{\mathrm{geom}}$，单模态样本只激活对应边际损失；Post-training 在机器人轨迹上继续训 prior 的 visual/motion flow，并加 **action expert flow loss** $L_A$，权重 $\lambda_a$。
- **对 wiki 的映射：**
  - [World Action Models](../../wiki/concepts/world-action-models.md) — **Cascaded** 族：先验 latent video-motion → action expert
  - [MotionWAM](../../wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md) — 对照：MotionWAM 走 **Joint 双 DiT + 单次 Video 前向隐状态**；Being-M0.7 强调 **人数据先验 + 后接轻量 expert**，不直接端到端闭环人形策略

### 5) 机器人后训练：VR 全身遥操作 + SONIC 跟踪

- **链接：** PDF §4.1.2 / Figure 5
- **摘录要点：** **宇树 G1** + 双 **Linker Hand O6** + 头部 **RealSense D435i**；采集：**PICO VR** 头显、手柄与踝部 tracker → **XRoboToolkit** 估计 SMPL → **SONIC** 转 **29-DoF** 全身命令（50 Hz）；同步记录 egocentric RGB、本体、motion command 与 robot motion 表示；推理在 **RTX 4090** 工作站，ZMQ 与机载低层闭环。
- **对 wiki 的映射：**
  - [Unitree G1](../../wiki/entities/unitree-g1.md)
  - [SONIC Motion Tracking](../../wiki/methods/sonic-motion-tracking.md)
  - [Teleoperation](../../wiki/tasks/teleoperation.md)

### 6) 真机实验：四项全身任务 vs GR00T-N1.6 / Ψ0

- **链接：** PDF §4.2 / Table 1
- **摘录要点：** 四任务：**镜面推理抓玩偶**、**网兜捞鱼**、**桌面整理**、**避障端篮**；定量：**Mirror Near/Far + Fish** 共 15 次，Being-M0.7 **7/15** vs GR00T-N1.6 **2/15**、Ψ0 **3/15**；Mirror 总体 **4/10** vs 基线各 **1/10**；Fish（仅操纵段）**3/5** vs **1/5** / **2/5**。作者称 Being-M0.7 为 **首个面向人形控制的 latent WAM**（technical report, 2026-07-14）。
- **对 wiki 的映射：**
  - [Being-M0.7 论文实体](../../wiki/entities/paper-being-m07-humanoid-latent-wam.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-being-m07-humanoid-latent-wam.md`](../../wiki/entities/paper-being-m07-humanoid-latent-wam.md)
- 互链参考：[World Action Models](../../wiki/concepts/world-action-models.md)、[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)、[Being-H0.7](../../wiki/methods/being-h07.md)、[MotionWAM](../../wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md)、[SONIC](../../wiki/methods/sonic-motion-tracking.md)、[Teleoperation](../../wiki/tasks/teleoperation.md)
