# DiT4DiT（arXiv:2603.10448）

> 来源归档（ingest）

- **标题：** DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control
- **缩写：** **DiT4DiT**
- **类型：** paper / video-action-model / world-action-model / manipulation
- **arXiv：** <https://arxiv.org/abs/2603.10448>（HTML：<https://arxiv.org/html/2603.10448v1>）
- **PDF：** <https://arxiv.org/pdf/2603.10448>
- **项目页：** <https://dit4dit.github.io/>
- **代码：** <https://github.com/Mondo-Robotics/DiT4DiT>
- **作者：** Teli Ma†, Jia Zheng†, Zifan Wang, Chunli Jiang, Andy Cui, Junwei Liang*, Shuo Yang*（† 共同一作；* 通讯 / 共同指导）
- **机构：** Mondo Robotics（摩多机器人）；香港科技大学（广州）；香港科技大学
- **入库日期：** 2026-06-10
- **一句话说明：** 端到端 **Video-Action Model（VAM）**：**Cosmos-Predict2.5-2B** Video DiT 与 GR00T-N1 系 Action DiT **联合 flow matching**；在固定 flow 步 **hook 视频去噪隐状态** 条件动作预测，提出 **三时间步**（$\tau_v$ / $\tau_f$ / $\tau_a$）解耦训练；LIBERO **98.6%**、RoboCasa-GR1 **50.8%**（论文）/ 项目页 **56.7%**，G1 真机八项桌面 + 三项全身 loco-manip；相对 VLA 基线报告 **>10× 样本效率** 与 **7× 收敛**；开源代码与权重。

## 核心论文摘录（MVP）

### 1) 动机与总贡献（Abstract / §1）

- **链接：** <https://arxiv.org/abs/2603.10448>
- **核心贡献：** VLA 继承 **静态图文** 表征，物理动力学主要靠小规模机器人数据补；**视频 生成模型** 天然编码时序与隐式物理。DiT4DiT 用 **级联双 DiT**：Video DiT 预测未来动力学，**不去噪完整未来帧**，而从生成过程 **抽取中间隐特征** 条件 Action DiT；**dual flow-matching** 与 **解耦时间步** 实现端到端联合训练。仿真 SOTA：**LIBERO 98.6%**、**RoboCasa-GR1 50.8%**；G1 真机优于 GR00T-N1.5 等同量级基线，并展示 **零样本** 物体/类别/数量泛化。
- **对 wiki 的映射：**
  - [DiT4DiT 论文实体](../../wiki/entities/paper-dit4dit-video-action-model.md)
  - [World Action Models](../../wiki/concepts/world-action-models.md)
  - [mimic-video（VAM）](../../wiki/methods/mimic-video.md)

### 2) 视频生成作为 scaling proxy（§3）

- **链接：** <https://arxiv.org/html/2603.10448v1>
- **核心贡献：** 在 RoboCasa-GR1 **24 任务**上对比三种自监督预训练：**Grounding**、**FLARE 式 VLM 潜对齐**、**视频生成**；冻结骨干仅训 action expert 时，**视频生成目标** 收敛最快（**~7×**）、样本效率最高（**~10×**），验证「视频生成可作机器人策略学习的 scaling proxy」。
- **对 wiki 的映射：**
  - [VLA](../../wiki/methods/vla.md) — 静态先验局限
  - [Generative World Models](../../wiki/methods/generative-world-models.md)

### 3) 双 DiT 架构与三时间步联合训练（§4.2–4.3）

- **核心贡献：** **Video DiT** 自 Cosmos-Predict2.5-2B（因果 VAE + flow DiT + Cosmos-Reason1 语言）；**Action DiT** 自 GR00T-N1 改编，AdaLN + cross-attn 读 $\mathbf{h}_t^{\tau_f}$。**三时间步：** $\tau_v \sim \mathcal{U}[0,1]$ 学全去噪轨迹；**固定 $\tau_f$** 稳定特征提取；$\tau_a$ 用 **Beta 分布** 偏重关键控制阶段。联合损失 $\mathcal{L}^{\text{total}} = \mathcal{L}_{\text{action}} + \lambda \mathcal{L}_{\text{video}}$，VAE 与文本编码器冻结。
- **对 wiki 的映射：**
  - [mimic-video](../../wiki/methods/mimic-video.md) — 冻结骨干 vs **联合微调** 对照
  - [MotionWAM](../../wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md) — 同团队将双 DiT + flow matching **推进到人形实时全身 WAM**

### 4) 仿真与 G1 真机评测（§5 / 项目页）

- **仿真：** LIBERO 四套件平均 **98.6%**；RoboCasa-GR1 24 任务 **50.8%**（摘要）/ 项目页 **56.7%** vs GR00T-N1.6 **47.8%**。
- **真机（G1 + 单目 egocentric）：** 八项桌面（pick-place、arrange flower、stack cups 等）+ 三项 **全身 loco-manip**（shelf organization、relocate chair、assembly work）；可选 **+SONIC** 全身控制或 **decoupled WBC**。
- **效率（A100）：** **6 Hz** 部署频率；**2.2B** 可训参数；相对 Cosmos Policy **0.7 Hz**、mimic-video **1.9 Hz** 更快。
- **对 wiki 的映射：**
  - [Unitree G1](../../wiki/entities/unitree-g1.md)
  - [SONIC](../../wiki/methods/sonic-motion-tracking.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)

### 5) 与 MotionWAM 谱系（后续工作）

- **核心贡献：** [MotionWAM](https://arxiv.org/abs/2606.09215)（同作者团队）明确 **受 DiT4DiT 启发**，将双 DiT + 中间隐状态条件化 **扩展到实时人形 loco-manipulation**：单次 Video DiT 前向（$\tau_f \approx 1$）、**SONIC 统一全身 motion token**、三阶段 egocentric 视频预训练；九项 G1 任务平均 **76.1%** vs GR00T-N1.7 **43.9%**。
- **对 wiki 的映射：**
  - [MotionWAM 论文实体](../../wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md)

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-dit4dit-video-action-model.md`](../../wiki/entities/paper-dit4dit-video-action-model.md)
- 互链参考：[MotionWAM](../../wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md)、[mimic-video](../../wiki/methods/mimic-video.md)、[World Action Models](../../wiki/concepts/world-action-models.md)、[VLA](../../wiki/methods/vla.md)
