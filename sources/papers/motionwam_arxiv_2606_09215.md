# MotionWAM（arXiv:2606.09215）

> 来源归档（ingest）

- **标题：** MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation
- **缩写：** **MotionWAM**
- **类型：** paper / world-action-model / humanoid loco-manipulation
- **arXiv：** <https://arxiv.org/abs/2606.09215>（HTML：<https://arxiv.org/html/2606.09215v1>）
- **PDF：** <https://arxiv.org/pdf/2606.09215>
- **作者：** Jia Zheng†, Teli Ma†, Yudong Fan, Zifan Wang, Shuo Yang*, Junwei Liang*（† 共同一作；* 通讯 / 共同指导）
- **机构：** Mondo Robotics（摩多机器人）；香港科技大学（广州）；香港科技大学
- **入库日期：** 2026-06-10
- **开源状态（2026-07-20 再核）：** arXiv abs / HTML **无** 项目页或 GitHub Code 链；暂无官方可运行仓。
- **一句话说明：** 面向 **实时闭环人形 loco-manipulation** 的 **World Action Model**：Video DiT 与 Motion DiT **双骨干** 经 **单次前向去噪隐状态** 耦合；用 **统一全身 motion token**（基于 SONIC 潜空间）替代上下身分层接口，三阶段从 egocentric 视频预训练到跨具身动作后训练再到全身遥操作微调；在 **宇树 G1 + 双 ALOHA2 夹爪 + 头部 RealSense D435i** 九项真机任务上相对同演示微调的 VLA 基线 **整体成功率 +32% 绝对值**。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / §1）

- **链接：** <https://arxiv.org/abs/2606.09215>
- **核心贡献：** 现有 WAM 在桌面臂上有效，但 **高维视频–动作迭代去噪** 难以满足人形 **实时闭环**；同时主流 **上身精细关节 + 下身粗粒度基座命令** 的分层范式使上下身 **动作空间不一致**，腿只能 **保平衡** 而无法 **任务驱动落脚**（踩踏板、踢球等）。MotionWAM 用 **单目 egocentric 相机** 驱动 **统一全身 motion latent**，在 **一次 Video DiT 前向**（固定 flow 步 $\tau_f \approx 1$ 的「想象」隐状态）条件下 Motion DiT 预测 **locomotion / 躯干 / 身高 / 足端交互 / 手部操作** 的联合 token；报告为 **首个实时闭环、端到端 WAM 驱动的人形全身 loco-manipulation**（含任务驱动足部行为）。
- **对 wiki 的映射：**
  - [MotionWAM 论文实体](../../wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md)
  - [World Action Models](../../wiki/concepts/world-action-models.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)

### 2) 双 DiT 架构与实时推理（§3.2）

- **链接：** <https://arxiv.org/html/2606.09215v1>
- **核心贡献：** **Video DiT** 自 **Cosmos-Predict2.5-2B** 初始化（因果时空 VAE + flow-matching DiT，Cosmos-Reason1 语言嵌入）；在某一 transformer block 上 **hook 固定 flow 步** 的隐状态 $\mathbf{h}_t^{\tau_f}$，**不做完整未来帧去噪**——单次前向即得「场景即将如何变化」的条件。 **Motion DiT** 经交错 self/cross-attention 消费 $\mathbf{h}_t^{\tau_f}$、本体 $p_t$ 与噪声 motion token，输出 flow 速度场；多具身阶段用 **per-embodiment projector** 包裹共享 trunk，部署时换 **Unitree G1 projector**。相对 **Cosmos Policy** 等需迭代去噪未来的 WAM，A100 上 **4.9 Hz vs 0.7 Hz**（约 7×），与 GR00T-N1.7（6.5 Hz）、Qwen3DiT（9.0 Hz）同量级。
- **对 wiki 的映射：**
  - [Generative World Models](../../wiki/methods/generative-world-models.md)
  - [VLA](../../wiki/methods/vla.md)（对照：静态 VLM 先验 vs 视频动力学先验）

### 3) 统一全身 motion latent 与 SONIC 解码（§3.1 / Eq. 6）

- **核心贡献：** 全身 motion latent $\mathbf{m}_t = (\mathbf{m}_t^{\text{cont}}, \mathbf{k}_t)$：**离散 SONIC token** $\mathbf{k}_t$（FSQ：2 token × 32 level → 64 维）概括 locomotion / 躯干 / 身高 / 足端意图；**连续通道** $\mathbf{m}_t^{\text{cont}}$ 驱动双手夹爪/灵巧手。Stage 3 将 SONIC 索引 $k_t$ 作为 **连续标量槽** 回归，推理时 **最近邻取整** 再经 **SONIC whole-body controller** 解码为关节命令——与 LEGS / 分层 VLA 的 **18-D 上身+基座命令** 接口形成对照。
- **对 wiki 的映射：**
  - [SONIC](../../wiki/methods/sonic-motion-tracking.md)
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md)
  - [LEGS](../../wiki/entities/paper-legs-embodied-gaussian-splatting-vla.md)（同 G1 + SONIC 栈，数据路线不同）

### 4) 三阶段训练与真机评测（§3.3 / §4）

- **Stage 1：** 仅训 Video DiT，$\sim$**2,136 小时** egocentric 人/人形机器人视频（无动作标签），学 **第一人称视觉动力学**。
- **Stage 2：** 接入 Motion DiT，异构 **Unitree G1** 数据（不同末端与标注格式）联合 **$\mathcal{L}_{\text{motion}} + \mathcal{L}_{\text{video}}$**；保留视频目标防动力学先验被覆盖。
- **Stage 3：** **PICO VR 三点追踪** 全身遥操作（SMPL-24 → G1 重定向），每任务 **200 episodes**；端到端微调，输出统一全身 token。
- **硬件：** G1 + **双 ALOHA2 夹爪** + 头部 **Intel RealSense D435i RGB**；策略以 **WebSocket server** 跑在 **RTX 4090**，机载控制器闭环查询。
- **九项真机任务**（各 20 次）：强调腰控、身高调节、蹲行、**任务驱动足端交互**、身–手协调（如 Kick Soccer、Load Cart、Do Laundry、Wipe Board 等）；**平均成功率 76.1%** vs 最强 VLA 基线 GR00T-N1.7 **43.9%**（+32.2% 绝对）；消融：去 Stage 1 / Stage 2 分别 **−11% / −28%** 平均成功率。
- **对 wiki 的映射：**
  - [Unitree G1](../../wiki/entities/unitree-g1.md)
  - [Teleoperation](../../wiki/tasks/teleoperation.md)
  - [Motion Retargeting Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md)

### 5) 局限与谱系（§6 / Related Work）

- **局限：** Stage 3 仅在 **G1** 验证；未见严格 **新物体 OOD** 评测；**单目 egocentric** 在物体出视野或头摄视角漂移时失稳。
- **谱系：** 方法直接继承 **[DiT4DiT](../../wiki/entities/paper-dit4dit-video-action-model.md)**（arXiv:2603.10448）双 DiT + flow matching 接口；与 **Cosmos Policy**、**GR00T-N1.7**、**π₀.₅** 等同演示微调 VLA 基线对比。
- **对 wiki 的映射：**
  - [World Action Models](../../wiki/concepts/world-action-models.md) — Joint 族 **人形实时** 实例
  - [DiT4DiT 论文实体](../../wiki/entities/paper-dit4dit-video-action-model.md)

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md`](../../wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md)
- 互链参考：[World Action Models](../../wiki/concepts/world-action-models.md)、[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)、[SONIC](../../wiki/methods/sonic-motion-tracking.md)、[VLA](../../wiki/methods/vla.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)
