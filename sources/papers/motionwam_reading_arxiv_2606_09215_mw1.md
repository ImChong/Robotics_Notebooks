# MotionWAM 论文通读（arXiv:2606.09215）

> 来源归档（原文通读 · 支撑 `wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md`）

- **标题：** MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation
- **作者：** Jia Zheng, Teli Ma, Yudong Fan, Zifan Wang, Shuo Yang, Junwei Liang（Mondo Robotics / HKUST-GZ / HKUST）
- **类型：** paper / world-action-model / humanoid loco-manipulation
- **原文：** <https://arxiv.org/abs/2606.09215>（2026-11 通过 arXiv PDF v1 获取全文，共 12 页）
- **一句话说明：** 用 Video DiT 单次前向（固定 flow 步 τ_f≈1）的隐状态条件 Motion DiT，在 SONIC 统一全身 motion token 空间预测动作，三阶段训练，在 Unitree G1 九项真机任务上平均 76.1%。
- **核对范围：** 通读摘要、§1 引言、§2 相关工作、§3 方法（式 1–6）、§4 实验（Fig.5、表 1、表 2）、§5 结论、§6 局限、参考文献列表，以及附录 A（任务提示词）、B（遥操作硬件与 VR 管线）、C（表 4 超参）、D（基线训练配置）、E（表 5 各阶段数据构成）、F（失败案例）；图 5 的逐任务柱状数值仅取正文给出的文字数字，未逐柱核对。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题（§1）：** 迭代去噪高维视频-动作 latent 太慢；分层范式上身关节目标 + 下身粗粒度基座命令，动作空间不一致，腿被限制为保持平衡。
- **形式化（§3.1，式 1）：** o_{t+1}~p_v(·|o_t,l)，m_t~p_a(·|o_t,p_t,H(o^{τv}_{t+1}))；m_t=(m_cont,k_t)，k_t 为 SONIC 的 FSQ token（2 个 token、每个 32 级，64 维离散向量），m_cont 为夹爪/灵巧手连续通道。
- **架构（§3.2，式 2）：** Video DiT 自 Cosmos-Predict2.5-2B 初始化（因果时空 VAE + flow matching DiT，Cosmos-Reason1 语言嵌入）；在单个 transformer block 装 forward hook，取 τ_f≈1（纯噪声端）处激活，一次前向、不做未来帧去噪；Motion DiT 以交错 self/cross attention 消费该隐状态、本体 p_t 和噪声 motion token；多具身阶段用 per-embodiment 投影器包裹共享主干，部署用 G1 投影器。
- **训练（§3.3，式 3–6）：** VAE 与文本编码器全程冻结；Stage 1 仅训 Video DiT（约 2,136 h 第一视角人/人形视频）；Stage 2 联合 L_motion+L_video（视频损失作表征正则）；Stage 3 每任务 200 episodes，k̃_t 作连续标量槽回归，推理时 round 取整再由 SONIC 解码。
- **实验设置（§4.1）：** G1 + 双 ALOHA2 夹爪 + 头部 D435i RGB；PICO VR 三点追踪经 SMPL 重定向；策略以 WebSocket server 跑在单张 RTX 4090；基线：DP、ACT、π0.5、GR00T-N1.7、Qwen3DiT（Qwen3-VL 2B + 同款 Motion DiT，参数匹配）；每任务 20 次。
- **主结果（§4.2）：** 平均成功率由最强基线 GR00T-N1.7 的 43.9% 升至 76.1%；Kick Soccer/Load Cart/Retrieve Item 各 +40%，Wipe Board +45%，Do Laundry +30%；Qwen3DiT 在 locomotion-heavy 任务近乎为零，π0.5 总体低于 20%。
- **消融（§4.3，表 1，五个子任务）：** 无 Stage 2 平均 42.0、无 Stage 1 平均 59.0、完整 70.0，即分别 −28 / −11 个百分点。
- **速度（§4.4，表 2，A100）：** GR00T-N1.7 1.6B 6.5 Hz；Qwen3DiT 2.3B 9.0 Hz；Cosmos Policy 2.0B 0.7 Hz；MotionWAM 2.5B 4.9 Hz（频率为整块 action chunk 输出速率）。
- **局限（§6、附录 F）：** 仅在 G1 验证；无受控新物体泛化评测；单目相机在物体出视野或头部视角漂移时失去视觉定位。
- **附录细节：** Motion DiT 为 DiT-B（沿用 DiT4DiT），动作维 66、状态维 64，推理 4 步；Stage 1/2/3 分别用 128/32/8 GPU，训练 10 万/5 万/1.5 万步；Stage 1 数据按人 30%、G1 类人形 50%、其他真机 20% 加权；Stage 3 以 50 Hz 记录 LeRobot 格式；上身为 7-DoF 双臂 + 夹爪共 16-DoF 接口，SONIC 输出 29-DoF 关节角。

## 对 wiki 的映射

- [`wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md`](../../wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md)
