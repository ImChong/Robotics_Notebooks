# Diffusion Policy 论文通读（操作策略学习视角）

> 来源归档（原文通读 · 支撑 `wiki/tasks/manipulation.md`）

- **标题：** Diffusion Policy: Visuomotor Policy Learning via Action Diffusion（arXiv v5，2024-03）
- **作者：** Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song
- **类型：** paper / manipulation / imitation-learning
- **原文：** <https://arxiv.org/abs/2303.04137>（2026-11 通过 arXiv PDF 获取全文并转文本阅读）
- **关联归档：** [diffusion_policy_arxiv_2303_04137.md](diffusion_policy_arxiv_2303_04137.md)（ingest 入库档）
- **一句话说明：** 把视觉运动策略写成「以观测为条件的动作 DDPM」，在 4 个基准 15 个任务上平均提升 46.9%，并给出操作策略的关键设计结论。
- **核对范围：** 通读摘要、§1 引言、§2 形式化（DDPM、条件化、闭环动作序列）、§3 设计决策（CNN / Transformer、视觉编码器、噪声调度、DDIM 加速）、§4 性质（多模态、位置控制、动作序列、训练稳定性、控制理论联系）、§5 仿真基准与消融、§6 真机 Push-T 与扰动鲁棒性、§8 相关工作、§9 局限、§10 结论；§6.2–6.3、§7（翻马克杯、倒酱、双臂任务）与附录超参仅浏览，未逐项核对。
- **通读：** 已完成

## 核心结论（据原文）

- **形式化：** 策略建模为 p(A_t | O_t) 的条件 DDPM，K 步去噪；视觉特征只算一次、不进入去噪输出，使实时推理可行。
- **闭环动作序列：** 观测 To 步 → 预测 Tp 步 → 执行 Ta 步再重规划（receding horizon），兼顾时间一致性与响应性。
- **网络：** CNN 版开箱即用但对高频、陡变动作（如速度指令）偏平滑；Transformer 版在高频任务更好但对超参敏感。作者建议先用 CNN 版。
- **视觉编码器：** 端到端训练的 ResNet-18（空间 softmax、GroupNorm）效果最好；冻结预训练表现差，小学习率微调（策略网络的 1/10）最佳（CLIP ViT-B/16 在 robomimic Square 上 98%）。
- **位置控制：** 与 diffusion 配合优于速度控制（多模态更显著、复合误差更小）。
- **训练稳定性：** 相比 IBC（需负采样、评测成功率振荡），diffusion 学习能量梯度，训练稳定。
- **推理速度：** DDIM 训练 100 步 / 推理 10 步，RTX 3080 约 0.1 s 延迟；对最多 4 步延迟仍保持峰值性能。
- **真机 Push-T：** 成功率 95%（人类示范 100%），最强 IBC 0%、LSTM-GMM 20%；遮挡相机、移动物体等扰动下可即时重规划。
- **局限（作者所述）：** 继承行为克隆对示范数据不足敏感；计算成本与推理延迟高于 LSTM-GMM 等简单方法，高频控制任务可能不够，可借助加速方法或结合 RL。
