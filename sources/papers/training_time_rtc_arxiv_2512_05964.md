# Training-Time Action Conditioning for Efficient Real-Time Chunking（arXiv:2512.05964）

> 来源归档（ingest）

- **标题：** Training-Time Action Conditioning for Efficient Real-Time Chunking
- **短名：** Training-Time RTC / T-RTC
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2512.05964>
- **原文版本：** [arXiv:2512.05964v2（2025-12-09，HTML 全文）](https://arxiv.org/html/2512.05964)
- **通读：** 已完成
- **通读核验：** 2026-11-19 逐节核对摘要、背景、方法、Algorithm 1、Kinetix 与真机实验、讨论和局限；不将论文结论误写成部署保证。
- **关联推理期论文：** [real_time_chunking_arxiv_2506_07339.md](real_time_chunking_arxiv_2506_07339.md)（[arXiv:2506.07339](https://arxiv.org/abs/2506.07339)）
- **项目页：** <https://www.pi.website/research/real_time_chunking>（2025-12-08 博客更新指向本文）
- **机构：** 物理智能（Physical Intelligence）；加州大学伯克利分校（UC Berkeley）
- **入库日期：** 2026-09-30
- **一句话说明：** 训练时随机模拟推理延迟并对 action prefix 做 flow 条件化，推理期零 inpainting 开销，可作为 inference-time RTC 的 drop-in 替代；π₀.₆ 真机盒装与咖啡任务验证。

## 开源状态（步骤 2.5，2026-09-30）

- **部分开源**：Kinetix 仿真与 training-time 微调步骤在 [real-time-chunking-kinetix](../repos/real-time-chunking-kinetix.md)（`simulated_delay` + 对 BC 检查点再训 8 epoch）；π₀.₆ 真机权重与咖啡演示配方**未**作为 openpi 独立入口全量公开。
- **工程复现**：社区 [openpi-rtc](../repos/openpi-rtc.md) 在 openpi 上实现推理期 RTC；LeRobot 内置 [RTC 文档](../sites/lerobot-rtc-docs.md) 面向 π0 / SmolVLA 部署。

## 核心摘录（面向 wiki 编译）

- Inference-time RTC 用伪逆引导 inpainting，每步去噪需 VJP，**增加延迟**；高 delay 下能力有上限。
- Training-time：学 \(p(A_{t+d:H}\mid o_t, A_{t:t+d})\)——prefix 用干净 GT、τ=1，postfix 正常 flow；loss 只算 postfix；训练时随机采样 \(d\)。
- 三处实现改动：逐 token 的 flow timestep（adaLN）、prefix 不噪声、mask loss；**不改参数量**。
- 仿真 Kinetix（\(H=8\)）：delay≥2 时 T-RTC 优于 inference-time RTC；真机 box building / espresso：相对 inference-time RTC **更快且不差成功率**。
- **对 wiki 的映射：** [paper-training-time-real-time-chunking](../../wiki/entities/paper-training-time-real-time-chunking.md)；对照 [paper-real-time-chunking](../../wiki/entities/paper-real-time-chunking.md)

## 原文通读核验（2026-11-19）

- **方法与边界**：按 token 设置不同 flow timestep，冻结已执行承诺的前缀，损失仅作用于后缀；保持可学习参数量与机器人运行时接口，但 token-wise 时间条件化需模型实现支持。只对 *hard prefix* 条件化，不提供 inference-time RTC 的重叠动作 soft mask。
- **Kinetix（原文 §V-A）**：预测长度 `H=8`、4 层 MLP-Mixer、每点 2048 次 rollout；对齐总 32 epoch 的训练预算（第 24 epoch 开始增加前缀条件化，再训练 8 epoch）；训练延迟从 `{0,1,2,3,4}` 按递减权重抽样。`d≥2` 优于推理期 RTC，`d=0/1` 略弱。
- **真机（原文 §V-B）**：π₀.₆ 在纸盒组装与意式咖啡任务上各微调 8000 steps、batch size 512，训练时均匀抽样延迟 0–10；50Hz 控制器、远端 H100、5 步去噪，测得端到端延迟 training-time RTC **108 ms**、inference-time RTC **135 ms**。两方案任务成功率与执行速度大致持平，都优于同步推理速度，不能据此声称训练期版本在全部任务上成功率更高。
- **限制（原文 §VI）**：prefix 约束需满足 `d≤H-s`；训练期抽样延迟分布依赖部署时的实际延迟，且仅支持硬前缀；本文未公开 π₀.₆ 真机的完整复现实验配方。
