# RoboTTT: Context Scaling for Robot Policies（NVIDIA GEAR）

> 来源归档（ingest · 官方项目页 + 摘要策展）

- **标题：** RoboTTT: Context Scaling for Robot Policies（全称 *Test-Time-Training Robot Policies*）
- **缩写：** **RoboTTT**
- **类型：** paper / vision-language-action / test-time-training / long-horizon-manipulation
- **项目页：** <https://research.nvidia.com/labs/gear/robottt/>
- **机构：** NVIDIA GEAR、Stanford、UT Austin（以项目页为准；作者列表见后续 preprint）
- **底座模型：** GR00T N1.7（[arXiv:2503.14734](https://arxiv.org/abs/2503.14734) 系列）
- **入库日期：** 2026-07-16
- **arXiv：** 入库时 **暂无独立编号**（仅官方项目页；见 [站点归档](../sites/nvidia-research-robottt.md)）
- **一句话说明：** 在机器人 foundation VLA 内嵌 **TTT 层**：每步传感器读数触发 fast weights 上 **一步自监督梯度更新**，以 **固定大小神经网络状态** 压缩任意长交互历史；配合 **sequence action forcing + TBPTT** 把预训练 visuomotor 上下文扩到 **8K 步** 且不增推理延迟，解锁 **单次人视频 in-context 模仿、在线自改进、扰动恢复** 与 **上下文长度 scaling law**。

## 摘录 1：问题与 TTT 层机制（Abstract + Model）

- **现状痛点：** 主流机器人 foundation 策略多为 **单步或极短历史** visuomotor 上下文，长程多阶段装配、在线纠偏与从历史 rollout 学习困难。
- **TTT 层：** 层内携带 **微小模型**；隐藏状态 = 其 **fast weights \(W\)**。每 incoming token \(x_t\)：**Update** \(W_t = W_{t-1} - \eta \nabla \ell(W_{t-1}; x_t)\)（\(\ell\) 为自监督，如重建刚收到的 token）；**Apply** \(o_t = f(x_t; W_t)\)。因状态 **固定大小**，对任意长历史的 conditioning 成本 **恒定**——流被 **写入权重** 而非缓存。
- **与 RNN/KV 的对照：** 类似 RNN 的递推状态，但更新规则是 **显式梯度步**；相对 full attention，不随上下文线性增 latency。

**对 wiki 的映射：** 沉淀到 [`wiki/entities/paper-robottt-test-time-training-vla-context.md`](../../wiki/entities/paper-robottt-test-time-training-vla-context.md)；与 NLP 侧 [TTT-E2E（arXiv:2512.23675）](https://arxiv.org/abs/2512.23675) 的「长上下文 = 持续学习」叙事互参，但 RoboTTT 面向 **闭环机器人 flow-matching VLA**。

## 摘录 2：VLA 集成、门控与序列训练（Model + Sequence Training）

- **实例化：** 在 **VLA**（论文实例 **GR00T N1.7**）中插入 TTT 层，保持与原架构兼容；动作头为 **flow matching**，序列损失在 **进入每步时的 fast-weight 状态 \(W_{t-1}\)** 上评估。
- **门控：** \(o = \tanh(\alpha) \odot o_{\mathrm{TTT}} + o_{\mathrm{attn}}\)，\(\alpha\) 可学习、初值近零 → 训练初期 **等价预训练 VLA**，再按需打开 TTT 分支。
- **Sequence action forcing：** 每个 action chunk **独立采样** flow-matching 噪声 \(\tau_t\)；全序列共享噪声会使整条轨迹 **同易同难** 并 destabilize 训练。
- **TBPTT：** 序列切段反传以控显存；**fast weights 跨段传递**，段边界 **detach 其梯度**；\(W_0\) 初始化仍经 **meta-learning（gradients of gradients）** 优化。

**对 wiki 的映射：** 与 [GR00T N1](../../wiki/entities/paper-hrl-stack-34-gr00t_n1.md)、[Action Chunking](../../wiki/methods/action-chunking.md)、[VLA](../../wiki/methods/vla.md) 的 flow/chunk 部署语境交叉；训练配方区别于 [TTT-Parkour](../../wiki/entities/paper-notebook-ttt-parkour.md) 的「仿真里短时微调跑酷策略」。

## 摘录 3：有效上下文学习与评测（Effective Learning + Evaluation）

- **Mask 损失 = 纯 context：** 选定 timestep **不监督动作、只更新 fast weights** → 同一配方支持：(1) **人视频前缀 + 机器人轨迹后缀** 的 one-shot in-context 模仿；(2) **DAgger Distillation**——机器人失败动作作 context、人工纠正作 target，把 **algorithm distillation** 实例化到机器人；测试时 **无人工** 在线自纠偏。
- **主结果（项目页）：** vs GR00T N1.7（单步）、+1 历史帧、**GDN** 固定状态模型——RoboTTT 在三项 dexterous 长程任务上 **全面领先**（平均完成分约 **79** vs **42** 单步基线，**+87%** 相对提升）；**8K** 预训练上下文 vs **1K** 同模型 **+62%**；**5 分钟十阶段** 装配 **仅 RoboTTT 完整完成**。
- **新能力：** 未见配置 **单次人视频** 电路装配；**Roof/Wheel** 在线自恢复；中途 **拆件扰动** 后凭自身 rollout 历史 **重装**。

**对 wiki 的映射：** 与 [Manipulation 长程](../../wiki/tasks/manipulation.md)、[DAgger](../../wiki/methods/dagger.md) 及 VLA 页「长程记忆/部署后学习」小节互链；区分 [TTT-Parkour](../../wiki/entities/paper-notebook-ttt-parkour.md)（感知跑酷、仿真 TTT 微调）与 RoboTTT（**模型内持续 fast-weight 学习**）。

## 参考来源（原始）

- 官方项目页：<https://research.nvidia.com/labs/gear/robottt/>
- 站点归档：[nvidia-research-robottt.md](../sites/nvidia-research-robottt.md)
