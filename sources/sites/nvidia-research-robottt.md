# NVIDIA Research — RoboTTT（GEAR Lab）

> 来源归档（ingest）

- **标题：** RoboTTT: Context Scaling for Robot Policies
- **类型：** site（官方项目页）
- **发布方：** NVIDIA GEAR、Stanford、UT Austin（以项目页机构 logo 为准）
- **原始链接：** <https://research.nvidia.com/labs/gear/robottt/>
- **入库日期：** 2026-07-16
- **一句话说明：** 官方对外页：在 **GR00T N1.7** 类 VLA 中插入 **Test-Time Training（TTT）层**，用 **固定大小的 fast weights** 在训练与推理时对每步传感器读数做自监督梯度更新，把 **visuomotor 上下文缩放到 8K 步**（约 5 分钟 @30Hz）且 **推理延迟不随上下文增长**；报告双臂长程装配、上下文单次人视频模仿、在线自纠偏与扰动恢复等能力。

## 摘录要点（与论文分工）

- **核心机制：** TTT 层的隐藏状态即 **小型神经网络 fast weights \(W_t\)**；每步 token \(x_t\) 先对自监督损失 \(\ell\)（如重建刚收到的 token）做 **一步梯度更新**，再用更新后的 \(W_t\) 产生输出 \(o_t=f(x_t;W_t)\)。历史被 **压缩进权重** 而非 KV cache。
- **机器人实例化：** 在 **Vision-Language-Action（VLA）** 策略（实例为 **GR00T N1.7**）中插入 TTT 层；经 **tanh 门控 \(\alpha\)** 与 attention 分支残差相加，\(\alpha\) 初值近零以保护预训练能力。
- **序列训练：** flow-matching 动作头 + **sequence action forcing**（每 action chunk 独立噪声 \(\tau_t\)）+ **TBPTT**（段内反传、段间 fast weights 传递但梯度 detach）。
- **上下文利用：** 对选定 timestep **mask 动作损失** 可仅更新 fast weights——用于 **人视频 in-context 模仿** 与 **DAgger Distillation**（失败 rollout 作 context、人工纠正作监督）。
- **评测：** YAM 双臂 setup 上三项 dexterous 长程装配（Pup Go Car ~5min、Gear Bot ~5min、Circuit 1–2min）；相对 GR00T N1.7 单步上下文平均 **+87%** 任务完成分，8K 预训练上下文相对 1K **+62%**；GDN 等固定大小循环记忆基线无类似 scaling 趋势。
- **论文状态：** 入库时项目页 **未给出独立 arXiv 编号**；正文引用 GR00T N1（2503.14734）、TTT-E2E 长上下文（2512.23675）、Algorithm Distillation（2210.14215）、Gated DeltaNet（2412.06464）等。后续若发布 preprint 应补 `sources/papers/` 条目并回链本页。

## 对 wiki 的映射

- [paper-robottt-test-time-training-vla-context](../../wiki/entities/paper-robottt-test-time-training-vla-context.md) — 面向读者的机制摘要、流程图与和 TTT-Parkour / 短上下文 VLA 的对照
- [VLA](../../wiki/methods/vla.md) — 长程 visuomotor 上下文与部署后持续学习叙事
- [GR00T N1](../../wiki/entities/paper-hrl-stack-34-gr00t_n1.md) — RoboTTT 所基于的 VLA 底座
