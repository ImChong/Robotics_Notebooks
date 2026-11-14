---
type: method
tags: [vla, transformer, manipulation, foundation-policy, google-robotics]
status: complete
updated: 2026-11-14
related:
  - ../entities/paper-rt-1.md
  - ../entities/paper-rt-2.md
  - ../overview/vla-wm-reading-roadmap-14-papers-technology-map.md
  - ./vla.md
  - ./dial-instruction-augmentation.md
  - ../concepts/foundation-policy.md
sources:
  - ../../sources/papers/rl_foundation_models.md
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
  - ../../sources/papers/rt_1_arxiv_2212_06817.md
  - ../../sources/papers/rt2_vla_reading_arxiv_2307_15818.md
original:
  - ../../sources/papers/rt_1_arxiv_2212_06817.md
  - ../../sources/papers/rt2_vla_reading_arxiv_2307_15818.md
summary: "RT-1 将语言条件图像操控建模为序列预测；RT-2 将大规模视觉语言模型接入同一接口形成 VLA，是通用操作策略的关键基线系列。"
---

# Robotics Transformer（RT-1 / RT-2）

## 一句话定义

**Robotics Transformer（RT 系列）**：把多视角图像与自然语言指令编码为 token，并输出离散化机器人动作 token，使单一 Transformer 可在海量真实演示上训练成多任务操控策略；**RT-2** 进一步用视觉–语言大模型作骨干，将网络知识与机器人动作对齐（即常见所称 **VLA**）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RT-2 | Robotics Transformer 2 | 将 web 规模 VLM 能力迁移到机器人控制的代表工作 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| VLM | Vision-Language Model | 视觉-语言多模态理解模型，VLA 的上游 |

## 为什么重要

- **确立「先真机规模化 Transformer，再接 VLM」的演进骨架**：RT-1 验证了单模型跨 700+ 真机任务的可扩展性；RT-2 随后把互联网图文知识注入同一动作接口并确立 VLA 范式，成为后续 OpenVLA、Octo、π₀ 等开源基线的设计参照。
- **用统一离散序列打通语言与控制**：证明不需要为机器人重造专用符号系统，就能让 VLM 直接参与开放词汇指令理解与零样本语义推理。

## 核心原理

- **FiLM + TokenLearner（RT-1）**：将语言向量通过 FiLM 调制 EfficientNet-B3 多帧特征图，再用 TokenLearner 将每帧 81 个空间 token 压缩至 8 个，使 6 帧历史序列适配 Decoder-only Transformer，预测 256-bin 离散动作（6-DoF 末端增量、夹爪、底盘与终止信号）。
- **动作文本化 + Co-fine-tuning（RT-2）**：沿用 RT-1 的 256-bin 整数表示并写成 VQA 格式文本（或占用最少用词表 token 的 symbol tuning），与互联网视觉–语言数据联合微调 PaLI-X（5B/55B）与 PaLM-E（12B），防止机器人微调阶段灾难性遗忘。

## 主要技术路线

- **统一序列接口**：视觉 token + 语言 token → 动作 token，推理频率与工程栈与 [Foundation Policy](../concepts/foundation-policy.md) 叙事一致。
- **两代跳跃**：RT-1 建立规模化真实演示上的 Transformer 策略；RT-2 将预训练 VLM 迁入同一接口即常见的 [VLA](./vla.md)。

## RT-1（Brohan et al.）

- **数据**：大规模真实机器人轨迹与多任务厨房类场景。
- **形态**：视觉编码 + 语言嵌入 + 解码器输出动作 token，强调可扩展的数据管线与推理频率工程。

## RT-2（Brohan et al.）

- **核心 jump**：直接基于预训练 VLM 微调出视觉–语言–动作联合模型，使开放词汇指令与物体语义迁移到控制决策。
- **与相关工作的位置**：常与 [SayCan](./saycan.md)（规划导向）、[DIAL](./dial-instruction-augmentation.md)（数据侧指令增强）、[Octo](./octo-model.md)（开源 generalist）对照阅读。

## 工程实践

- **闭环延迟预算（RT-1 vs RT-2）**：RT-1（~35M）可在机载端跑实时 3 Hz 控制（Appendix C.1 观测–动作缓冲 ~280 ms）；RT-2 55B 则需部署在多 TPU 云端服务并经网络查询（55B 约 1–3 Hz，5B 约 5 Hz）。
- **复现选型分界**：RT-1 提供官方 TensorFlow/JAX 模型代码与 SavedModel 检查点（不含 130k 真机轨迹数据）；RT-2 官方完整训练与推理代码均未开源，实操应转向 [OpenVLA](../entities/paper-openvla.md) 或 [Octo](../entities/paper-octo.md)。
- **动作约束解码**：RT-2 推理阶段必须在词表上加合法动作 token 掩码，避免自回归输出退化为纯自然语言说明。

## 局限与风险

- **泛化非对称（语义强、运动学弱）**：RT-2 可理解「充当锤子的石头」等新语义并提升未见物体/背景泛化（论文报告综合泛化约 2 倍于 RT-1/MOO），但无法凭空获得机器人演示集之外的新物理运动技能。
- **离散 bin 分辨率天花板**：256-bin 逐维离散在高精度力控与高频双手装配中精度与流畅度均落后于扩散/流匹配连续动作头（如 Diffusion Policy、π₀）。

## 关联页面

- [RT-1 论文实体](../entities/paper-rt-1.md)
- [RT-2 论文实体](../entities/paper-rt-2.md)
- [VLA](./vla.md)
- [Foundation Policy](../concepts/foundation-policy.md)
- [DIAL](./dial-instruction-augmentation.md)

## 参考来源

- RT-1: Brohan et al., https://arxiv.org/abs/2212.06817 — https://research.google/blog/rt-1-robotics-transformer-for-real-world-control-at-scale/
- RT-2: Brohan et al., https://arxiv.org/abs/2307.15818
- [rt_1_arxiv_2212_06817.md](../../sources/papers/rt_1_arxiv_2212_06817.md) — RT-1 全文核对（FiLM/TokenLearner/280ms 时延/Model Card 局限）
- [rt2_vla_reading_arxiv_2307_15818.md](../../sources/papers/rt2_vla_reading_arxiv_2307_15818.md) — RT-2 全文核对（co-fine-tuning/symbol tuning/1-5Hz 推理与局限）
- [rl_foundation_models.md](../../sources/papers/rl_foundation_models.md)
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)

## 推荐继续阅读

- [RT-1 论文实体](../entities/paper-rt-1.md)
- [RT-2 论文实体](../entities/paper-rt-2.md)
- [OpenVLA 论文实体](../entities/paper-openvla.md)
