---
type: method
tags: [vla, transformer, manipulation, foundation-policy, google-robotics]
status: complete
updated: 2026-05-10
related:
  - ./vla.md
  - ./dial-instruction-augmentation.md
  - ../concepts/foundation-policy.md
sources:
  - ../../sources/papers/rl_foundation_models.md
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
summary: "RT-1 将语言条件图像操控建模为序列预测；RT-2 将大规模视觉语言模型接入同一接口形成 VLA，是通用操作策略的关键基线系列。"
---

# Robotics Transformer（RT-1 / RT-2）

## 一句话定义

**Robotics Transformer（RT 系列）**：把多视角图像与自然语言指令编码为 token，并输出离散化机器人动作 token，使单一 Transformer 可在海量真实演示上训练成多任务操控策略；**RT-2** 进一步用视觉–语言大模型作骨干，将网络知识与机器人动作对齐（即常见所称 **VLA**）。

## 主要技术路线

- **统一序列接口**：视觉 token + 语言 token → 动作 token，推理频率与工程栈与 [Foundation Policy](../concepts/foundation-policy.md) 叙事一致。
- **两代跳跃**：RT-1 建立规模化真实演示上的 Transformer 策略；RT-2 将预训练 VLM 迁入同一接口即常见的 [VLA](./vla.md)。

## RT-1（Brohan et al.）

- **数据**：大规模真实机器人轨迹与多任务厨房类场景。
- **形态**：视觉编码 + 语言嵌入 + 解码器输出动作 token，强调可扩展的数据管线与推理频率工程。

## RT-2（Brohan et al.）

- **核心 jump**：直接基于预训练 VLM 微调出视觉–语言–动作联合模型，使开放词汇指令与物体语义迁移到控制决策。
- **与相关工作的位置**：常与 [SayCan](./saycan.md)（规划导向）、[DIAL](./dial-instruction-augmentation.md)（数据侧指令增强）、[Octo](./octo-model.md)（开源 generalist）对照阅读。

## 关联页面

- [VLA](./vla.md)
- [Foundation Policy](../concepts/foundation-policy.md)
- [DIAL](./dial-instruction-augmentation.md)

## 参考来源

- RT-1: Brohan et al., https://arxiv.org/abs/2212.06817 — https://research.google/blog/rt-1-robotics-transformer-for-real-world-control-at-scale/
- RT-2: Brohan et al., https://arxiv.org/abs/2307.15818
- [rl_foundation_models.md](../../sources/papers/rl_foundation_models.md)
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
