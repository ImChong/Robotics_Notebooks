---
type: method
tags: [language-conditioned, vlm, behavior-cloning, data-augmentation, google-robotics]
status: complete
updated: 2026-05-10
related:
  - ./robotics-transformer-rt-series.md
  - ./bc-z.md
  - ./imitation-learning.md
sources:
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
summary: "DIAL 用微调后的视觉语言模型为大规模无详细语言标注的演示轨迹自动生成多样指令，再训练语言条件 BC，扩展语义覆盖与泛化指令。"
---

# DIAL（指令增强）

## 一句话定义

**DIAL（Data-driven Instruction Augmentation for Language-conditioned control）**：在少量人工指令标注上微调 VLM，再对海量离线轨迹进行弱监督的「事后语言标签」生成，从而在不线性增加标注成本的前提下扩充语言–轨迹配对多样性。

## 主要技术路线

1. 小集合高质量轨迹–文本对微调 CLIP 式图文匹配。
2. 对未标注轨迹用候选指令集合（来自标注池与生成模型提案）检索或打分，生成多条候选指令。
3. 用增强后的数据集训练语言条件策略（论文中与 RT-1 架构结合讨论）。

上述管线把「弱标注海量轨迹」与 [Foundation Policy](../concepts/foundation-policy.md) 所需的语言多样性对齐。

## 关联页面

- [Robotics Transformer](./robotics-transformer-rt-series.md)
- [BC-Z](./bc-z.md)

## 参考来源

- Xiao et al., *Robotic Skill Acquisition via Instruction Augmentation with Vision-Language Models*, https://arxiv.org/abs/2211.11736
- 项目页：https://instructionaugmentation.github.io/
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
