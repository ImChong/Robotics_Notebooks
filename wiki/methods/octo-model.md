---
type: method
tags: [vla, open-source, generalist-policy, diffusion, manipulation]
status: complete
updated: 2026-05-10
related:
  - ./vla.md
  - ../concepts/open-x-embodiment.md
  - ../concepts/foundation-policy.md
sources:
  - ../../sources/papers/rl_foundation_models.md
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
summary: "Octo 是基于 Open X-Embodiment 等数据训练的开源通用机器人策略，强调少样本适配新相机与动作空间，常与 RT 系列对照。"
---

# Octo（开源 Generalist Policy）

## 一句话定义

**Octo**：开源的通用机器人操作策略，通常基于 Transformer / 扩散式动作头，在 Open X-Embodiment 等多数据集上预训练，支持语言或图像目标，并可在新机器人上做高效微调。

## 主要技术路线

- **开源 generalist 模板**：在 [Open X-Embodiment](../concepts/open-x-embodiment.md) 等大混合数据上预训练，暴露动作头与观测适配接口以便迁移到新硬件。
- **社区基线**：降低「规模化预训练 + 微调」试错成本，与闭源的 [Robotics Transformer](./robotics-transformer-rt-series.md)、[Foundation Policy](../concepts/foundation-policy.md) 叙事并列。

## 关联页面

- [Open X-Embodiment](../concepts/open-x-embodiment.md)
- [VLA](./vla.md)

## 参考来源

- Ghosh et al., *Octo: An Open-Source Generalist Robot Policy*, https://arxiv.org/abs/2405.12213
- 项目页：https://octo-models.github.io/
- [rl_foundation_models.md](../../sources/papers/rl_foundation_models.md)
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
