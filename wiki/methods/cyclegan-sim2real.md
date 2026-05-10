---
type: method
tags: [sim2real, domain-adaptation, vision, manipulation]
status: complete
updated: 2026-05-10
related:
  - ../concepts/sim2real.md
  - ./qt-opt.md
sources:
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
summary: "CycleGAN 一类无配对图像翻译可把仿真渲染转成接近真实风格的观测，用于缩小视觉 Sim2Real 域差；常与大规模 RL 数据采集并列讨论。"
---

# CycleGAN 与视觉 Sim2Real

## 一句话定义

**CycleGAN**：无配对样本下的图像到图像翻译网络，通过循环一致性约束学习仿真图像与真实风格图像之间的映射；在机器人叙事中常作为**像素域对齐**工具，辅助视觉策略迁移。

## 主要技术路线

- **无配对图像翻译**：学习仿真渲染图与真实风格图像之间的映射，不改动力学或控制器。
- **服务视觉闭环 RL**：与 [QT-Opt](./qt-opt.md) 等像素策略并列讨论时常用于减小观测域差；总体归类见 [Sim2Real](../concepts/sim2real.md)。

## 为什么单独成页

同一时期的大规模抓取 RL（如 [QT-Opt](./qt-opt.md)）依赖仿真预训练或混合数据时，视觉外观差异是主要瓶颈之一；CycleGAN 代表了一条与动力学随机化并行的**表征对齐**路线。

## 关联页面

- [Sim2Real](../concepts/sim2real.md)
- [QT-Opt](./qt-opt.md)

## 参考来源

- Zhu et al., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, https://arxiv.org/abs/1703.10593
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
