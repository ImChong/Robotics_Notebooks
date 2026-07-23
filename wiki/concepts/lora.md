---
type: concept
tags: [fine-tuning, parameter-efficient, adaptation, peft]
status: complete
updated: 2026-07-22
related:
  - ../methods/vla.md
  - ../methods/mimic-video.md
  - ../entities/paper-fada-humanoid.md
  - ../entities/paper-any2any-cross-embodiment-wbt.md
  - ../entities/paper-m4world.md
  - ../entities/paper-wam-ttt-human-video-test-time-steering.md
  - ../entities/rldx-1.md
sources:
  - ../../sources/notes/lora.md
summary: "LoRA（低秩适配）冻结预训练权重，只训练一对低秩矩阵 B、A，用 W₀x + BAx 以极小参数量微调大模型；在本库多为 VLA/世界模型做动力学敏感模块的低成本适配。"
---

# LoRA (Low-Rank Adaptation，低秩适配)

**LoRA** 是一种参数高效微调（PEFT）方法：不改动预训练权重，只在其旁路上训练一对低秩矩阵，使大模型能以极小的可训练参数量适配新任务、新形态或新动力学。

## 一句话定义

冻结原权重 $W_0$，只学它的低秩增量 $\Delta W = BA$，前向从 $W_0 x$ 变成 $W_0 x + BA x$。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LoRA | Low-Rank Adaptation | 用低秩旁路矩阵近似权重增量的 PEFT 方法 |
| PEFT | Parameter-Efficient Fine-Tuning | 只训练少量参数的一类微调技术总称 |
| rank ($r$) | Rank | 旁路矩阵的秩，控制可吸收的适配容量 |
| QLoRA | Quantized LoRA | 4-bit 量化基座 + LoRA，进一步压显存的衍生法 |

## 为什么重要

- **极低训练成本**：可训练参数从「整个 $W$」降到「两条 $d\times r$ / $r\times k$ 矩阵」，显存与优化器状态大幅下降，单卡即可微调数十亿参数基座。
- **不动基座、可插拔**：$W_0$ 冻结，多个下游任务各自持有一份小 LoRA 权重，按需加载/合并，避免全量副本。
- **适配「动力学敏感模块」**：在机器人/具身场景，常只对 action decoder、IDM 等对动力学敏感的子模块插 LoRA，用少量目标域数据缩小外观 / 动力学 gap，而不破坏预训练得到的语义与几何先验。
- **推理零额外延迟**：训练完成后可将 $BA$ 合并回 $W_0$（$W' = W_0 + BA$），部署时与原模型同构、无额外算子。

## 核心机制

对一层线性权重 $W_0 \in \mathbb{R}^{d\times k}$，LoRA 令其增量为低秩分解：

$$
W' = W_0 + \Delta W = W_0 + BA,\quad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r\ll\min(d,k)
$$

- **只训 $A$、$B$**：反向传播只更新这两条矩阵，$W_0$ 全程冻结。
- **秩 $r$ 是容量旋钮**：$r$ 越大能吸收的适配信息越多、参数也越多；动力学 gap 小的场景用小 $r$ 即可。
- **缩放系数 $\alpha$**：实际用 $\frac{\alpha}{r}BA$ 缩放旁路输出，稳定不同 $r$ 下的学习率尺度。
- **初始化**：$A$ 随机、$B$ 置零，训练起点等价于原模型（$\Delta W = 0$），保证微调从预训练权重平滑出发。

## 在本库中的典型用法

| 页面 | LoRA 的角色 |
|------|-------------|
| [FADA](../entities/paper-fada-humanoid.md) | 冻结 $P$/$I$ 预训练权重，仅在 IDM 上优化 LoRA $\Delta\psi$ |
| [Any2Any](../entities/paper-any2any-cross-embodiment-wbt.md) | 在 Action Decoder 等动力学敏感模块插 LoRA，$r$ 控吸收动力学 gap 的容量 |
| [M4World](../entities/paper-m4world.md) | few-clip 后训练用 LoRA 绑定稀有外观/文本，保留基座几何与天气控制 |
| [mimic-video](../methods/mimic-video.md) | 视频阶段对骨干加 LoRA，用机器人域视频缩小外观/动力学域差 |

## 局限与风险

- **容量上限**：低秩假设对「与预训练分布差异极大」的任务可能不足，需增大 $r$ 或改全量微调。
- **模块选择敏感**：插在哪些层（attention 投影 / MLP / decoder）对效果影响大，需按任务消融。
- **多 LoRA 组合**：多个 LoRA 叠加/合并时可能相互干扰，需评估。

## 参考来源

- [LoRA 来源归档](../../sources/notes/lora.md)
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685（2021）
- 官方实现：<https://github.com/microsoft/LoRA>

## 关联页面

- [VLA](../methods/vla.md)
- [mimic-video](../methods/mimic-video.md)
- [FADA](../entities/paper-fada-humanoid.md)
- [Any2Any Cross-Embodiment WBT](../entities/paper-any2any-cross-embodiment-wbt.md)
- [M4World](../entities/paper-m4world.md)
- [WAM-TTT](../entities/paper-wam-ttt-human-video-test-time-steering.md)
- [RLDX-1](../entities/rldx-1.md)

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [LoRA 原始论文](https://arxiv.org/abs/2106.09685)
