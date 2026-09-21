# DDPM：去噪扩散概率模型（arXiv:2006.11239）

> 论文来源归档（ingest）

- **标题：** Denoising Diffusion Probabilistic Models
- **作者：** Jonathan Ho, Ajay Jain, Pieter Abbeel
- **类型：** paper / generative-model / diffusion
- **arXiv：** <https://arxiv.org/abs/2006.11239> · PDF：<https://arxiv.org/pdf/2006.11239.pdf>
- **会议：** NeurIPS 2020
- **官方代码：** <https://github.com/hojonathanho/diffusion>
- **入库日期：** 2026-09-21
- **一句话说明：** 固定前向高斯加噪，训练网络预测所加噪声，用简单均方损失学逆向马尔可夫链，使扩散生成在图像上达到当时 SOTA。

## 核心摘录（面向 wiki 编译）

### 1) 任意 \(t\) 可一步采样 \(x_t\)

- **要点：** \(x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon\)。训练随机抽时间步与噪声，监督目标明确，避开 GAN 对抗动态。
- **对 wiki 的映射：** [`wiki/concepts/diffusion-model.md`](../../wiki/concepts/diffusion-model.md)

### 2) 预测噪声的均方损失

- **要点：** \(\mathcal{L} = \mathbb{E}\|\epsilon-\epsilon_\theta(x_t,t)\|^2\)。同一网络覆盖高噪声（结构）与低噪声（细节）。骨干当时用 U-Net。
- **对 wiki 的映射：** [`wiki/concepts/diffusion-model.md`](../../wiki/concepts/diffusion-model.md)、[`wiki/concepts/unet.md`](../../wiki/concepts/unet.md)

### 3) 机器人动作生成直接搬这条损失

- **要点：** Diffusion Policy 把 \(x_0\) 换成动作块；VLA 的 DiT/flow 头把骨干从 U-Net 换成 Transformer，但「多步去噪表达多模态」仍是 DDPM 的遗产。
- **对 wiki 的映射：** [`wiki/methods/diffusion-policy.md`](../../wiki/methods/diffusion-policy.md)、[`wiki/overview/ai-architecture-map.md`](../../wiki/overview/ai-architecture-map.md)

## 开源状态（步骤 2.5）

- `hojonathanho/diffusion` **已开源**。

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
