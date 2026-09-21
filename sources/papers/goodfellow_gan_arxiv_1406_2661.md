# GAN：生成对抗网络（arXiv:1406.2661）

> 论文来源归档（ingest）

- **标题：** Generative Adversarial Nets
- **作者：** Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
- **类型：** paper / generative-model / gan
- **arXiv：** <https://arxiv.org/abs/1406.2661> · PDF：<https://arxiv.org/pdf/1406.2661.pdf>
- **会议：** NeurIPS 2014
- **入库日期：** 2026-09-21
- **一句话说明：** 生成器与判别器做 **极小极大博弈**，在判别器逼近最优时等价于最小化生成分布与数据分布的 Jensen–Shannon 散度。

## 核心摘录（面向 wiki 编译）

### 1) 极小极大目标

- **要点：** \(\min_G\max_D \mathbb{E}_{x\sim p_{data}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]\)。\(D\) 学「真假」，\(G\) 学「骗过 \(D\)」。不显式写似然也能出样本。
- **对 wiki 的映射：** [`wiki/concepts/generative-adversarial-network.md`](../../wiki/concepts/generative-adversarial-network.md)、[`wiki/formalizations/generative-foundations.md`](../../wiki/formalizations/generative-foundations.md)

### 2) 训练不稳与模式崩溃

- **要点：** 原文已提示饱和与平衡困难；后续实践中的模式崩溃、振荡是机器人 **不要默认用 GAN 当策略头** 的主因。GAN 更适合图像域迁移，而不是多模态动作回归。
- **对 wiki 的映射：** [`wiki/concepts/generative-adversarial-network.md`](../../wiki/concepts/generative-adversarial-network.md)、[`wiki/overview/ai-architecture-map.md`](../../wiki/overview/ai-architecture-map.md)

### 3) 机器人遗产：视觉 Sim2Real 与对抗运动先验

- **要点：** CycleGAN / RL-CycleGAN 做外观迁移；AMP 把对抗判别用在 **运动分布** 而不是像素。二者都借 GAN 的「隐式分布匹配」，但稳定配方已经高度特化。
- **对 wiki 的映射：** [`wiki/methods/amp-reward.md`](../../wiki/methods/amp-reward.md)、[`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md)

## 开源状态（步骤 2.5）

- 早期参考见 Goodfellow 相关实现；**无单一长期官方仓**。思想进入大量视觉与 AMP 代码。

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
