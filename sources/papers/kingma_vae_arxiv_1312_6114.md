# VAE：变分自编码贝叶斯（arXiv:1312.6114）

> 论文来源归档（ingest）

- **标题：** Auto-Encoding Variational Bayes
- **作者：** Diederik P. Kingma, Max Welling
- **类型：** paper / generative-model / vae
- **arXiv：** <https://arxiv.org/abs/1312.6114> · PDF：<https://arxiv.org/pdf/1312.6114.pdf>
- **会议：** ICLR 2014
- **入库日期：** 2026-09-21
- **一句话说明：** 用 **重参数化技巧** 把变分推断做成可反向传播的编码器–解码器，最大化 ELBO，得到连续可采样的潜空间。

## 核心摘录（面向 wiki 编译）

### 1) ELBO = 重建 − KL

- **要点：** \(\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x)\,\|\,p(z))\)。第一项逼真重建，第二项把后验拉向先验（通常 \(\mathcal{N}(0,I)\)），使潜空间可插值、可采样。
- **对 wiki 的映射：** [`wiki/concepts/autoencoder.md`](../../wiki/concepts/autoencoder.md)、[`wiki/formalizations/generative-foundations.md`](../../wiki/formalizations/generative-foundations.md)

### 2) 重参数化让编码器可训

- **要点：** \(z = \mu_\phi(x) + \sigma_\phi(x)\odot \epsilon,\ \epsilon\sim\mathcal{N}(0,I)\)，随机性从计算图中剥离，梯度能穿过采样。这是 VAE 相对不可微采样推断的工程突破。
- **对 wiki 的映射：** [`wiki/concepts/autoencoder.md`](../../wiki/concepts/autoencoder.md)

### 3) 机器人侧：世界模型与动作平滑

- **要点：** Dreamer 类世界模型用 VAE/RSSM 压观测；运动生成常用 VAE 学动作流形。代价是重建偏平均、细节糊——这也是后来扩散路线要补的缺口。
- **对 wiki 的映射：** [`wiki/concepts/latent-imagination.md`](../../wiki/concepts/latent-imagination.md)、[`wiki/overview/ai-architecture-map.md`](../../wiki/overview/ai-architecture-map.md)

## 开源状态（步骤 2.5）

- 经典论文，**无单一现代官方仓**；实现已是教材与框架范例。

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
