# AI 架构地图：一手奠基论文簇

> 来源归档（ingest 合集）

- **类型：** paper-collection / deep-learning / architecture
- **入库日期：** 2026-09-21
- **最后更新：** 2026-09-21
- **一句话说明：** 按六支函数族收录 **MLP / MoE / CNN / ResNet / U-Net / ViT / RNN–LSTM–GRU / TCN / Transformer / SSM–Mamba / AE–VAE / GAN / Diffusion / DiT / GNN** 的奠基论文，供 [AI 架构地图](../../wiki/overview/ai-architecture-map.md) 编译，而不是转存摘要。

## 主论文（本批 ingest 核心）

| 族 | 论文 | 标识 | 角色 |
|----|------|------|------|
| 前馈 | Rumelhart et al. Learning representations by back-propagating errors | Nature 1986 | 多层感知机可端到端训练 |
| 前馈 | Shazeer et al. Outrageously Large Neural Networks | [arXiv:1701.06538](https://arxiv.org/abs/1701.06538) | 稀疏门控 MoE |
| 空间 | LeCun et al. Gradient-Based Learning Applied to Document Recognition | Proc. IEEE 1998 | CNN / LeNet-5 |
| 空间 | He et al. Deep Residual Learning | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) | ResNet 捷径 |
| 空间 | Ronneberger et al. U-Net | [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) | 编码–解码 + 跳跃 |
| 空间 | Dosovitskiy et al. An Image is Worth 16x16 Words | [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) | ViT |
| 序列 | Hochreiter & Schmidhuber LSTM | Neural Computation 1997 | 长程门控记忆 |
| 序列 | Cho et al. Learning Phrase Representations | [arXiv:1406.1078](https://arxiv.org/abs/1406.1078) | GRU / seq2seq |
| 序列 | Bai et al. Empirical Evaluation of Generic Conv/RNN | [arXiv:1803.01271](https://arxiv.org/abs/1803.01271) | TCN |
| 序列 | Vaswani et al. Attention Is All You Need | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | Transformer → [实体页](../../wiki/entities/paper-attention-is-all-you-need.md) |
| 序列 | Gu et al. Efficiently Modeling Long Sequences (S4) | [arXiv:2111.00396](https://arxiv.org/abs/2111.00396) | 结构化 SSM |
| 序列 | Gu & Dao Mamba | [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) | 选择性 SSM |
| 生成 | Kingma & Welling Auto-Encoding Variational Bayes | [arXiv:1312.6114](https://arxiv.org/abs/1312.6114) | VAE |
| 生成 | Goodfellow et al. Generative Adversarial Nets | [arXiv:1406.2661](https://arxiv.org/abs/1406.2661) | GAN |
| 生成 | Ho et al. Denoising Diffusion Probabilistic Models | [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) | DDPM |
| 生成 | Peebles & Xie Scalable Diffusion Models with Transformers | [arXiv:2212.09748](https://arxiv.org/abs/2212.09748) | DiT |
| 图 | Kipf & Welling Semi-Supervised Classification with GCN | [arXiv:1609.02907](https://arxiv.org/abs/1609.02907) | GCN |
| 决策 | Chi et al. Diffusion Policy | [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) | 动作扩散策略 |

详细摘录见各单篇归档；本簇只固定 **选型坐标** 与 wiki 映射。

## 核心摘录（面向 wiki 编译）

### 1) 函数族由数据几何决定，而不是由年代决定

- **要点：** MLP 处理向量；卷积处理栅格；递推/注意力/SSM 处理序列；GNN 处理关系图；生成模型处理分布而不是单点回归。机器人策略只是把上述函数族接到 `obs → action`。
- **对 wiki 的映射：** [`wiki/overview/ai-architecture-map.md`](../../wiki/overview/ai-architecture-map.md)

### 2) 容量扩展走两条路：加深（残差）与条件计算（MoE）

- **要点：** ResNet 用恒等捷径让深度可训；Shazeer MoE 用稀疏门控在几乎不增加逐步算力的前提下扩大参数容量。机器人低层策略通常只要浅 MLP；稀疏 MoE 更常见于 VLA 动作专家。
- **对 wiki 的映射：** [`wiki/concepts/mlp.md`](../../wiki/concepts/mlp.md)、[`wiki/concepts/mixture-of-experts.md`](../../wiki/concepts/mixture-of-experts.md)、[`wiki/entities/paper-resnet-deep-residual-learning.md`](../../wiki/entities/paper-resnet-deep-residual-learning.md)

### 3) 序列骨干的三维权衡：长程、并行、推理复杂度

- **要点：** RNN/LSTM/GRU 推理线性但训练难并行；Transformer 训练并行、注意力 \(O(n^2)\)；TCN 用膨胀因果卷积换并行；S4/Mamba 用状态压缩换近线性长上下文。
- **对 wiki 的映射：** [`wiki/concepts/recurrent-neural-network.md`](../../wiki/concepts/recurrent-neural-network.md)、[`wiki/concepts/temporal-convolutional-network.md`](../../wiki/concepts/temporal-convolutional-network.md)、[`wiki/concepts/transformer.md`](../../wiki/concepts/transformer.md)、[`wiki/concepts/mamba.md`](../../wiki/concepts/mamba.md)

### 4) 生成模型把「多模态动作」从均值回归里救出来

- **要点：** VAE 给连续潜空间；GAN 给对抗样本质量；DDPM/DiT 把生成拆成稳定的多步监督。机器人侧 Diffusion Policy 与 VLA 的 flow/DiT 动作头直接继承这条线。
- **对 wiki 的映射：** [`wiki/concepts/autoencoder.md`](../../wiki/concepts/autoencoder.md)、[`wiki/concepts/generative-adversarial-network.md`](../../wiki/concepts/generative-adversarial-network.md)、[`wiki/concepts/diffusion-model.md`](../../wiki/concepts/diffusion-model.md)、[`wiki/concepts/diffusion-transformer.md`](../../wiki/concepts/diffusion-transformer.md)、[`wiki/methods/diffusion-policy.md`](../../wiki/methods/diffusion-policy.md)

## 开源核查（步骤 2.5，项目页 / 官方仓）

| 论文 | 官方代码（入库日 2026-09-21） | 结论 |
|------|------------------------------|------|
| ResNet | <https://github.com/KaimingHe/deep-residual-networks> | **已开源**（Caffe 参考实现） |
| U-Net | 项目页 <https://lmb.informatik.uni-freiburg.de/people/ronneberger/u-net/> | **已开源**（Caffe 参考 + 大量再实现） |
| ViT | <https://github.com/google-research/vision_transformer> | **已开源**（Apache-2.0） |
| TCN | <https://github.com/locuslab/TCN> | **已开源** |
| Transformer | 论文配套 tensor2tensor；现代实现遍布框架 | **已开源** |
| S4 | <https://github.com/state-spaces/s4> | **已开源** |
| Mamba | <https://github.com/state-spaces/mamba> | **已开源**（CUDA 扫描核） |
| DDPM | <https://github.com/hojonathanho/diffusion> | **已开源** |
| DiT | <https://github.com/facebookresearch/DiT> | **已开源** |
| GCN | <https://github.com/tkipf/gcn> | **已开源** |
| Diffusion Policy | 见既有 [`diffusion_policy_arxiv_2303_04137.md`](./diffusion_policy_arxiv_2303_04137.md) | **已开源** |
| LSTM / GRU / VAE / GAN / LeCun 1998 | 经典论文；实现已进入教材与框架 | **无单一现代官方仓**，按「确认无统一官方仓、生态已开源」处理 |

## 对 wiki 的映射（全量）

- [`wiki/overview/ai-architecture-map.md`](../../wiki/overview/ai-architecture-map.md)
- [`wiki/concepts/mlp.md`](../../wiki/concepts/mlp.md)
- [`wiki/concepts/mixture-of-experts.md`](../../wiki/concepts/mixture-of-experts.md)
- [`wiki/concepts/convolutional-neural-network.md`](../../wiki/concepts/convolutional-neural-network.md)
- [`wiki/concepts/unet.md`](../../wiki/concepts/unet.md)
- [`wiki/concepts/vision-transformer.md`](../../wiki/concepts/vision-transformer.md)
- [`wiki/concepts/recurrent-neural-network.md`](../../wiki/concepts/recurrent-neural-network.md)
- [`wiki/concepts/temporal-convolutional-network.md`](../../wiki/concepts/temporal-convolutional-network.md)
- [`wiki/concepts/transformer.md`](../../wiki/concepts/transformer.md)
- [`wiki/concepts/state-space-model-ssm.md`](../../wiki/concepts/state-space-model-ssm.md)
- [`wiki/concepts/mamba.md`](../../wiki/concepts/mamba.md)
- [`wiki/concepts/autoencoder.md`](../../wiki/concepts/autoencoder.md)
- [`wiki/concepts/generative-adversarial-network.md`](../../wiki/concepts/generative-adversarial-network.md)
- [`wiki/concepts/diffusion-model.md`](../../wiki/concepts/diffusion-model.md)
- [`wiki/concepts/diffusion-transformer.md`](../../wiki/concepts/diffusion-transformer.md)
- [`wiki/concepts/graph-neural-network.md`](../../wiki/concepts/graph-neural-network.md)
- [`wiki/concepts/humanoid-policy-network-architecture.md`](../../wiki/concepts/humanoid-policy-network-architecture.md)
- [`wiki/methods/diffusion-policy.md`](../../wiki/methods/diffusion-policy.md)
- [`wiki/methods/vla.md`](../../wiki/methods/vla.md)

## 当前提炼状态

- [x] 六支函数族与一手论文索引
- [x] 开源边界按项目页 / 官方仓核对
- [x] wiki 页面映射
