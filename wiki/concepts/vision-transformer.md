---
type: concept
tags: [vit, vision-transformer, computer-vision, deep-learning, perception, backbone]
status: complete
updated: 2026-11-07
code: https://github.com/google-research/vision_transformer
related:
  - ../entities/transformer-cv-curriculum.md
  - ../queries/robot-perception-stack-selection-loop.md
  - ../comparisons/cnn-vs-vit-backbones.md
  - ./vision-backbones.md
  - ../overview/hub-vision-backbone.md
  - ./generative-vision-pretraining.md
  - ../methods/unified-multimodal-tokens.md
  - ../queries/perception-backbone-selection.md
  - ../entities/paper-turingvit.md
original:
  - ../../sources/papers/vit_dosovitskiy_arxiv_2010_11929.md
sources:
  - ../../sources/papers/vit_dosovitskiy_arxiv_2010_11929.md
  - ../../sources/blogs/wechat_human_five_vit_intro.md
  - ../../sources/papers/turingvit_arxiv_2606_24253.md
summary: "Vision Transformer 将图像切为 patch token，以全局自注意力替代卷积堆叠，经仅编码器栈与 class token 完成分类；大数据下 scaling 优异，是 VLM/VLA 视觉塔与多模态统一架构的基础模块。"
---

# Vision Transformer（ViT，视觉 Transformer）

**Vision Transformer（ViT）**：把图像划分为固定尺寸 **图像块（patch）**，将每块视作 **token**，经线性/卷积嵌入与 **可学习位置编码** 后送入 **Transformer 编码器栈**；分类任务依靠 **class token** 聚合全局上下文，再经 MLP 头输出类别 logit。

## 一句话定义

用 **「图像块 = 词元」** 把二维视觉问题改写成 NLP 式序列建模，**第一层即可全局自注意力**，以弱归纳偏置换取大数据场景下的 **表征上限与架构统一性**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ViT | Vision Transformer | 图像分块 + Transformer 编码的视觉骨干 |
| MLP | Multi-Layer Perceptron | ViT 分类头与前馈子层 |
| Q/K/V | Query / Key / Value | 自注意力三组投影向量 |
| CLS | Class Token | 聚合全图信息的可学习特殊 token |
| SSL | Self-Supervised Learning | MAE、DINOv2 等 ViT 预训练范式 |

## 为什么重要

- **全局上下文**：远距离部件关联（物体整体轮廓、前景–背景关系）无需像 CNN 那样堆叠数十层。原文检查注意力距离：部分头在最低层就关注大范围图像，注意力距离随深度增加。
- **算力–精度**：Dosovitskiy 等（ICLR 2021）在 JFT-300M 预训练下，ViT-H/14 的 ImageNet 精度 88.55%，ViT-L/16 为 87.76%，高于 BiT-L 的 87.54%，而预训练算力远低于 BiT-L（TPUv3-core-days：ViT-H/14 约 2.5k、ViT-L/16 约 0.68k、BiT-L 约 9.9k）；受控缩放实验里 ViT 比 ResNet 约省 2–4 倍预训练算力。
- **与 NLP 栈统一**：同一套 Transformer 工具链、优化器与分布式训练经验可直接迁移到视觉与 **多模态 VLM**。
- **机器人上游**：[VLA](../methods/vla.md)、[视觉骨干](../concepts/vision-backbones.md) 中 **DINOv2 / SigLIP / ViT 塔** 已是默认选项之一；理解 patch 嵌入与 class token 有助于读懂策略输入接口。

## 核心原理

以下对照原论文 §3 核实。

### 1. ViT vs CNN：感受野与归纳偏置

| 维度 | CNN | ViT |
|------|-----|-----|
| 早期感受野 | 局部邻域 | **全图**（自注意力） |
| 归纳偏置 | 局部性、平移等变 | 弱；位置靠学习 |
| 数据效率 | 中小数据友好 | 需大数据或强预训练；原文：ImageNet 预训练时 ViT-L 不如 ViT-B 且不如 BiT，JFT-300M 才显出优势 |
| 算力 | 对分辨率相对线性 | 朴素注意力随 patch 数 **O(n²)** |

### 2. 分块嵌入（两种等价实现）

对高 $H$、宽 $W$、块尺寸 $P$ 的图像，patch 数 $N = HW/P^2$。

1. **展平 + 线性**：每块 RGB 张量展平为长度 $3P^2$ 向量，经共享线性层映射至嵌入维 $D$。
2. **等效卷积**：`Conv2d(in=3, out=D, kernel=P, stride=P)`，输出 $D \times (H/P) \times (W/P)$ 再展平为序列。

标准 ViT-Base（224×224，patch 16）即 `Conv2d(3, 768, 16, 16)` → 196 个 patch token。

### 3. class token 与位置嵌入

- **class token**：可学习向量 $\mathbf{z}_0^0$ 拼在序列最前，经多层注意力聚合全部 patch 信息；分类 **仅使用** 其最终输出 $\mathbf{z}_L^0$（再过 LayerNorm）。
- **位置嵌入**：可学习矩阵 $\mathbf{E}_{pos} \in \mathbb{R}^{(N+1)\times D}$ 与 token 逐元素相加，补偿注意力本身 **置换不变** 导致的顺序丢失。原文用标准 **1D** 可学习位置嵌入，未见 2D 感知版本带来显著增益；学出的嵌入自行呈现 2D 拓扑（近处相似、同行同列相似）。更高分辨率微调时保持 patch 大小，对预训练位置嵌入做 2D 插值。

### 4. 仅编码器栈与自注意力

每层编码器块（原文式 2–3，**Pre-LN**：LayerNorm 在块之前、残差在块之后）：

$$z'_\ell=\mathrm{MSA}(\mathrm{LN}(z_{\ell-1}))+z_{\ell-1},\qquad z_\ell=\mathrm{MLP}(\mathrm{LN}(z'_\ell))+z'_\ell$$

MLP 为两层、GELU 激活。无解码器；序列长度与嵌入维 $D$ 全程不变。这与 [Transformer 原版](./transformer.md) 的 Post-LN 排列不同。

单头注意力三步：

1. $\mathbf{Q}=\mathbf{X}\mathbf{W}_Q,\ \mathbf{K}=\mathbf{X}\mathbf{W}_K,\ \mathbf{V}=\mathbf{X}\mathbf{W}_V$
2. $\text{Attn}=\text{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_h})\mathbf{V}$
3. 多头输出拼接后经线性投影还原维度 $D$

### 5. 分类头与迁移学习

- 推理：$\mathbf{z}_0^L \rightarrow \text{MLP} \rightarrow \text{logits} \rightarrow \text{softmax}$
- **原文微调做法**：去掉预训练头，接一个零初始化的 $D\times K$ 线性层，整体微调（SGD+动量，batch 512）；更高分辨率微调往往更好（ImageNet 取 512/518）。
- **冻结 encoder、仅训新分类头**（参数量可降至总量的 ~0.03%）是配套教程博客的做法，非本文结论（推测适用于机器人侧小数据集快速适配）；全量微调或 LoRA 用于数据充足场景。

## 流程总览

```mermaid
flowchart LR
  img["输入图像 H×W×3"] --> patch["分块 P×P"]
  patch --> embed["线性/卷积嵌入 → D 维"]
  embed --> cls["+ class token"]
  cls --> pos["+ 位置嵌入"]
  pos --> enc["L 层 Transformer 编码器"]
  enc --> head["取 class token → MLP 头"]
  head --> out["类别 logit"]
```

## 模型规模与结果（原文 Table 1–2）

| 模型 | 层数 | 隐藏维 D | MLP | 头数 | 参数 |
|------|------|---------|-----|------|------|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

JFT-300M 预训练后：ViT-H/14 ImageNet 88.55%、CIFAR-100 94.55%、VTAB(19 任务) 77.63%；Noisy Student（EfficientNet-L2）88.4/88.5%。序列长度与 patch 边长平方成反比，故 ViT-L/16 比 ViT-L/32 算力高很多。

## 工程实践

- **训练配方（原文）**：Adam（β1=0.9，β2=0.999）、batch 4096、权重衰减 0.1（作者发现对所有模型的迁移都有益）、线性 warmup 后线性衰减；对 ResNet 基线 Adam 也略优于 SGD。
- **数据量判断**：中小数据（ImageNet 级）不要期待 ViT 胜过同算力 ResNet；在 90M+ 图像预训练后才反超。机器人侧通常直接用公开预训练权重而非从头训练（推测）。
- **自监督**：原文的掩码补丁预测让 ViT-B/16 达 79.9% ImageNet，比从头训练高约 2%，但比有监督预训练低约 4%；对比学习留作未来工作。
- **选型**：实时检测等延迟敏感场景仍可优先 CNN/轻量骨干，见 [CNN vs ViT](../comparisons/cnn-vs-vit-backbones.md)；这一点来自本站选型页，不是本文实验。

## 优势与局限

**优势**

- 全局关系建模强，利于细粒度分类、开放场景语义
- 与语言 Transformer **架构同构**，便于 [统一多模态 token](../methods/unified-multimodal-tokens.md)
- 大数据 / 大参数 scaling 收益显著（对比 [CNN vs ViT](../comparisons/cnn-vs-vit-backbones.md)）

**局限**

- 中小数据集易欠拟合（原文：ViT 在小数据上比算力相近的 ResNet 更易过拟合），常依赖大规模预训练
- 原文自述尚未验证检测与分割，自监督与大规模有监督预训练仍有明显差距
- 高分辨率下 token 数暴涨 → 算力与显存瓶颈；工程上采用 Swin、窗口注意力、混合 CNN–Transformer
- 机载 **实时检测** 场景 CNN/轻量骨干仍常占优（延迟、量化算子成熟度）

## 常见误区

1. **「ViT 完全取代 CNN。」** 边缘部署、小数据、高吞吐检测仍大量用 ResNet/YOLO 系。
2. **「注意力等于卷积的感受野。」** ViT 第一层即全局，但 **数据与算力** 需求不同；不是免费午餐。
3. **「必须用 class token。」** 检测/分割变体常用 **全 patch 特征** 或 FPN 式 neck，而非单一 CLS 向量。
4. **「patch 越大越好。」** patch 增大 → token 减少、算力下降，但 **空间细节** 损失，需按任务权衡。

## 关联页面

- [Transformer](./transformer.md)、[多头注意力](./multi-head-attention.md)：ViT 沿用原版 Transformer 的多头自注意力与 MLP 块
- [Diffusion Transformer](./diffusion-transformer.md)：DiT 在 ViT 基础上加入 adaLN 条件调制

- [CNN vs ViT 骨干对比](../comparisons/cnn-vs-vit-backbones.md)：机器人感知选型
- [视觉骨干](../concepts/vision-backbones.md)：预训练 → 检测/VLA 链条
- [生成式视觉预训练](./generative-vision-pretraining.md)：MAE、DINOv2 等 ViT SSL
- [感知骨干选型 query](../queries/perception-backbone-selection.md)：任务导向决策树
- [TuringViT](../entities/paper-turingvit.md)：VLM-native **线性注意力主导** 的可定制 SOTA ViT（小鹏；配方公开、资产未开源）
- [AI 架构地图](../overview/ai-architecture-map.md)

## 推荐继续阅读

- 原始论文：*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*（[arXiv:2010.11929](https://arxiv.org/abs/2010.11929)）
- 官方代码：<https://github.com/google-research/vision_transformer>
- human five 配套教程：<https://github.com/VizuaraAI/Transformers-for-vision-BOOK>
- Hugging Face 预训练权重：`google/vit-base-patch16-224`
- TuringViT 项目页：<https://turingvit.github.io/>

## 参考来源

- [ViT 一手论文（arXiv:2010.11929）](../../sources/papers/vit_dosovitskiy_arxiv_2010_11929.md)
- [wechat_human_five_vit_intro.md](../../sources/blogs/wechat_human_five_vit_intro.md) — human five 微信公众号《ViT入门》（<https://mp.weixin.qq.com/s/ugiOirWHrSgEefG8W1-o6Q>）
- [TuringViT 论文摘录（arXiv:2606.24253）](../../sources/papers/turingvit_arxiv_2606_24253.md)
