---
type: concept
tags: [vit, vision-transformer, computer-vision, deep-learning, perception, backbone]
status: complete
updated: 2026-07-22
related:
  - ../comparisons/cnn-vs-vit-backbones.md
  - ./vision-backbones.md
  - ../overview/topic-vision-backbone.md
  - ./generative-vision-pretraining.md
  - ../methods/unified-multimodal-tokens.md
  - ../queries/perception-backbone-selection.md
  - ../entities/paper-turingvit.md
sources:
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

- **全局上下文**：远距离部件关联（物体整体轮廓、前景–背景关系）无需像 CNN 那样堆叠数十层。
- **与 NLP 栈统一**：同一套 Transformer 工具链、优化器与分布式训练经验可直接迁移到视觉与 **多模态 VLM**。
- **机器人上游**：[VLA](../methods/vla.md)、[视觉骨干](../concepts/vision-backbones.md) 中 **DINOv2 / SigLIP / ViT 塔** 已是默认选项之一；理解 patch 嵌入与 class token 有助于读懂策略输入接口。

## 核心机制

### 1. ViT vs CNN：感受野与归纳偏置

| 维度 | CNN | ViT |
|------|-----|-----|
| 早期感受野 | 局部邻域 | **全图**（自注意力） |
| 归纳偏置 | 局部性、平移等变 | 弱；位置靠学习 |
| 数据效率 | 中小数据友好 | 需大数据或强预训练 |
| 算力 | 对分辨率相对线性 | 朴素注意力随 patch 数 **O(n²)** |

### 2. 分块嵌入（两种等价实现）

对高 $H$、宽 $W$、块尺寸 $P$ 的图像，patch 数 $N = HW/P^2$。

1. **展平 + 线性**：每块 RGB 张量展平为长度 $3P^2$ 向量，经共享线性层映射至嵌入维 $D$。
2. **等效卷积**：`Conv2d(in=3, out=D, kernel=P, stride=P)`，输出 $D \times (H/P) \times (W/P)$ 再展平为序列。

标准 ViT-Base（224×224，patch 16）即 `Conv2d(3, 768, 16, 16)` → 196 个 patch token。

### 3. class token 与位置嵌入

- **class token**：可学习向量 $\mathbf{z}_0$ 拼在序列最前，经多层注意力聚合全部 patch 信息；分类 **仅使用** 其最终上下文向量 $\mathbf{z}_0^L$。
- **位置嵌入**：可学习矩阵 $\mathbf{E}_{pos} \in \mathbb{R}^{(N+1)\times D}$ 与 token 逐元素相加，补偿注意力本身 **置换不变** 导致的顺序丢失。

### 4. 仅编码器栈与自注意力

每层编码器块：**多头自注意力** → 残差 + LayerNorm → **前馈 MLP** → 残差 + LayerNorm。无解码器；序列长度与嵌入维 $D$ 全程不变。

单头注意力三步：

1. $\mathbf{Q}=\mathbf{X}\mathbf{W}_Q,\ \mathbf{K}=\mathbf{X}\mathbf{W}_K,\ \mathbf{V}=\mathbf{X}\mathbf{W}_V$
2. $\text{Attn}=\text{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_h})\mathbf{V}$
3. 多头输出拼接后经线性投影还原维度 $D$

### 5. 分类头与迁移学习

- 推理：$\mathbf{z}_0^L \rightarrow \text{MLP} \rightarrow \text{logits} \rightarrow \text{softmax}$
- **微调惯例**：ImageNet 预训练骨干 + **冻结 encoder、仅训新分类头**（参数量可降至总量的 ~0.03%），适合机器人侧小数据集快速适配；全量微调或 LoRA 用于数据充足场景。

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

## 优势与局限

**优势**

- 全局关系建模强，利于细粒度分类、开放场景语义
- 与语言 Transformer **架构同构**，便于 [统一多模态 token](../methods/unified-multimodal-tokens.md)
- 大数据 / 大参数 scaling 收益显著（对比 [CNN vs ViT](../comparisons/cnn-vs-vit-backbones.md)）

**局限**

- 中小数据集易欠拟合，常依赖 ImageNet 或 SSL 预训练
- 高分辨率下 token 数暴涨 → 算力与显存瓶颈；工程上采用 Swin、窗口注意力、混合 CNN–Transformer
- 机载 **实时检测** 场景 CNN/轻量骨干仍常占优（延迟、量化算子成熟度）

## 常见误区

1. **「ViT 完全取代 CNN。」** 边缘部署、小数据、高吞吐检测仍大量用 ResNet/YOLO 系。
2. **「注意力等于卷积的感受野。」** ViT 第一层即全局，但 **数据与算力** 需求不同；不是免费午餐。
3. **「必须用 class token。」** 检测/分割变体常用 **全 patch 特征** 或 FPN 式 neck，而非单一 CLS 向量。
4. **「patch 越大越好。」** patch 增大 → token 减少、算力下降，但 **空间细节** 损失，需按任务权衡。

## 与其他页面的关系

- [CNN vs ViT 骨干对比](../comparisons/cnn-vs-vit-backbones.md)：机器人感知选型
- [视觉骨干](../concepts/vision-backbones.md)：预训练 → 检测/VLA 链条
- [生成式视觉预训练](./generative-vision-pretraining.md)：MAE、DINOv2 等 ViT SSL
- [感知骨干选型 query](../queries/perception-backbone-selection.md)：任务导向决策树
- [TuringViT](../entities/paper-turingvit.md)：VLM-native **线性注意力主导** 的可定制 SOTA ViT（小鹏；配方公开、资产未开源）

## 推荐继续阅读

- 原始论文：*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*
- human five 配套教程：<https://github.com/VizuaraAI/Transformers-for-vision-BOOK>
- Hugging Face 预训练权重：`google/vit-base-patch16-224`
- TuringViT 项目页：<https://turingvit.github.io/>

## 参考来源

- [wechat_human_five_vit_intro.md](../../sources/blogs/wechat_human_five_vit_intro.md) — human five 微信公众号《ViT入门》（<https://mp.weixin.qq.com/s/ugiOirWHrSgEefG8W1-o6Q>）
- [TuringViT 论文摘录（arXiv:2606.24253）](../../sources/papers/turingvit_arxiv_2606_24253.md)
