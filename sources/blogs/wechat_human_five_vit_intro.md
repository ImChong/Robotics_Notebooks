# ViT入门

> 来源归档（blog / 微信公众号）

- **标题：** ViT入门
- **类型：** blog
- **作者：** human five（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/ugiOirWHrSgEefG8W1-o6Q
- **发表日期：** 2026-07-05
- **入库日期：** 2026-07-05
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`）；正文约 2.5 万字 / 22 图；配套代码库 [VizuaraAI/Transformers-for-vision-BOOK](https://github.com/VizuaraAI/Transformers-for-vision-BOOK)
- **原始落盘：** [wechat_human_five_vit_intro_2026-07-05.md](../raw/wechat_human_five_vit_intro_2026-07-05.md)
- **一句话说明：** 从 CNN 局部感受野对比切入，系统讲解 ViT 分块嵌入（展平+线性 / 等效卷积）、class token、可学习位置编码、仅编码器栈与 MLP 分类头，并以 Oxford-IIIT Pet 微调 `google/vit-base-patch16-224` 演示迁移学习全流程。

## 核心摘录（归纳，非全文）

### ViT vs CNN（全局 vs 局部）

- **CNN**：局部卷积堆叠逐步扩大感受野；擅长纹理与局部结构，远距离依赖需深层间接传递。
- **ViT**：第一层起 **全局自注意力**；任意 patch 可直接关联全图任意区域（鸟类示例、视错觉人脸）。
- **归纳偏置**：CNN 自带局部性/平移等变；ViT 弱偏置、更依赖数据规模。

### 架构要点

| 模块 | 机制 |
|------|------|
| **分块嵌入** | 图像 → 固定尺寸 patch → 展平+线性投影 **或** kernel=stride=patch 的 Conv2D（等价） |
| **class token** | 可学习向量拼在序列前端，经注意力聚合全局信息 |
| **位置嵌入** | 可学习向量逐 token 相加，补偿注意力无顺序感知 |
| **编码器栈** | 仅 encoder：多头自注意力 + MLP + 残差 + LayerNorm；**无解码器** |
| **分类头** | 仅取 class token 最终上下文向量 → MLP → logit |

### 自注意力三步（文内公式化）

1. 线性投影得 Q/K/V
2. 缩放点积 + softmax 得注意力权重
3. 加权求和 V 得新 token 表征；多头并行后拼接投影

### 优势与局限

- **优势**：全局上下文、大数据/大模型 scaling、与 NLP Transformer 工具链统一；多模态 VLM 视觉塔基础。
- **局限**：中小数据效率低于 CNN；注意力对 patch 数 **二次** 复杂度；高分辨率需 Swin/窗口变体。

### 实操（Oxford-IIIT Pet）

- 37 类猫狗细粒度分类；`ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")` 替换分类头。
- **冻结骨干、仅训 classifier**（约 2.8 万可训练参数 / 8600 万总量）；AdamW + cosine warmup；50 epoch 可达高验证精度。
- 文内 patch embed 实现为 `Conv2d(3, 768, kernel_size=16, stride=16)`。

### 主要参考

- 原始论文：*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*
- 配套教程仓库：[VizuaraAI/Transformers-for-vision-BOOK](https://github.com/VizuaraAI/Transformers-for-vision-BOOK)

## 对 wiki 的映射

| 主题 | 关系 |
|------|------|
| [Vision Transformer（概念）](../../wiki/concepts/vision-transformer.md) | **主沉淀页**：机制、训练/推理链路与误区 |
| [CNN vs ViT 骨干对比](../../wiki/comparisons/cnn-vs-vit-backbones.md) | 归纳偏置与机器人感知取舍 |
| [视觉骨干](../../wiki/concepts/vision-backbones.md) | 预训练→检测/VLA 上游链条 |
| [视觉感知骨干专题](../../wiki/overview/topic-vision-backbone.md) | 专题入口交叉引用 |
