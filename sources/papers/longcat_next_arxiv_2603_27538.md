# LongCat-Next: Lexicalizing Modalities as Discrete Tokens（arXiv:2603.27538v1）

- **论文：** <https://arxiv.org/abs/2603.27538v1>
- **全文 HTML：** <https://arxiv.org/html/2603.27538v1>
- **代码：** <https://github.com/meituan-longcat/LongCat-Next>
- **模型：** <https://huggingface.co/meituan-longcat/LongCat-Next>
- **机构：** 美团 LongCat 团队
- **类型：** 原始论文
- **版本：** v1（2026-03-29）
- **入库日期：** 2026-11-10
- **核对范围：** arXiv v1 正文的摘要、DiNA、SAE/RVQ、dNaViT、统一训练与评测/消融及局限，交叉阅读官方仓库技术说明；未逐项验证所有附录实现。
- **通读：** 未完成

## 关键机制

1. **DiNA（Discrete Native Autoregression）：** 语言、视觉、音频转成离散 token，在同一个自回归 next-token 目标下训练。
2. **语义完整视觉词表：** 语义对齐编码器（SAE）和多级残差矢量量化（RVQ）在保留语义的同时编码细节，降低离散视觉建模的理解/生成冲突。
3. **dNaViT：** 原生分辨率视觉 Transformer，将输入视觉信息转换为可与语言联合建模的“视觉词”。
4. **统一理解与生成：** 图像理解对应视觉 token 条件下的文本预测；图像生成对应文本条件下的视觉 token 预测，生成端还需要视觉解码器。
5. **语言/视觉/音频统一训练：** 论文讨论多模态预训练与 SFT 组合，以及不同模态信号之间的损失与容量权衡。

## 数据与评测读法

- 报告视觉理解、视觉生成和语音多项 benchmark；“28× 压缩”是视觉表示的压缩配置，不意味着固定的图像重建质量保证。
- 必须对照同数据版本、评测协议和 model size 读分数；表中的结果是作者报告，尚未独立复测。
- LongCat-Next 采用 A3B 级 MoE（每 token 激活约 3B）；**不能**将其当成 LongCat-2.5 的 1.6T/48B 同一个模型。

## 开源结论

- 已有 [官方 GitHub](https://github.com/meituan-longcat/LongCat-Next)、[HF 模型卡](https://huggingface.co/meituan-longcat/LongCat-Next)，并有 `sft/` 训练样例，属于**代码及模型公开**。
- 论文附录与未逐项复核的消融细节仍需复核；因此当前归档“通读：未完成”，wiki 节点暂列草稿。

**对 wiki 的映射：** [LongCat-Next](../../wiki/entities/paper-longcat-next.md)
