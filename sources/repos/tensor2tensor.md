# tensor2tensor — Transformer 官方参考实现

> 来源归档（ingest · Attention Is All You Need 配套代码）

- **标题：** Tensor2Tensor（T2T）
- **类型：** repo
- **官方仓库：** <https://github.com/tensorflow/tensor2tensor>
- **维护方：** Google Research / Google Brain（历史）
- **许可证：** Apache-2.0
- **入库日期：** 2026-09-21
- **一句话说明：** Vaswani et al. *Attention Is All You Need*（NeurIPS 2017）的 **官方 TensorFlow 参考实现**；提供 Transformer encoder–decoder、机器翻译训练与 beam search 解码入口，现代框架（PyTorch/JAX/HuggingFace）均由此架构演化。

## 与论文的对应关系

| 论文模块 | 仓库入口（典型） |
|----------|------------------|
| Scaled dot-product attention | `tensor2tensor/models/transformer.py` |
| Multi-head attention + FFN block | Transformer layer stack |
| Positional encoding | 正弦/可学习位置嵌入 |
| WMT EN–DE / EN–FR 训练 | `t2t_trainer` + 公开数据预处理 recipe |

## 开源边界（步骤 2.5）

| 状态 | 说明 |
|------|------|
| **已开源** | 官方 GitHub 可 clone；README 含训练/翻译命令 |
| **维护状态** | 仓库仍可读，但 Google 侧新工作多迁移至 JAX/Flax 与 HuggingFace 生态；**复现论文数值** 优先对照 README 与 commit 历史，工程新项勿默认 T2T 为唯一依赖 |

## 对 wiki 的映射

- [paper-attention-is-all-you-need](../../wiki/entities/paper-attention-is-all-you-need.md)
- [transformer.md](../../wiki/concepts/transformer.md)

## 参考来源（原始）

- 官方仓库：<https://github.com/tensorflow/tensor2tensor>
- 论文：<https://arxiv.org/abs/1706.03762>
