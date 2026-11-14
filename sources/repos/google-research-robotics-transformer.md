# Robotics Transformer（google-research/robotics_transformer）

- **URL：** <https://github.com/google-research/robotics_transformer>
- **组织：** google-research
- **关联论文：** [rt_1_arxiv_2212_06817](../papers/rt_1_arxiv_2212_06817.md)

## 一句话说明

RT-1 官方 TensorFlow / TF-Agents 实现：基于 `SequenceAgent` 与 `TransformerNetwork` 的多帧历史语言条件操作策略与评估代码。

## 源码目录与关键入口（2026-11-14 核查）

- **主策略与智能体：** [`sequence_agent.py`](https://github.com/google-research/robotics_transformer/blob/master/sequence_agent.py)（定义 `SequencePolicy` 与 `SequenceAgent`，封装 `_loss`、`_train`、`_apply_gradients` 与 `AsHalfTransition`）
- **核心网络：** [`transformer_network.py`](https://github.com/google-research/robotics_transformer/blob/master/transformer_network.py)（`TransformerNetwork`：区分训练期 `outer_rank == 2` 单次前向与推理期 `outer_rank == 1` 自回归循环；维护 `context_image_tokens` / `action_tokens` / `seq_idx` 滑动窗口状态）
- **视觉/动作分词器：** [`tokenizers/image_tokenizer.py`](https://github.com/google-research/robotics_transformer/blob/master/tokenizers/image_tokenizer.py)（`RT1ImageTokenizer`：FiLM EfficientNet + `TokenLearnerModule` 压缩到 8 token）、[`tokenizers/action_tokenizer.py`](https://github.com/google-research/robotics_transformer/blob/master/tokenizers/action_tokenizer.py)（`RT1ActionTokenizer`：离散 `vocab_size=256` 动作桶化与 `detokenize`）
- **解码器骨干：** [`transformer.py`](https://github.com/google-research/robotics_transformer/blob/master/transformer.py)（因果解码器 `_TransformerLayer` / `Transformer`）
- **测试与检查点：** `bazel test ...`、`python -m robotics_transformer.tokenizers.action_tokenizer.test`；`trained_checkpoints/` 含 `rt1main`、`rt1multirobot`、`rt1simreal` 三个 SavedModel 目录

## 交叉链接

- [RT-1 论文实体](../../wiki/entities/paper-rt-1.md)
