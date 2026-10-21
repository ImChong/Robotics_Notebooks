# You could have invented transformers from memory

- **类型：** blog
- **作者：** Alex Wang
- **发布日期：** 2026-10-15
- **入库日期：** 2026-10-18
- **原文与交互演示：** <https://alexwang.ai/posts/associative-memory/>
- **代码：** <https://github.com/KeAWang/associative-memory-blog>
- **源码归档：** [实验仓库](../repos/keawang_associative_memory_blog.md)
- **一句话说明：** 从联想记忆的检索干扰出发，解释线性层、MLP 与自注意力。

## 核心摘录

1. 外积累加形成线性键值记忆；最小二乘可消除部分键间干扰。
2. 非线性门控抑制低相似度键的贡献，提供理解 MLP 的记忆视角。
3. 自注意力从上下文构造键值，区别于写入参数的 MLP 记忆；容量推导只针对随机键值与存储键查询，不能直接外推真实模型知识量。

## 对 wiki 的映射

- [Transformer：联想记忆视角](../../wiki/concepts/transformer.md) — 补充机制、假设边界与实验入口。

## 开放状态

截至 2026-10-18，原文链接到公开实验代码；README 给出 `uv run crosstalk.py`，结果写入 `data/crosstalk.json`。这是数值教学实验，不是机器人策略训练或部署代码；本次未执行该实验。
