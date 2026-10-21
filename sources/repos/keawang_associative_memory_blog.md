# KeAWang/associative-memory-blog

- **类型：** repo
- **链接：** <https://github.com/KeAWang/associative-memory-blog>
- **配套原文：** <https://alexwang.ai/posts/associative-memory/>
- **博客归档：** [Alex Wang 联想记忆教程](../blogs/alexwang_associative_memory.md)
- **入库日期：** 2026-10-18
- **开放状态：** 实验源码公开可访问；本次核对 README，未执行实验，未核实许可证授权范围。
- **一句话说明：** 用随机键值对比较线性、MLP 非线性与 softmax 记忆检索的干扰。

## 运行入口

README：`uv run crosstalk.py`；脚本内联依赖为 NumPy。默认键维度 64，5 个随机种子平均，输出 `data/crosstalk.json`；长上下文实验比较固定 sharpness 与多种长度缩放。

## 对 wiki 的映射

- [Transformer](../../wiki/concepts/transformer.md) — 联想记忆教学实验入口。
