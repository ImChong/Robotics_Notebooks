# Spark-H3：官方代码与运行入口核对

> 一手代码仓库来源归档（repo）

- **名称：** zechengtang/Spark-H3
- **类型：** repo / 稀疏注意力推理后端
- **地址：** https://github.com/zechengtang/Spark-H3
- **论文：** https://arxiv.org/abs/2611.24256
- **主页：** https://zechengtang.github.io/Spark-H3/
- **Hugging Face：** https://huggingface.co/Aazeus/Spark-H3
- **入库日期：** 2026-11-23
- **核对范围：** 阅读 main 的 README 英文版、`h3_sparse_attention/README.md` 安装/集成/路由/作用范围及默认参数，查看 `h3_sparse_attention/__init__.py`、`spark.py`、`processor.py` 的 `install_h3_spark_attn`、`H3SparseAttentionPlugin`、`_H3SparseProcessor` 的真实调用路径，及 `spark_integration.py` 的 route/sparse 调度，核对 `comfyui/` 与 `Ref2VA/` 存在。未逐行审计全部底层 CUDA/Triton、没有安装依赖或在 GPU 上运行。
- **通读：** 已完成
- **开源状态：** **已开源**：MiniMax-H3 的 Spark 稀疏注意力替换层、示例调用、Ref2VA 和 ComfyUI；不包括 MiniMax-H3 基座权重。
- **许可证：** 官方 Spark-H3 源码 Apache-2.0，子依赖保留原许可。

## 官方 Python / Diffusers 入口

先配置上游 MiniMax-H3，依 README 执行 `python -m pip install -e '.[cuda]'`；在**已构造且加载的** `pipe` 上运行如下示例（不含外部 H3 模型加载）：

```python
from h3_sparse_attention import install_h3_spark_attn

num_denoise_steps = 19
with install_h3_spark_attn(
    pipe.transformer,
    num_denoise_steps=num_denoise_steps,
    warmup_mode="warmup_steps",
    warmup_steps=4,
    dense_layers=0,
    topk_ratio=0.1,
    mode="full",
) as attention:
    result = pipe(**inputs, num_inference_steps=num_denoise_steps + 1)
    stats = attention.summary()
```

**重要：** MiniMaxH3Scheduler 的 `num_inference_steps` 计入最终 σ=0，真实 Transformer 评估次数少 1；上例是 19 次去噪 + σ 终点，不应把 20 次 scheduler 点等同 20 次 DiT 调用。

## 已核对的实际模块调用链

1. `h3_sparse_attention/__init__.py` 导出 `install_h3_spark_attn`；在 `processor.py` 中构造 `H3SparseAttentionConfig.spark(...)` 与 `H3SparseAttentionPlugin`。
2. `H3SparseAttentionPlugin.__enter__` 通过 `register_forward_pre_hook` 注册 `_Controller.begin_forward`，把各 `transformer_blocks[*].attn` 原 processor 备份并替换成 `_H3SparseProcessor`。
3. `_Controller.begin_forward` 从 `token_tags`、`position_ids` 和视频位置等参数构造 packed layout；`_H3SparseProcessor.__call__` 对 warmup / 指定 dense 层走原 processor，其余先计算 norm+RoPE 后的 Q/K/V 并按 layout permutation 对齐。
4. `_sol_attention` 按配置进入 `spark_attention`（`spark_integration.py`）；使用 Spark-Reblock 分块、Top-K 路由、Spark-Reweight 全局 anchor 加权 KV 和稀疏核的精确+压缩融合；对应 `landmark_tree_v2.py`、`sol_numerator_virtual_q.py` 及后端实现。
5. `_H3SparseProcessor` 把输出按 `inverse_permutation` 还原、做 `attn.to_out`；上下文管理器 `__exit__` 通过 `remove()` 恢复原 processor 并移除 prehook。可使用 `attention.summary()` 查看统计。
6. 仅目标视频 token 的主区间启用 sparse；文本、音频及条件 context 仍可按 README 保持精确。`Ref2VA` 有独立 scope 选项，具体以 README 与配置为准。

## 可复现条件和限制

- 用户本地已有 MiniMax-H3 模型、兼容 PyTorch/CUDA/Triton，必须核对 GPU Compute Capability、融合核版本及对应 ComfyUI release；原项目 README 和 backend 文档之间存在支持矩阵随版本演进，勿按论文示例保证在所有卡上运行。
- `topk_ratio` 定义的是目标视频候选块中的精确分支预算，不能拿它直接推导完整 packed Q/K 比例或全模型 FLOP 减少量。
- 默认较短序列可回退 dense；保留 warmup 与上下文密集区域。
- 尚未实机执行或验证端到端时延，使用论文 Table 1 时引用论文固定测试口径。

## 对 wiki 的映射

- [Spark-H3 论文与工程实体](../../wiki/entities/paper-spark-h3.md)
- [项目主页归档](../sites/spark-h3-project.md)
- [论文 v1 来源归档](../papers/spark_h3_arxiv_2611_24256.md)
