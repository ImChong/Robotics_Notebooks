# ChunkTrust（hf618/ChunkTrust）

- **标题:** ChunkTrust
- **链接:** https://github.com/hf618/ChunkTrust
- **类型:** repo / vla / execution-horizon / test-time
- **许可:** MIT
- **项目页:** https://hf618.github.io/ChunkTrust.github.io/
- **论文:** https://arxiv.org/abs/2609.39754
- **Hugging Face:** https://huggingface.co/Niugan/ChunkTrust（QHA checkpoints）
- **最后核查:** 2026-10-01
- **入库日期:** 2026-10-01

## 开源状态（步骤 2.5）

- **已开源：** Python 包 `chunktrust`（`AHS`、`HybridQHARuntimeSelector`）、`docs/integration.md` 对接任意需 `predict_with_trace` 的 flow chunk policy；`docs/backends.md` 覆盖 RoboTwin 2.0 / RoboCasa / 真机栈。
- **QHA 权重：** HF 上 `qha_pi05_8task_step5000`、`qha_pi0_8task_step5000`、`qha_pi05_heldout6_step10000` 等；`python scripts/download_asset.py ...` 拉取。
- **CI：** GitHub Actions `core.yml`（README badge）。

## 核心内容摘要

1. **AHS 接口：** `policy.predict_with_trace(obs)` → velocity trace `[T,H,D]` + actions `[H,D]`；`selector.select(velocity, actions, executed_history)` → 前缀长度 `k`。
2. **QHA 部署：** 冻结特征 → prior 分布；`HybridQHARuntimeSelector.select(prior, q_mix, max_exec_length=H)` 与 AHS 证据 **单次** 融合并更新记忆。
3. **Quickstart：** `examples/quickstart.py` + `pytest` smoke；非实验结果，演示 prior 融合。
4. **配置 / 结果：** `configs/` 任务划分；`results/` 记录表与 manifest。

## 对 wiki 的映射

- **wiki/entities/paper-chunktrust.md** — 论文实体
- **wiki/concepts/receding-horizon-policy-execution.md** — execution horizon 概念
- **wiki/methods/action-chunking.md** — chunk 部署与动态 horizon 谱系
