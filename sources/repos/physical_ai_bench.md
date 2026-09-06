# SHI-Labs/physical-ai-bench

> 来源归档

- **标题：** Physical AI Bench（PAI-Bench）
- **类型：** repo / benchmark
- **组织：** SHI-Labs（Georgia Tech / CMU）
- **代码：** <https://github.com/SHI-Labs/physical-ai-bench>
- **论文：** <https://arxiv.org/abs/2512.01989>
- **许可：** MIT
- **入库日期：** 2026-09-06
- **一句话说明：** PAI-Bench 官方仓：三轨评测（G 生成 / C 条件生成 / U 理解）、HF 数据集与 Leaderboard 链接；Python 3.10 + `uv`。
- **开源状态：** **已开源** — 三轨 `evaluate.py` / `lmms-eval` 集成可辨识；Leaderboard Space 在线。
- **沉淀到 wiki：** [`wiki/entities/paper-sa-2512-01989-pai-bench-a-comprehensive-benchmark-for-physical-ai.md`](../../wiki/entities/paper-sa-2512-01989-pai-bench-a-comprehensive-benchmark-for-physical-ai.md)

## 仓库概况（2026-09-06）

| 字段 | 值 |
|------|-----|
| 顶栏 | `generation/`、`conditional_generation/`、`understanding/` |
| G 轨 | `generation/evaluate.py`（VBench 八维）+ `evaluate_vqa.py`（Domain MLLM judge） |
| C 轨 | `conditional_generation/` + Grounded-SAM-2 / DOVER / LPIPS |
| U 轨 | 推荐 `lmms-eval` `pai-bench` 分支，`--tasks pai_reason` |
| 数据 | 三份 HF dataset（见 [`hf-physical-ai-bench.md`](../sites/hf-physical-ai-bench.md)） |
| 榜单 | <https://huggingface.co/spaces/shi-labs/physical-ai-bench-leaderboard> |

## 对 wiki 的映射

- 论文实体：[`wiki/entities/paper-sa-2512-01989-pai-bench-a-comprehensive-benchmark-for-physical-ai.md`](../../wiki/entities/paper-sa-2512-01989-pai-bench-a-comprehensive-benchmark-for-physical-ai.md)
- 论文来源：[`sources/papers/physical_ai_bench_arxiv_2512_01989.md`](../papers/physical_ai_bench_arxiv_2512_01989.md)
- HF 站点：[`sources/sites/hf-physical-ai-bench.md`](../sites/hf-physical-ai-bench.md)
