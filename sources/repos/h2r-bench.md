# Rongdingyi/H2R-Bench

- **标题：** H2R-Bench 官方仓库（基准标注 + MLLM 五维评测代码）
- **类型：** repo
- **URL：** <https://github.com/Rongdingyi/H2R-Bench>
- **配套论文：** [arXiv:2608.13049](https://arxiv.org/abs/2608.13049) — [`sources/papers/h2r_bench_arxiv_2608_13049.md`](../papers/h2r_bench_arxiv_2608_13049.md)
- **入库日期：** 2026-08-19
- **通读：** 已完成
- **核对范围：** 2026-11-21 核对 `Rongdingyi/H2R-Bench@d41cecb`：`README.md`、`benchmark/`、`prompts/`、`evaluation/prepare.py`、`evaluation/run.py`

## 一句话说明

已发布 **benchmark 标注 + 生成/裁判提示词 + M1–M4 MLLM 评测入口**（`evaluation.prepare` → `evaluation.run`）。

## 仓库状态（2026-11-21 复核）

| 项 | 内容 |
|----|------|
| 已发布 | `benchmark/scene_ids.txt`（120 场景）、`benchmark/annotations.jsonl`、`prompts/generation.jsonl`（240 条）、`prompts/judging/`、`evaluation/*.py`、`docs/` 项目页 |
| 依赖与边界 | 运行 `evaluation.run` 需自备源/生成视频及 Qwen (`qwen3.7-plus`) / Gemini (`gemini-3.5-flash`) / GPT (`gpt-5.4`) API Key；HF 目前仍以 papers 页为主 |
| 核心入口 | `python -m evaluation.prepare` → `python -m evaluation.run --manifest ... --providers qwen,gemini,gpt` |

## 与 wiki 的关系

- 实体页：[paper-h2r-bench](../../wiki/entities/paper-h2r-bench.md) — 含基于 `evaluation.prepare` / `evaluation.run` 的源码运行时序图。
