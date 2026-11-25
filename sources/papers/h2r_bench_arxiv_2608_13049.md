# H2R-Bench（arXiv:2608.13049）

> 来源归档（ingest）

- **标题：** H2R-Bench: Benchmarking Human-to-Robot Manipulation Video Generation in World Models
- **类型：** paper / world-model / cross-embodiment / manipulation / benchmark
- **arXiv：** <https://arxiv.org/abs/2608.13049>
- **项目页：** <https://rongdingyi.github.io/H2R-Bench/>（归档见 [`sources/sites/h2r-bench.md`](../sites/h2r-bench.md)）
- **代码：** <https://github.com/Rongdingyi/H2R-Bench>（归档见 [`sources/repos/h2r-bench.md`](../repos/h2r-bench.md)）
- **入库日期：** 2026-08-19
- **通读：** 已完成
- **核对范围：** 2026-11-21 核对 arXiv:2608.13049v1 PDF 全文 3,398 行及官方仓 `Rongdingyi/H2R-Bench@d41cecb` 已开源的 `benchmark/` 与 `evaluation/` 评测脚本
- **一句话说明：** 120 个 EgoDex 源场景 × 2 种机器人本体（夹爪/灵巧手）= 240 个跨本体转移测试例，用 M1–M5 五维加权 H2RCore（3 MLLM 裁判 + VQA 人类相关性 $r=0.930$ / $\rho=0.883$）横评 11 个生成模型。

## 开源状态（步骤 2.5，2026-11-21 复核）

- **项目页核查：** 有 Paper / Project / HF Papers / GitHub 链；官方仓 [`Rongdingyi/H2R-Bench@d41cecb`](https://github.com/Rongdingyi/H2R-Bench) 现已发布完整基准与评测管线：`benchmark/scene_ids.txt`、`benchmark/annotations.jsonl`、`prompts/generation.jsonl`、`prompts/judging/` 与 `evaluation/`（`prepare.py`、`run.py`、`goal_action.py`、`contact.py`、`embodiment.py`、`aggregate.py`）。
- **结论：** **已开源基准标注与 MLLM 评测代码**（注：生成模型与 EgoDex 源视频需按指南本地准备，M1–M4 默认调用 Qwen3.7-Plus / Gemini 3.5 Flash / GPT-5.4 OpenAI 兼容端点）。

## 摘录：评测维度

| 维度 | 含义 |
|------|------|
| 目标状态 | 操作后物体/场景是否到达期望状态 |
| 动作事件 | 关键 manipulation 事件是否出现 |
| 功能接触 | 功能相关接触是否合理 |
| 本体正确性 | 生成视频是否符合目标机器人本体 |
| 视频质量 | 时序一致性与视觉保真 |

**对 wiki 的映射：** [`wiki/entities/paper-h2r-bench.md`](../../wiki/entities/paper-h2r-bench.md)；交叉 [生成式世界模型](../../wiki/methods/generative-world-models.md)、[跨本体迁移策略](../../wiki/queries/cross-embodiment-transfer-strategy.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查
