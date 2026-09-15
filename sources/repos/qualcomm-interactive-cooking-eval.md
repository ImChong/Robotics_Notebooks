# Qualcomm Interactive Cooking Evaluator（qualcomm_interactive_cooking_eval）

- **URL：** <https://github.com/Qualcomm-AI-research/qualcomm_interactive_cooking_eval>
- **组织：** Qualcomm AI Research
- **License：** BSD-3 Clause Clear
- **关联项目页：** [LiveCook](../sites/livecook.md)、[Streaming Interventions](../sites/livecookv2-streaming-interventions.md)
- **关联论文：** [livecook_arxiv_2511_21998](../papers/livecook_arxiv_2511_21998.md)、[streaming_interventions_arxiv_2606_09547](../papers/streaming_interventions_arxiv_2606_09547.md)

## 一句话说明

Qualcomm Interactive Cooking / Ego-MC-Bench 官方评测：`data.py` 自动拉取 HF 数据集，`eval.py` 计算 IC-Acc 与 Mistake Detection（Prec/Rec/F1、BERT、ROUGE-L）。

## 入口与结构

| 路径 | 作用 |
|------|------|
| `data.py` | 加载 HF 上的 Qualcomm Interactive Cooking 或 Ego-MC-Bench 数据 |
| `eval.py` | 读取预测 JSON（`Instruction:` / `Feedback:` / `Success:` 前缀），输出流式评测指标 |
| `utils.py` | 辅助函数 |

预测 JSON 格式：`video_id`、`pred_texts`（index 0 为 instruction）、`pred_timestamps`。

## 交叉链接

- [LiveCook 论文实体](../../wiki/entities/paper-livecook.md)
- [Streaming Interventions 论文实体](../../wiki/entities/paper-streaming-interventions.md)
- [AI Coach Cooking 2026 入门](../repos/ai-coach-cooking-2026.md)
