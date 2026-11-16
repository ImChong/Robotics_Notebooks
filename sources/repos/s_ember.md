# S-EMBER 官方源码（facebookresearch）

- **仓库：** https://github.com/facebookresearch/S-EMBER
- **原文：** https://arxiv.org/abs/2607.02689
- **数据：** https://huggingface.co/datasets/facebook/S-EMBER
- **论文源：** [S-EMBER](../papers/s_ember_arxiv_2607_02689.md)
- **Wiki：** [S-EMBER](../../wiki/entities/paper-s-ember.md)
- **入库日期：** 2026-11-16

## 已核对源码入口

| 路径 | 作用 |
|---|---|
| `data/README.md` | 数据/视频布局、JSONL 字段及严格视频截断协议 |
| `scripts/run_mcq_smoke.sh` | 检查视频与 JSONL、过滤已下载样本、调用模型 MCQ runner |
| `scripts/mcq/run_qwen3vl.sh` | Qwen3-VL MCQ 执行入口 |
| `scripts/grounding/run_qwen3vl.sh` | `accelerate launch -m lmms_eval eval --tasks sember_grounding` |
| `lmms_eval/tasks/sember/utils.py` | `sember_doc_to_visual` 把 `video_end` 设为 `question_time`；解析选项与时间区间并聚合指标 |
| `lmms_eval/tasks/sember/sember_grounding.yaml` | grounding 的 JSONL 和指标配置 |
| `lmms_eval/models/simple/qwen3_vl.py` | 裁剪输入视频的 patched 模型 wrapper |
| `tools/judge_grounding.py` | 可选外部 Gemini 语义判分 |

## 开放与复现边界

需要 HF JSONL 和 MP4 视频，并设置 `SEMBER_VIDEO_DIR`；Qwen3-VL 需相容的 flash-attn。默认 grounding 只报告 **temporal IoU**，论文联合评测不能直接由默认脚本宣称完整复现。仓库主要代码标注 CC BY-NC 4.0（第三方除外），数据受 HF 单独条款限制。未运行模型。
