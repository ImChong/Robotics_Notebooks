# DBtxy/DASH-OPSD

- **标题：** DASH 官方训练与评测仓
- **类型：** repo
- **URL：** <https://github.com/DBtxy/DASH-OPSD>
- **配套论文：** [arXiv:2608.06243](https://arxiv.org/abs/2608.06243) — [`sources/papers/dash_opsd_arxiv_2608_06243.md`](../papers/dash_opsd_arxiv_2608_06243.md)
- **权重：** <https://huggingface.co/dbtxy/DASH-Qwen3-1.7B-LoRA>（另有 4B / 8B）
- **入库日期：** 2026-08-08

## 一句话说明

DASH / OPSD 训练入口：基于 TRL GOLD 与 Zhao et al. OPSD 栈，提供 `opsd_train.py`、`scripts/run_dash_*.sh` 与三档 Qwen3 LoRA。

## 仓库状态（2026-08-08 核查）

| 项 | 内容 |
|----|------|
| default branch | `main` |
| 关键内容 | `opsd_train.py`、`opsd_trainer.py`、`grpo_train.py`、`sft_train.py`、`math_reward.py`、`rlvr_reward.py`、`environment.yml`、`accelerate.yaml`、`scripts/`、`eval/`、`NOTICE`、`CITATION.cff`（作者 TODO） |
| 训练入口 | `opsd_train.py` + `scripts/run_dash_{1b,4B,8B}.sh` |
| 依赖 | conda env `opsd`（torch 2.8+cu128、trl 0.26、vllm 0.11 等）；可选 flash-attn 2.8.3 |
| 数据 | 运行时拉 `siyanzhao/Openthoughts_math_30k_opsd` |
| 许可 | GitHub license 字段空；NOTICE 声明第三方 Apache-2.0 / MIT |

## 关键复现路径

1. `conda env create -f environment.yml && conda activate opsd`
2. （可选）`pip install flash-attn==2.8.3 --no-build-isolation`
3. 运行 `scripts/run_dash_1b.sh`（或 4B / 8B）
4. 评测见 `eval/`；推理可挂 HF LoRA（`PeftModel.from_pretrained`）

## 与 wiki 的关系

- 实体页：[paper-dash-opsd](../../wiki/entities/paper-dash-opsd.md) — 含 `## 源码运行时序图`
