# MiniMind（jingyaogong/minimind）

> 来源归档（GitHub README + 项目页交叉，2026-09-29 核查）

- **标题：** MiniMind
- **类型：** repo
- **作者：** [jingyaogong](https://github.com/jingyaogong)（龚建雅）
- **代码：** <https://github.com/jingyaogong/minimind>
- **项目页：** <https://jingyaogong.github.io/minimind/>
- **许可：** Apache-2.0
- **权重：** [Hugging Face Collection](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)
- **在线体验：** [ModelScope Studio](https://www.modelscope.cn/studios/gongjy/MiniMind)
- **入库日期：** 2026-09-29
- **一句话说明：** 从 **0 用 PyTorch 原生代码** 训练 **~64M 级** 小语言模型的开源教程仓：Pretrain → SFT → LoRA / DPO / PPO·GRPO·CISPO / Agentic RL；主线结构对齐 **Qwen3 / Qwen3-MoE**，宣称单卡 3090 约 **2h + 低成本** 可复现 Zero 对话模型。

## 开源边界（步骤 2.5）

| 已发布 | 备注 |
|--------|------|
| 训练/推理/评测代码 | **已开源**（Apache-2.0） |
| 数据集 jsonl | **已发布**（仓库 `dataset/` 说明 + 下载指引） |
| HF 权重 | **已发布**（多版本 minimind-3 / minimind2 等） |
| 姊妹仓 | [minimind-v](https://github.com/jingyaogong/minimind-v) · [minimind-o](https://github.com/jingyaogong/minimind-o) 等 **独立仓库** |

## 仓库结构（归纳）

| 路径 | 作用 |
|------|------|
| `model/` | `model_minimind.py` 等 Dense + MoE 结构 |
| `trainer/` | `train_pretrain.py`、`train_full_sft.py`、`train_lora.py`、`train_dpo.py`、`train_ppo.py`、`train_grpo.py`、`train_agent.py`、`train_distillation.py` … |
| `trainer/rollout_engine.py` | RLAIF / Agentic RL 生成后端解耦 |
| `scripts/` | `web_demo.py`（Streamlit）、`serve_openai_api.py`、`convert_model.py`、`chat_api.py` |
| `eval_llm.py` | 推理与 `--weight` 前缀加载 |
| `dataset/` | `pretrain_t2t*.jsonl`、`sft_t2t*.jsonl`、`rlaif.jsonl`、`agent_rl*.jsonl` 等 |

## 训练主线（README 归纳）

1. **Pretrain** — `train_pretrain.py` → `pretrain_*.pth`
2. **SFT** — `train_full_sft.py`（含 toolcall 混入主线数据）→ `full_sft`
3. **可选** — LoRA、DPO、PPO/GRPO/CISPO、Agentic RL（`train_agent.py`）、蒸馏
4. **推理** — `eval_llm.py`；OpenAI 兼容 API；llama.cpp / vllm / ollama 生态

**快速复现数据对：** `pretrain_t2t_mini.jsonl` + `sft_t2t_mini.jsonl`（MiniMind Zero）。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [PyTorch](../../wiki/entities/pytorch.md) | 核心算法 **PyTorch 原生** 实现；分布式 DDP / DeepSpeed |
| [深度学习基础](../../wiki/concepts/deep-learning-foundations.md) | 教学向「从 0 理解 LLM 训练栈」 |
| [强化学习 / GRPO](../../wiki/methods/grpo.md) | 仓内 **从 0 实现** PPO/GRPO/CISPO 与 Agentic RL |
| [Transformer](../../wiki/concepts/transformer.md) | 结构对齐 Qwen3 生态的入门对照 |

## 对 wiki 的映射

- [`wiki/entities/minimind.md`](../../wiki/entities/minimind.md)
