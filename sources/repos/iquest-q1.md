# IQuestLab/IQuest-Q1

> 来源归档（ingest）

- **标题：** IQuest-Q1 — 官方 GitHub（推理与 agent 集成）
- **类型：** repo
- **组织：** 至知创新研究院（IQuest Research）
- **代码：** <https://github.com/IQuestLab/IQuest-Q1>
- **权重：** <https://huggingface.co/IQuestLab/IQuest-Q1>
- **项目页：** <https://iquestlab.github.io/>
- **License：** **iquest-q1**（HF / 仓内 LICENSE 链）
- **入库日期：** 2026-09-29
- **一句话说明：** IQuest-Q1 官方入口仓：模型规格、评测说明、SGLang/vLLM 启动命令（含 MTP/EAGLE 草稿）、OpenAI 兼容 Python 示例，以及 Claude Code / Codex CLI 经网关接入的环境变量模板。

## 开源核查（2026-09-29）

| 项 | 状态 |
|----|------|
| **模型权重** | **已开源（开放权重）** · HF `IQuestLab/IQuest-Q1` |
| **本仓代码** | README + `assets/` 管线图；**无训练脚本**；推理依赖 **SGLang/vLLM** 与 Hub 权重 |
| **Docker** | `iquestlabworkspace/sglang-iquest-q1:cu130`、`iquestlabworkspace/vllm-iquest-q1:cu130` |
| **训练数据 / RL 环境合成栈** | **未公开** |
| **ModelScope 镜像** | **待发布**（项目页） |

## 入口速查（对齐 README）

| 路径 / 链接 | 作用 |
|-------------|------|
| `README.md` | 规格表、Performance 图、Quick Start、Deployment、Claude Code / Codex |
| `assets/IQuest-Q1-Training-Pipeline.png` | 预训练 / 中期训练叙事图 |
| `assets/IQuest-Q1-Post-Training-Pipeline.png` | 合成环境 + 多 harness RL + MOPD + merge |
| HF 权重 | `hf download IQuestLab/IQuest-Q1` |
| SGLang | `python -m sglang.launch_server` + `iquest_q1` parsers |
| vLLM | `vllm serve` + `--tool-call-parser iquest_q1` |

## Model Summary（README 规格表摘录）

| Property | IQuest-Q1 |
| :--- | :--- |
| Total Parameters | 320B |
| Activated Parameters | 15B |
| Transformer layers | 88 |
| Hidden dimension | 3,072 |
| Attention Heads (Q/KV) | 48/8 |
| Hybrid Attention | 3 SWA + 1 FA |
| Sliding Window | 4,096 |
| Experts (Total/Activated) | 256/8 |
| Context length | 524,288 |
| MTP | 2 Independent (train) / 1 Recursive ×8 (infer) |

## 对 wiki 的映射

- [IQuest-Q1 实体页](../../wiki/entities/iquest-q1.md)
- [项目页归档](../sites/iquest-q1-project.md)
- [HF 归档](../sites/huggingface-iquestlab-iquest-q1.md)
