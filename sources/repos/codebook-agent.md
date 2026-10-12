# Codebook Agent（jinxiy1104/CodebookAgent）

> 来源归档

- **标题：** CodebookAgent
- **类型：** repo
- **来源：** 论文作者（UCLA 等）
- **链接：** <https://github.com/jinxiy1104/CodebookAgent>
- **代码：** https://github.com/jinxiy1104/CodebookAgent
- **论文：** <https://arxiv.org/abs/2609.02264>；归档 [codebook_agent_arxiv_2609_02264.md](../papers/codebook_agent_arxiv_2609_02264.md)
- **入库日期：** 2026-10-12（克隆 depth 1，tip 56175cb，2026-08-03）
- **许可：** 仓库根目录未见 LICENSE 文件（未核实声明）
- **一句话说明：** Codebook Agent 的官方实现：VQ 拓扑码本、码预测器、MLP 代理与六个基准的复现脚本。
- **沉淀到 wiki：** [`wiki/entities/paper-codebook-agent-topology-design.md`](../../wiki/entities/paper-codebook-agent-topology-design.md)

## 目录与入口

- `codebook_agent/topology/`：`vqvae_model.py`（VQ 自编码器）、`code_predictor.py`、`mlp_proxy.py`、`codebook_framework.py`（`VQTGFramework`，含 `train_vqvae`、`train_code_predictor`）；另含扩散基线（`diffusion_model.py`、`graph_transformer.py` 等）。
- `experiments/run_gsm8k.py`、`run_math.py`、`run_mmlu.py`、`run_humaneval.py`：每基准入口；`--mode GTD --gtd_generator vqvae --gtd_proxy mlp` 即 Codebook Agent。
- `scripts/<benchmark>/run_*.sh`：串联三阶段（收集记录、训练、评测）；`scripts/gsm8k/run_gsm8k_2x2.sh` 跑生成器×代理 2×2 矩阵。
- 依赖：Python 3.11、torch 2.3.0（CUDA 12.1，CPU 亦可）、OpenAI 兼容 API（`template.env`）。

## 开放范围

- 已开源训练与评测代码；数据集不随仓库提供，需自行下载并抽取 test200 子集；未见预训练检查点发布（目录中无 checkpoints）。
