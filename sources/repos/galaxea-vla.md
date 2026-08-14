# OpenGalaxea/GalaxeaVLA

> 来源归档

- **标题：** GalaxeaVLA（G0.5 官方仓）
- **类型：** repo
- **组织 / 作者：** OpenGalaxea / 星海图（Xinghaitu (Beijing) AI Technology）
- **代码：** <https://github.com/OpenGalaxea/GalaxeaVLA>
- **项目页：** <https://opengalaxea.github.io/G05/>
- **论文：** arXiv:2608.11739 — [`sources/papers/galaxea_g05_arxiv_2608_11739.md`](../papers/galaxea_g05_arxiv_2608_11739.md)
- **权重：** <https://huggingface.co/OpenGalaxea/G05>
- **数据集：** <https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset>
- **入库日期：** 2026-08-14
- **一句话说明：** G0.5 训练 / 推理 / 真机与仿真评测入口；当前 `main` 聚焦 G0.5（旧 G0/G0Plus 见 commit `13a16a9`）。**已开源、可运行**。

## 开源核查（2026-08-14）

| 项 | 状态 |
|----|------|
| 仓库可见 | 是（公开；约 730★） |
| License | **G0.5 Community License**（`LICENSE-G0.5`）：学术 / 个人 / 教育 / 内部评估；**不是** Apache/MIT 商用宽松许可。另附 `LICENSE_QWEN3_5.txt` |
| 可运行入口 | **有** — `uv sync`；`scripts/run/finetune.sh`；`scripts/serve_policy.py`；`experiments/{r1lite,r1pro,droid,libero,robotwin,so100}` |
| 权重 | HF `OpenGalaxea/G05`：`g05-base` / droid / libero / robotwin20 / so101 + `action_tokenizer.pt`（全套约 55 GB） |
| 结论 | **已开源**（推理、微调、评测、权重）。预训练全量混合数据不全随仓分发 |

## 入口速查

| 路径 / 命令 | 作用 |
|-------------|------|
| `uv sync` | Python 3.10.x、CUDA 12.8 / PyTorch 2.7.1 环境 |
| `huggingface-cli download OpenGalaxea/G05 --local-dir checkpoints` | 权重落到 README 约定布局 |
| `bash scripts/run/finetune.sh <gpus> <task>` | 从 `g05-base` 全参微调（单卡 >70 GB；`torchrun` DDP） |
| `python scripts/serve_policy.py --ckpt_path ... eval_embodiment=galaxea_r1lite` | WebSocket 策略服务；msgpack 观测 → 反归一化动作 |
| `experiments/r1lite` / `r1pro` | 真机零样本客户端（ROS2 bridge + 推理引擎） |
| `experiments/libero` / `robotwin` / `droid` / `so100` | 仿真或对应本体评测 / 部署 |
| `src/g05/models/g05/` | AR 策略、`ar_helper` / 可选 `fm_helper`、Qwen3.5 视觉记忆 |
| `configs/tokenizer/actioncodec.yaml` | 跨本体 RVQ codec |

**硬件（README）：** 推理 >8 GB（推荐 4090）；全参微调 >70 GB（A100-80 / H20-96）。

## 对 wiki 的映射

- 论文：[`sources/papers/galaxea_g05_arxiv_2608_11739.md`](../papers/galaxea_g05_arxiv_2608_11739.md)
- 项目页：[`sources/sites/opengalaxea-g05.md`](../sites/opengalaxea-g05.md)
- 沉淀 **[`wiki/entities/paper-galaxea-g05.md`](../../wiki/entities/paper-galaxea-g05.md)**
