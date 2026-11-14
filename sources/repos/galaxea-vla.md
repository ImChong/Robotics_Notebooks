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
- **通读：** 已完成
- **核对范围：** `_tree.txt`（327 项）、`README.md`（507 行）、`scripts/{run/finetune.sh,finetune.py,serve_policy.py,serve_policy_mem.py}`、`configs/{tokenizer/actioncodec.yaml,model/g05.yaml,model/g05_mem.yaml}`、`src/g05/models/g05/{g05_policy_qwen35.py,g05_model_qwen35.py,g05_policy.py,g05_model.py,inferencer.py,helpers/ar_helper.py}`、`src/g05/tokenizer/models/actioncodec2_v2/wrapper.py`、`experiments/{r1lite/README.md,libero/eval_libero_parallel.py,robotwin/run_robotwin_manager.py}`，及 `LICENSE-G0.5` 历史边界（`38b31e4` / `dc0a1ef`）。
- **一句话说明：** G0.5 训练 / 推理 / 真机与仿真评测入口；当前 `main` 聚焦 G0.5（旧 G0/G0Plus 见 commit `13a16a9`）。**已开源、可运行**（`LICENSE-G0.5` 社区非商用许可；HF `OpenGalaxea/G05` 为 `gated: auto`）。

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
| `huggingface-cli download OpenGalaxea/G05 --local-dir checkpoints` | 权重落到 README 约定布局（HF 仓库为 `gated: auto`，需账号接受协议） |
| `bash scripts/run/finetune.sh <gpus> <task>` | 启动 `torchrun` DDP 调 `scripts/finetune.py` 从 `g05-base` 全参微调（单卡 >70 GB） |
| `python scripts/serve_policy.py --ckpt_path ... eval_embiment=...` | 实为 `--ckpt_path ... eval_embodiment=galaxea_r1lite` WebSocket 策略服务（`PolicyInferencer`）；多帧记忆任务改用 `scripts/serve_policy_mem.py`（维护每相机 `deque` 缓存） |
| `experiments/r1lite` / `r1pro` | 真机零样本客户端（ROS2 bridge + 推理引擎） |
| `experiments/libero` / `robotwin` / `droid` / `so100` | 仿真或对应本体评测入口（如 `eval_libero_parallel.py`、`run_robotwin_manager.py`） |
| `src/g05/models/g05/` | `G05PolicyQwen35` / `G05ModelQwen35`（`proprio_embedder` MLP 连续本体状态注入 + Qwen3.5 视觉记忆）、`ARHelper` / 可选 `FMHelper`、`PolicyInferencer` |
| `src/g05/tokenizer/models/actioncodec2_v2/wrapper.py` | `ActionCodecV2Wrapper` 配合 `configs/tokenizer/actioncodec.yaml`（Block-DCT + 4 码本 RVQ，codebook_size=4096） |
| `configs/model/g05.yaml` / `g05_mem.yaml` | 标准配置与追加混合线性/全注意力 action_expert 的记忆配置 |

**硬件（README）：** 推理 >8 GB（推荐 4090）；全参微调 >70 GB（A100-80 / H20-96）。
**许可边界（README）：** 2026-01-04 前 `Apache-2.0` → 2026-01-04 (`38b31e4`) `Galaxea G0 PLUS License` → 2026-06-16 (`dc0a1ef`) `LICENSE-G0.5`。

## 对 wiki 的映射

- 论文：[`sources/papers/galaxea_g05_arxiv_2608_11739.md`](../papers/galaxea_g05_arxiv_2608_11739.md)
- 项目页：[`sources/sites/opengalaxea-g05.md`](../sites/opengalaxea-g05.md)
- 沉淀 **[`wiki/entities/paper-galaxea-g05.md`](../../wiki/entities/paper-galaxea-g05.md)**
