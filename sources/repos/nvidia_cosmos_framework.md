# NVIDIA/cosmos-framework

> 来源归档

- **标题：** Cosmos-Framework
- **类型：** repo
- **组织：** NVIDIA
- **代码：** <https://github.com/NVIDIA/cosmos-framework>
- **Homepage：** <https://github.com/NVIDIA/cosmos>
- **Stars：** ~505（2026-09-05）
- **入库日期：** 2026-09-05
- **一句话说明：** Cosmos 模型族的端到端训练 / 推理框架：`cosmos_framework.scripts.train`（SFT）与 `inference`；FSDP/TP/CP/PP、DCP↔safetensors、JSONL/WebDataset/LeRobot 适配器，并附 AGENTS.md 技能。
- **沉淀到 wiki：** 是 → [`wiki/entities/nvidia-cosmos.md`](../../wiki/entities/nvidia-cosmos.md)、[`wiki/entities/cosmos-3.md`](../../wiki/entities/cosmos-3.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源**（与 NVIDIA/cosmos 配套；权重仍走 HF 门控仓） |
| **代码** | <https://github.com/NVIDIA/cosmos-framework> |
| **训练文档** | <https://github.com/NVIDIA/cosmos-framework/blob/main/docs/training.md> |
| **Policy Server** | `docs/action_policy_droid_server.md` |
| **许可** | LICENSE = Other（以仓库 LICENSE 为准；权重走 OpenMDW / NVIDIA Open Model License） |

README 写明：Cosmos 3 cookbooks 的 Framework 后端需要本仓访问（部分文档仍写 `git@github.com:NVIDIA/cosmos-framework.git`）。公开 GitHub 页可浏览；克隆权限以当时组织设置为准。

## 包结构（README）

单一顶层包 `cosmos_framework/`：

- **Training** — 分布式 FSDP / TP / CP / PP；原生 DCP checkpoint；Hugging Face `safetensors` 导入/导出；JSONL / WebDataset / LeRobot 数据集适配器。入口：`cosmos_framework.scripts.train`。
- **Inference** — Diffusers / Transformers / vLLM 后端；离线 batch 与在线 serving（Ray + Gradio）。入口：`cosmos_framework.scripts.inference`。下游 shim 在 `packages/`。

## 安装与硬件（摘要）

- 系统依赖：`curl ffmpeg git-lfs libx11-dev tree wget`
- `uv sync --all-extras --group=cu130-train`（推荐 CUDA 13.0）；或 `cu128-train`
- 要求 `uv >= 0.11.3`
- 推荐 NGC 基座：`nvcr.io/nvidia/pytorch:26.06-py3`
- 随仓 SFT recipe 按 **8× H100 80 GB** 验证；可改 `NPROC_PER_NODE` 与并行度

单卡推理示例：

```bash
python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i "inputs/omni/t2v.json" \
    -o outputs/omni_nano \
    --checkpoint-path Cosmos3-Nano \
    --seed=0
```

SFT 启动示例：`bash examples/launch_sft_vision_nano.sh`。

## Agent Skills

`.agents/skills/`（canonical）镜像到 `.claude/skills/`：`cosmos3-setup`、`cosmos3-codebase-nav`、`cosmos3-inference`、`cosmos3-post-training`、`cosmos3-env-troubleshoot`。

## 对 wiki 的映射

- 平台总览：[`wiki/entities/nvidia-cosmos.md`](../../wiki/entities/nvidia-cosmos.md)
- Cosmos 3 部署 / 微调：[`wiki/entities/cosmos-3.md`](../../wiki/entities/cosmos-3.md)
- 平台仓：[`sources/repos/nvidia_cosmos.md`](./nvidia_cosmos.md)
