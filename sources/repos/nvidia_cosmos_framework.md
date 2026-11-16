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
- 当前固定版要求 `uv==0.11.29`
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

## 固定版本源码核读（2026-11-13）

- **版本：** [1f1737931214166d5e7b253def319c562cf7a1eb](https://github.com/NVIDIA/cosmos-framework/tree/1f1737931214166d5e7b253def319c562cf7a1eb)
- **核对范围：** 根README与docs/action_policy_droid_server.md全文；action_policy_server_robolab.py的main/serve/Service构造、_build_transform/_build_sample/_generate/_format_outputs/infer完整函数；inference/action.py全文。未逐读全部MoT内核或分布式worker实现。
- **通读：** 已完成
- 官方文档入口python -m cosmos_framework.scripts.action_policy_server_robolab；默认单进程，CFG并行需恰好两个rank，仅rank0持有socket。Service通过OmniInference.create恢复模型/训练数据配置，锁定请求顺序。
- _build_sample拼图像canvas和可选state/history，按joint_pos（8D）或midtrain（10D）选择表示；_transform使用checkpoint数据配置。model.generate_samples_from_batch联合采样，_format_outputs去history并翻转gripper约定，midtrain另积分相对EEF，decode_video可关但latent视频仍采样。
- 参数与mode契约必须一致：inference/action.py当前unified Bimanual导出要求schema-5 sidecar且只支持记录动作forward dynamics；不能将它当通用WAM policy。LIBERO/RoboCasa部分原生宽度按dataset决定。
- 当前训练、DPO/GRPO/PPO等为可用框架/recipe矩阵，不代表任意模型/算法组合已验证。基础权重、gated Guardrail、第三方依赖条款分别核对。静态阅读，未下载模型/申请访问或运行机器人。
