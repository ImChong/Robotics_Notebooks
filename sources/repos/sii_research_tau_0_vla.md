# sii-research/tau-0-vla

> 来源归档

- **标题：** τ₀-VLA（官方实现）
- **类型：** repo
- **组织：** sii-research（上海创智学院、智元机器人 Finch 团队、香港中文大学）
- **代码：** <https://github.com/sii-research/tau-0-vla>
- **项目页：** <https://tau0-vla.github.io/>
- **论文：** <https://arxiv.org/abs/2608.16885>
- **Hugging Face 权重集合：**
  - 低层预训练：<https://huggingface.co/sii-research/tau-0-vla>（Apache-2.0）
  - LIBERO 后训练：<https://huggingface.co/sii-research/tau-0-vla-libero>
  - 高层 Proposal：<https://huggingface.co/sii-research/tau-0-vla-proposal>（Qwen3.5-9B full checkpoint）
  - 高层 World Model：<https://huggingface.co/sii-research/tau-0-vla-world-model>（Step1X-Edit-v1p2 rank-64 robotics LoRA）
- **入库日期：** 2026-08-19
- **核对日期：** 2026-11-14
- **通读：** 已完成
- **核对范围：** 检视镜像仓库 `sii-research__tau-0-vla` 全量 216 文件目录树（`_tree.txt`）、根 `README.md`、`high_level/{README.md,proposal/README.md,world_model/README.md}`、`configs/libero/README.md`、`deploy/{README.md,policy.py,server.py,openloop.py}`、`scripts/{train.sh,proposal_to_world_model.py,validate_gpu_release.sh}`、`src/tau0_vla/trainer/train.py`、`src/tau0_vla/models/vision_language_action_models/tau0_vla/modeling_tau_vla.py` 及 `src/tau0_vla/high_level/proposal/{runtime.py,infer.py,serve.py,finetune.py}` 与 `high_level/world_model/src/tau0_world_model/infer.py`；并全仓 `rg` 确认尚未包含 Value/Reflective/Beam Search 代码。
- **一句话说明：** τ₀-VLA 官方代码库（Apache-2.0，追踪至 `2026.09.21` 阶段发布）：除低层 generalist VLA 后训练、真机 joint-control serving（`deploy.server`）及 LIBERO EEF 仿真评测（平均 `97.35%`）外，已于 2026 年 9 月增发高层 **Proposal（Qwen3.5-9B）** 与 **World Model（Step1X-Edit LoRA）** 推理/微调栈；**Value 模型、Reflective 模型、置信路由与完整 TTC Beam Search 闭环尚未释放**。

## 真实开源进度与组件边界（核对至 2026.09.21 仓库快照）

| 发布日期（README News） | 组件与 HF 权重 | 核心真实入口与文件 | 状态与使用边界 |
|---|---|---|---|
| **2026.07.27** | 低层通用 VLA 预训练权重 & 后训练栈（`sii-research/tau-0-vla`） | `scripts/train.sh` → `src/tau0_vla/trainer/train.py`；`modeling_tau_vla.py`（`Tau0VLAModel`、`Qwen35vlWithExpertModel`、`FlowMatching`）；`deploy/{policy.py,server.py,openloop.py}` | **已发布（Apache-2.0）**；硬件 `deploy.server` 严格限定 **joint-control**（`_require_public_v1_joint_only` 显式拒绝 EEF 路由） |
| **2026.09.20** | LIBERO EEF 后训练与仿真评测（`sii-research/tau-0-vla-libero`） | `configs/libero/{train.yaml,data.py}`（40D 激活槽 `0:9` + `18`）；`deploy/libero_server.py`（`POST /act_libero`）+ `deploy/libero/main.py` | **已发布（仅仿真）**；60k 步后训练 4 套件均值 **97.35%**（Spatial `97.40%`、Goal `98.20%`、Object `98.80%`、Long `95.00%`） |
| **2026.09.21** | 高层 Proposal（`sii-research/tau-0-vla-proposal`） | `src/tau0_vla/high_level/proposal/{runtime.py,infer.py,serve.py,finetune.py}` | **已发布**；Qwen3.5-9B，`full_qa` / `subtask_only` 模式，支持 `<think>/<memory>/<subtask>` 6 特殊 token，head→left→right 480×640，HTTP `POST /predict`（port 10089）及 assistant-only LoRA 微调 |
| **2026.09.21** | 高层 World Model（`sii-research/tau-0-vla-world-model`） | `high_level/world_model/src/tau0_world_model/{convert.py,infer.py,finetune.py}`；`scripts/proposal_to_world_model.py` | **已发布**；基于 `stepfun-ai/Step1X-Edit-v1p2` 的 rank-64 robotics LoRA（28 步去噪、CFG=6.0、512 尺寸，须独立 `.venv-world-model` 环境） |
| **未释放** | **Value 模型 ($V$)、Reflective 模型 ($F$)、置信路由 ($g_t$) 与 TTC Beam Search** | 全仓 `rg` 无 `reflect` / `value_model` / `beam_search` / `adaptive_rout` 实现 | **待发布**；当前仅可通过 `scripts/proposal_to_world_model.py` 串联 Proposal 输出与 World Model 生成目标图，尚无官方闭环 SEARCH 与 1 Hz/30 Hz 异步调度器 |

## 实测工程要点（源码验证）

- **底层实现差异（`modeling_tau_vla.py`）：** VLM 骨干在 `linear_attention` 层采用 GatedDeltaNet、动作专家全层用 `full_attention`，仅在 VLM `full_attention` 层做联合注意力；具备 `Gated Query Attention`（`q_proj -> [Q, gate]`）、25% 部分 RoPE 与 `AdaRMSNorm`。
- **训练与数据契约（`train.py`、`deploy/README.md`）：** `state_dim` / `action_dim` / `action_horizon` 严禁手填于 YAML，强绑定 `data_args.config_name` 对应 `@register_config`；推理端由 checkpoint 内 `finch_data_spec/<route>/` 与 `Tau0VLAPolicy`（`encode_payload` / `restore_action`）保证训练/部署零漂移。
- **安全传输协议（`deploy/server.py`、`deploy/wire.py`）：** 硬件 server 默认仅绑 `127.0.0.1` 且废除 `pickle`，采用带流式体积上限和 `allow_pickle=False` 校验的 `.npz` 封装（`POST /act` 与 `/act_lerobot_bytes`）。
- **环境隔离要求：** 低层（Python 3.11 / PyTorch 2.7.1 / CUDA 12.8）、Proposal（Python 3.12 `.venv-proposal`）、World Model（Python 3.12 `.venv-world-model`）与 LIBERO 仿真客户端（Python 3.10 / MuJoCo 3.2.3）须分环境部署，不可混装。

## 对 wiki 的映射

- 沉淀并修正 [`wiki/entities/paper-tau0-vla.md`](../../wiki/entities/paper-tau0-vla.md)（重点更新 2026-09-20/21 的 LIBERO 与 Proposal+WorldModel 开源状态、完整真实入口与尚未开源模块边界）。
- 论文全文对照见 [`sources/papers/tau0_vla_arxiv_2608_16885.md`](../papers/tau0_vla_arxiv_2608_16885.md)。
