# OxyGen 官方代码：面向多任务并行 VLA 推理的统一 KV Cache 管理框架

> 来源归档（repo）；核查日：2026-11-27；**已开源**（Apache-2.0 协议，Gemma 组件受 `LICENSE_GEMMA.txt` 约束）

- **仓库：** <https://github.com/air-embodied-brain/OxyGen>
- **项目主页：** <https://air-embodied-brain.github.io/OxyGen/>
- **默认分支：** `main`（快照可见 22 Stars）
- **对应论文：** [OxyGen（arXiv:2603.14371v5）](../papers/paper-agentic-2603-14371.md)（NeurIPS 2026）
- **许可：** Apache-2.0 License（`LICENSE`），Gemma 相关组件遵循 `LICENSE_GEMMA.txt`，扩展第三方组件说明见 `model_extensions/THIRD_PARTY.md` 与 `model_extensions/licenses/LICENSE_XIAOMI`
- **核对范围：** 基于仓库主页、`README.md` 预览及文件目录清单（`files_sample`）进行静态结构与入口核对；未本地克隆、下载权重或运行脚本（`静态代码核对，未本地运行`）。
- **通读：** 已完成

## 开源范围与关键路径

| 路径 | 职责与内容 |
|---|---|
| `src/openpi/` | 面向 $\pi_{0.5}$ 的核心模型与统一 KV Cache 管理器改造实现（含 JAX 与 PyTorch 双后端） |
| `experiments/common/setup.py` / `workload.py` | 策略实例化（`create_policy`）与合成观测构造（`create_synthetic_observation`） |
| `experiments/baseline/runner.py` | 顺序隔离执行（Baseline）测试入口与网格搜索脚本（`grid_search.py`） |
| `experiments/parallel_mps/runner.py` | 基于 CUDA MPS 的进程级并行隔离执行对比入口 |
| `experiments/shared_kv/runner.py` | 仅开启跨任务 KV 共享（`Ours w/o Batching`）消融运行入口 |
| `experiments/continuous_batching/runner.py` | 完整 OxyGen（跨任务 KV 共享 + 跨帧连续批处理）运行入口 |
| `experiments/run_experiments.py` / `run_overhead.py` / `run_workload_sweep.py` | 端到端实验、KV 管理开销剖析与变到达率负载扫描主控脚本 |
| `experiments/analysis/` | 指标聚合（`aggregate_metrics.py`）、加速比统计与论文图表复现脚本（`plot_all.py`） |
| `language_adaptation/libero/` | LIBERO 文本记忆生成任务适配：LoRA 微调（`train.py`）、评测（`evaluate.py`）、服务（`serve_adapter.py`）及监督标注流水线（`supervision/`） |
| `model_extensions/run.py` / `src/oxygen_models/` | Xiaomi-Robotics-0、StarVLA（PI_v3 / GR00T / OFT 多专家）及 Qwen3.5（0.8B–9B）多框架共享前缀与连续批处理扩展（含 `common/cache.py`、`starvla/batching.py`、`starvla/runtime.py`、`qwen35/runtime.py` 等） |
| `packages/openpi-client/` | 远端 WebSocket 推理客户端与智能体控制循环运行时（`websocket_client_policy.py`、`action_chunk_broker.py`） |

## README 给出的复现入口与依赖关系

```bash
# 1. 环境准备（Python 3.11 + uv，跳过 LeRobot 大文件 LFS 拉取）
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 2. 快速推理验证（首次运行自动下载 pi05_base 至 ~/.cache/openpi）
uv run python - <<'PY'
from experiments.common.setup import create_policy
from experiments.common.workload import create_synthetic_observation

policy = create_policy("pi05_o2_arx")
obs = create_synthetic_observation("pick up the red cup", seed=0, policy_config="pi05_o2_arx")
out = policy.infer_text(obs, max_decoding_steps=20)
print(out["text"])
PY
```

- **上游与配套资源：** 核心代码基于 Physical Intelligence 的 `openpi` 构建，并参考社区复现 `BrunoFANG1/openpi_subtask_generation` 实现初始 $\pi_{0.5}$ 语言解码基线；LIBERO 文本记忆适配器与标注数据另发布于 Hugging Face（`xxxxyu/oxygen-pi05-textual-memory-lora` 与 `xxxxyu/libero-textual-memory-annotations`）。`model_extensions/` 为 Xiaomi-Robotics-0、StarVLA 与 Qwen3.5 提供了独立锁文件环境（`environments/*.lock.txt`）与模型下载脚本（`download_models.py`）。
- **复现边界：** 静态代码核对，未本地运行。运行完整实验需 Ubuntu 22.04+、NVIDIA GPU（推理 >8 GB 显存，完整实验建议 >16 GB；原文在 RTX 4090 / Jetson AGX Thor 验证）及下载 `openpi` 权重；论文 §4.7 的人形机器人硬件驱动与底层控制接入未在本仓库文件中直接提供。

## 映射

- [原论文归档](../papers/paper-agentic-2603-14371.md)
- [唯一主 Wiki 节点](../../wiki/entities/paper-agentic-2603-14371.md)
