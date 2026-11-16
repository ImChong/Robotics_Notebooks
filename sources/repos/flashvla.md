# FlashVLA（z-lab/flashvla）

> 来源归档

- **标题：** FlashVLA
- **类型：** repo
- **来源：** z-lab（论文作者 UCSD / MIT）
- **链接：** <https://github.com/z-lab/flashvla>
- **论文：** <https://arxiv.org/abs/2608.27384>
- **博客：** <https://z-lab.ai/projects/flashvla/>
- **权重集合：** <https://huggingface.co/collections/z-lab/flashvla>
- **LIBERO 权重：** <https://huggingface.co/z-lab/flashvla-pi05-libero>
- **RoboTwin 权重：** <https://huggingface.co/z-lab/flashvla-pi05-robotwin>
- **许可：** Apache-2.0（`LICENSE` 核对）
- **入库日期：** 2026-08-29
- **核对更新：** 2026-11-14
- **通读：** 已完成
- **核对范围：** 仓库完整目录树（`_tree.txt`，94 文件）、`README.md`、`LICENSE`、`flashvla/async_manager.py`、`flashvla/train.py`、`flashvla/datasets/flashvla_dataset.py`、`flashvla/policies/pi05/modeling_pi05_flashvla.py`、`flashvla/policies/smolvla/modeling_smolvla_flashvla.py`、`sim_eval/libero/{eval.sh,eval.py}`、`sim_eval/robotwin/overlay/script/policy_model_server.py`、`sim_eval/robotwin/policy/pi05_flashvla/{deploy_policy.py,pi05_flashvla_model.py}`、`benchmarks/benchmark_latency.py`、`train/train.sh`、`train/configs/pi05/{libero,robotwin}/pi05_flashvla.yaml`。
- **一句话说明：** 流匹配 VLA 的流式动作解码官方实现：覆盖 \(\pi_{0.5}\)/\(\pi_0\)/SmolVLA/LingBot-VLA 多缓冲微调、异步调度器、LIBERO/RoboTwin 仿真评测与延迟基准（基于 LeRobot 与 VLASH）。
- **沉淀到 wiki：** [`wiki/entities/paper-flashvla.md`](../../wiki/entities/paper-flashvla.md)

---

## 仓库架构与源码入口

| 组件 | 路径 / 入口 | 核对实现细节 |
|------|-------------|--------------|
| 环境安装 | `environment.yml` | 安装核心 `flashvla` 环境（LIBERO/RoboTwin 仿真需额外在 `sim_eval/` 下配仿真依赖） |
| 多缓冲数据集 | `flashvla/datasets/flashvla_dataset.py` | `FlashVLADataset.__getitem__` 为同一观测 `o_t` 生成 \(N\) 组冷启动配置（第 `k` 组含 `k*C` 真实动作 + `(N-k)*C` pad）并拼接为 `[N*H_cfg, action_dim]` |
| 核心流式模型 | `flashvla/policies/pi05/modeling_pi05_flashvla.py` | `PI05FlashVLAModel` 实现 `_build_per_chunk_time`（每 slot 独立 Beta 采样）、`_build_shared_obs_mask`（跨配置隔离 + chunk 级因果掩码）、`forward_shared_observation`、`_cold_start` 与可编译成 CUDA Graph 的 `_steady_streaming` |
| 跨架构适配 | `flashvla/policies/smolvla/modeling_smolvla_flashvla.py` 等 | 为 SmolVLA 引入 `LlamaAdaRMSNorm` FiLM 调节层；`policies/pi0` 与 `policies/lingbot` 各自提供 FlashVLA 封装 |
| 异步动作调度器 | `flashvla/async_manager.py` | `AsyncStreamingActionManager` 封装 `overlap_steps`（`d`）、`warmup()`（预捕获 `_steady_streaming` CUDA Graph 后重置）、非阻塞 GPU 派发与 CPU 动作缓存（另支持可选 `skip_stale_actions` 重对齐开关） |
| 训练入口 | `train/train.sh` → `flashvla/train.py` | 支持单机/多卡 FSDP2；`apply_flashvla_training_init` 支持按配置对动作专家归一化层做 `ae-norm-zero` 初始化（RoboTwin 多任务默认开启 `flashvla_init`） |
| LIBERO 评测 | `sim_eval/libero/eval.sh` → `eval.py` | 传 `--inference_overlap_steps`、`--policy.compile_model`、`--policy.fuse_qkv` / `fuse_gate_up` 调用 `AsyncStreamingActionManager` 跑四套件 rollout 并统计逐步时间 |
| RoboTwin 2.0 | `sim_eval/robotwin/policy/pi05_flashvla/` | `eval_server.sh`（`flashvla` 环境 + `policy_model_server.py` + `PI05FlashVLAModel` Socket 服务）与 `eval_client.sh`（SAPIEN 客户端）跨环境通信 |
| 延迟基准 | `benchmarks/benchmark_latency.py` | `--config_path=benchmarks/configs/latency_flashvla.yaml`（可调 `--num_views=1/2/3`），10 次 warm-up 后统计 100 样本前向耗时与分段剖析 |

## 开源边界（截至 2026-11-14）

- **已开源（Apache-2.0）：** 完整流式解码核心库、4 个主流流匹配 VLA 适配（\(\pi_{0.5}\)、\(\pi_0\)、SmolVLA、LingBot-VLA）、多缓冲数据集与 FSDP2 训练脚本、LIBERO / RoboTwin 2.0 仿真评测及延迟基准脚本均在仓内；LIBERO 与 RoboTwin 2.0 的 \(\pi_{0.5}\) 微调权重已在 Hugging Face 公开。
- **未随仓发布：** 真机 Franka 机械臂硬件控制闭环（论文 §5 部署）及 Gello 遥操作演示数据集未包含在仓内（仅 `README.md` 提及核心环境可用于真机部署，`sim_eval/` 外无 `real_robot/` 脚本）。
