# VERITAS 官方代码库｜princeton-prism/veritas

- **官方仓库：** <https://github.com/princeton-prism/veritas>
- **项目主页：** <https://veritas-improvement.github.io>（网页源码仓库：<https://github.com/veritas-improvement/veritas-improvement.github.io>）
- **关联论文：** [Visual Verification Enables Inference-time Steering and Autonomous Policy Improvement](https://arxiv.org/abs/2606.18247)（RSS 2026）
- **分支 / 许可证 / 开源状态（2026-11-27 核查）：** 默认分支 `main`；许可证 `MIT`（含 `LICENSE` 文件）。**已开源**：发布基于 `open-pi-zero` 改编的 $\pi_0$ / PaliGemma 模型实现、SimplerEnv WidowX 评测管线、三种视觉验证器（`veritas_verifier`、`vlm_constraints_verifier`、`heuristic_verifier`）、验证成功轨迹采集器（`verified_data_collector.py`）、目标追踪服务端（`tracking_server.py`）与微调训练入口（`train.py`）。注：arXiv HTML 页首 `\code` 字段存在拼接笔误（`prism-princeton/projectgithub.com`），`README.md` 克隆示例亦提及旧路径 `robo-alex/veritas.git`，当前有效公开仓库为 `princeton-prism/veritas`。
- **核对范围：** 静态代码核对，未本地运行。直接核对了 `princeton-prism/veritas` 默认分支 `main` 的 `files_sample`（根目录脚本、配置、`src/` 与 `scripts/` 模块清单）及 `README.md` 安装、权重下载、最小示例与评测说明。

## 仓库结构与可运行入口（来自文件清单与 README 核对）

- **顶层评测、运行与配置入口：**
  - `eval_with_veritas_verifier.sh`：一键评测脚本，在配置 `GEMINI_API_KEY` 后依次在 SimplerEnv 的 4 项 WidowX 任务（`widowx_put_eggplant_in_basket`、`widowx_stack_cube`、`widowx_carrot_on_plate`、`widowx_spoon_on_towel`）上运行 VERITAS 验证器，每项任务评测 240 个回合（24 变体 × 10 次试验），并将 `.mp4` 视频保存至 `logs/verifier_eval/eval_<task>/videos/`。
  - `scripts/run.py`、`scripts/set_path.sh`、`scripts/try_checkpoint_in_simpler.py`：训练/评测统一启动脚本、路径环境变量设置与 SimplerEnv 权重快速验证脚本；`scripts/tests/` 下含 `oxe.py`、`run_paligemma.py`、`sampling.py`、`simpler.py` 单元测试。
  - `config/eval/bridge.yaml`、`config/train/bridge.yaml`、`config/bridge_statistics.json`：Bridge / SimplerEnv 评测与微调配置及动作归一化统计量。
  - `pyproject.toml`、`uv.lock`、`.gitmodules`：基于 `uv`（默认 CUDA 13）的依赖管理，以及子模块 `Grounded-SAM-2`（含 `checkpoints/download_ckpts.sh`）。
- **智能体、验证器与数据飞轮核心（`src/agent/`）：**
  - `src/agent/eval.py`（`EvalAgent`）：仿真/评测核心代理。支持关闭验证器直接运行基线策略（`use_veritas_verifier = False`、`verifier_samples = 1`），或开启 Best-of-$N$（默认 `5` 候选动作块）采样与验证选择；配合 `src/agent/eval_verifier_utils.py` 与 `src/agent/utils/verifier_factory.py`、`verifier_helpers.py` 实例化具体验证器。
  - `src/agent/veritas_verifier/`（`veritas_verifier.py`、`gemini_client.py`、`schemas.py`）：VERITAS 主验证器实现，回合初通过 Gemini 生成像素路径点方案（`(u, v)`、`tol_px`、`min_hold`），随后将候选动作块预测的末端位置投影到像素平面评分，并通过迟滞驻留、跨段跳过（transit-skip）与停滞跳过（stalled-skip）推进当前路径点。
  - `src/agent/vlm_constraints_verifier/`（`vlm_constraints_verifier.py`、`gemini_client.py`、`schemas.py`）：基于相对物体角色引用的动态约束验证器。
  - `src/agent/heuristic_verifier/`（`verifier.py`、`fsm.py`、`features.py`、`primitives.py`、`tracker_adapter.py`、`gemini_client.py`、`schemas.py`）：基于有限状态机（FSM）与几何运动特征的启发式验证器。
  - `src/agent/object_tracker.py`、`src/agent/tracking_server.py`、`src/agent/tracking_server_core.py`：封装 Grounded-SAM-2（SAM2 `hiera_large` + Grounding-DINO `base`）的物体检测与在线追踪服务。
  - `src/agent/verified_data_collector.py`：记录验证器引导下成功的执行轨迹，沉淀用于离线自改进的数据集。
  - `src/agent/train.py`、`src/agent/dataset.py`、`src/agent/model_averaging.py`、`src/agent/env_adapter/simpler.py`：离线行为克隆微调入口、数据集封装、模型权重平均与 SimplerEnv 环境适配器。
- **数据管线与 VLA 生成器模型（`src/data/` & `src/model/`）：**
  - `src/model/vla/pizero.py`、`joint_model.py`、`mixture.py`、`modules.py`、`processing.py` 与 `src/model/paligemma/`：基于 `open-pi-zero` 与 `pytorch-paligemma` 的 $\pi_0$（PaliGemma + 流匹配动作专家，10 步流匹配采样）生成器实现，以及 `lora.py`、`kv_cache.py`。
  - `src/data/`：包含 `oxe/`、`dlimp/` 转换管线、`dataset.py`、`dataset_torch.py`、`obs_transforms.py` 与 `traj_transforms.py`。

## 复现边界说明

- **静态代码核对，未本地运行：** 本次仅核对了仓库文件目录结构、`README.md` 说明与调用示例，未在本地执行 `uv sync`、权重下载或运行仿真评测。
- **运行前提与外部依赖：**
  1. 需递归克隆仓库（含 `Grounded-SAM-2` 子模块）并使用 `uv sync` 配置环境（`README.md` 注明已在 CUDA 13 + NVIDIA RTX PRO 6000 Blackwell 与 RTX 5090 测试）。
  2. 需自行下载 `google/paligemma-3b-pt-224` 权重、Grounded-SAM-2 检查点（`bash Grounded-SAM-2/checkpoints/download_ckpts.sh`）以及 HuggingFace 上的 Bridge 策略检查点（`allenzren/open-pi-zero` 的 `bridge_uniform_step19296_2024-12-26_22-31_42.pt`）。
  3. VERITAS 与基于 VLM 的验证器运行需配置有效的 `GEMINI_API_KEY`。
  4. 公开仓库重点提供 Bridge / SimplerEnv 仿真评测与训练配置（`config/eval/bridge.yaml`、`config/train/bridge.yaml`），真机 FR3-DROID 硬件驱动未直接包含在上述 Bridge 示例配置中。

## 对 wiki 的映射

- [Visual Verification Enables Inference-time Steering and Autonomous Policy Improvement](../../wiki/entities/paper-agentic-2606-18247.md)
- [论文原文核查](../papers/paper-agentic-2606-18247.md)
