# CoVer-VLA 官方代码库｜cover-vla/cover-vla

- **官方仓库：** <https://github.com/cover-vla/cover-vla>
- **配套仓库（PolaRiS / DROID 评测）：** <https://github.com/cover-vla/cover-vla-polaris>
- **项目主页：** <https://cover-vla.github.io>
- **关联论文：** [Scaling Verification Can Be More Effective than Scaling Policy Learning for Vision-Language-Action Alignment](https://arxiv.org/abs/2602.12281)
- **分支 / 许可证 / 开源状态（2026-11-27 核查）：** 默认分支 `main`；GitHub 仓库元数据未显式填充 `license` 字段，仓库内含 `CoVer_VLA/LICENSE`（README 注释徽章标注 MIT）。**部分开源**：已开源 Bridge/SIMPLER 推理与评测管线、环境安装脚本及 Hugging Face 验证器权重（`cover-vla/cover-vla-bridge`、`cover-vla/cover-vla-polaris`）；README To-Do 明确标明验证器训练管线（`Release verifier training pipeline`）与独立验证器服务（`Develop CoVer verifier server`）尚未发布。
- **核对范围：** 静态代码核对，未本地运行。核对对象为缓存中的 GitHub 仓库元数据、`README.md` 与目录文件采样列表（`files_sample`），以及配套仓库 `cover-vla/cover-vla-polaris` 的 `README.md` 与目录列表。

## 仓库结构与可运行入口（来自 README 与 files_sample）

- **环境构建入口：** `CoVer_VLA/scripts/env_simpler_pi.sh`（自动安装 `uv`，创建 `.venv_cover` 虚拟环境，Python 3.10 + CUDA GPU + PyTorch / TensorFlow / SimplerEnv / LeRobot with $\pi_0$ / Bridge Verifier）。
- **验证器权重：** 通过 `huggingface-cli download cover-vla/cover-vla-bridge cover_verifier_bridge.pt --local-dir .` 下载至 `bridge_verifier/cover_verifier_bridge.pt`（约 312 MB）。
- **SIMPLER 评测入口：**
  - 批处理脚本：`CoVer_VLA/inference/experiments/robot/simpler/bashes/test_pi.sh`
  - 主推理脚本：`CoVer_VLA/inference/experiments/robot/simpler/run_simpler_eval_with_openpi.py`（支持 `--task_suite_name` 如 `simpler_widowx` / `simpler_ood`、`--lang_transform_type rephrase`、`--pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge`、`--num_trials_per_task`、`--use_verifier True|False`、`--policy_batch_inference_size 5`、`--lang_rephrase_num 8`）
  - 结果分析脚本：`CoVer_VLA/inference/experiments/robot/simpler/bashes/analyze_success_rate.py --output-dir ./analysis_plots`
- **主要子目录：**
  - `CoVer_VLA/SimplerEnv/`（含 `ManiSkill2_real2sim/` 仿真环境与资产）
  - `bridge_verifier/`（Bridge 环境对比验证器模块与权重存放目录）
  - `INT-ACT/`（INT-ACT 基线代码）
  - `lerobot_custom/`（集成 $\pi_0$ 策略的 LeRobot 定制分支）
- **PolaRiS / DROID 配套入口（`cover-vla/cover-vla-polaris`）：**
  - `third_party/openpi/scripts/serve_policy_with_verifier_Gemini.py`（加载 `cover_verifier_polaris.pt` 与 Gemini 在线生成/缓存改写，暴露 `--port 8000` 策略服务）
  - `eval.sh`、`scripts/eval.py`、`scripts/eval_verifier.py`、`src/polaris/policy/droid_jointpos_verifier_client.py`、`third_party/droid_verifier/model.py`、`third_party/droid_verifier/efficient_ensemble_merged_droid.py`

## 复现边界说明

- **静态代码核对，未本地运行：** 本仓库依赖 CUDA GPU、MuJoCo/OSMesa 渲染及外部 Hugging Face / Google Cloud 权重与 Gemini API（用于在线/启动时改写）。
- **未开源部分：** 主仓库尚未公开 CoVer 对比验证器的训练脚本与数据预处理管线，亦未提供独立的验证器微服务；真机 WidowX 部署控制代码未在主仓库根目录列出。

## 对 wiki 的映射

- [Scaling Verification Can Be More Effective than Scaling Policy Learning for Vision-Language-Action Alignment](../../wiki/entities/paper-agentic-2602-12281.md)
- [论文原文核查](../papers/paper-agentic-2602-12281.md)
