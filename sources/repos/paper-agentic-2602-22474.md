# Uncertainty-Aware Policy Steering (UPS) 官方代码仓库归档

- **论文实体：** [When to Act, Ask, or Learn: Uncertainty-Aware Policy Steering](../../wiki/entities/paper-agentic-2602-22474.md)
- **论文归档：** [paper-agentic-2602-22474.md](../papers/paper-agentic-2602-22474.md)
- **官方 GitHub 仓库：** https://github.com/CMU-IntentLab/uncertainty_aware_policy_steering （默认分支 `main`，截至核对日仓库未附带 `LICENSE` 文件）
- **配套开源权重与数据集（Hugging Face）：**
  - 扩散策略与残差策略权重：`https://huggingface.co/jzyuan04/ups_nutassembly_diffusion_checkpoints`
  - Dreamer-v3 世界模型权重：`https://huggingface.co/jzyuan04/ups_nutassembly_wm_checkpoint`
  - 基础策略演示集 / 世界模型数据 / 残差 rollout 数据：`https://huggingface.co/jzyuan04/ups_nutassembly_base_policy_demos`、`https://huggingface.co/jzyuan04/ups_nutassembly_wm_data`、`https://huggingface.co/jzyuan04/ups_nutassembly_wm_data_residual`
- **开源状态：** 已开源（包含 RoboMimic `NutAssembly` 仿真环境的完整训练、世界模型、VLM 提示词模板、共形预测校准与残差持续学习流水线；仓库说明真机实验遵循同一架构，但为简化复现本仓仅面向仿真运行）
- **核对边界：** 静态代码核对，未本地运行（基于仓库 `README.md` 与已抽样文件列表）。

## 仓库结构与已核对文件（`CMU-IntentLab/uncertainty_aware_policy_steering`）

已核对根目录及子模块文件树与 README 说明如下：

- **环境与子模块配置：** `.gitmodules`、`environment.yml`、`README.md`（说明依赖 `robosuite` 仿真环境与 `diffusion4robotics` 子模块，以及 `robobuf`、`r3m`）。
- **Dreamer-v3 世界模型模块（`model_based_irl_torch/`）：**
  - `model_based_irl_torch/__init__.py`、`model_based_irl_torch/setup.py`
  - `model_based_irl_torch/dreamer/`：`dreamer.py`、`models.py`、`networks.py`、`tools.py`
  - `model_based_irl_torch/common/`：`constants.py`、`ood_utils.py`、`utils.py`
  - `scripts/train_wm_real_data.py`（配合 `wm_nutassembly_config.yaml` 训练/重训世界模型）
- **策略引导与 VLM 校准封装（`policy/`）：**
  - `policy/sim_policy.py`：基础/残差扩散策略封装
  - `policy/wm_predictor.py`：交错动作生成与世界模型长视距推演封装
  - `policy/vlm_video_translator.py`：调用 VLM（默认 `Gemini 3 flash preview`）将解码推演视频转述为文本选项，并与共形预测验证器对接
- **VLM 提示词与上下文样例（`vlm_query/`）：**
  - 参考图像：`in_context_images_grasp/gt_grasp_{failure,success}.png`、`in_context_images_place/gt_{handle_facing_left,handle_facing_right,place_failure}.png`
  - 提示词模板：`grasp_prompt_with_images.txt`、`place_prompt_with_images.txt`、`possible_intents.txt`、`score_intents_{logit,self}.txt`、`score_modes_{cot,intent,vanilla}_{logit,self}.txt`
- **基础扩散策略、干预分类器与残差策略训练（`diffusion4robotics/`，README 核对）：**
  - `combine_traj_to_buffer.py`（支持 `--base`、`--classifier`、`--residual` 三种缓冲转换）
  - `compute_class_weights.py`（计算干预分类器正样本权重 `agent.pos_weight`）
  - `finetune.py`（支持 `agent=diffusion_unet`、`--config-name=finetune_classifier`、`--config-name=finetune_residual_policy`）
- **校准与全流程运行入口（`scripts/`，README 核对）：**
  - `scripts/generate_calibration_data.py` 与 `scripts/run_calibration`：生成校准转述并计算经验分位数阈值 $\hat{q}$（官方预训练权重对应 `qhat=0.7`）
  - `scripts/run_ups`：串联世界模型、VLM 验证/澄清与残差纠偏采集（支持 `--device spacemouse` 或 `keyboard`）
  - `scripts/process_corrections`：处理纠偏轨迹并计算残差动作标签
