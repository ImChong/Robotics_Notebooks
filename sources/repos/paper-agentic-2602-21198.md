# Reflective-Test-Time-Planning 官方代码仓库核对

- **仓库地址：** <https://github.com/Reflective-Test-Time-Planning/Reflective-Test-Time-Planning>
- **项目主页：** <https://reflective-test-time-planning.github.io>
- **默认分支：** `main`（核对时观测星标 35，提交 7 次）
- **许可证状态：** README 徽章标注 `MIT`（链接指向仓库内 `LICENSE`）；GitHub API 元数据 `license` 字段返回空字符串；子模块 `BEHAVIOR/BEHAVIOR-1K/OmniGibson/` 含独立 `LICENSE` 文件。
- **开源状态：** 部分开源（Partial）
  - **已提供：** `BEHAVIOR`（含改版 `BEHAVIOR-1K/OmniGibson`、环境安装脚本 `setup.sh` 及修改版 `sample_kinematics`）与 `BEHAVIOR-model` 推理入口脚本（仅含样例评测集），以及 `cupboard` 下的 MuJoCo 交互环境脚本。
  - **未完全开放：** README 明确标注 BEHAVIOR 完整评测集将于论文接收后发布；`cupboard` 的模型推理代码尚在清理中，计划于年内发布。
- **核对边界：** 静态代码核对，未本地运行（基于仓库主页目录列表、`files_sample` 采样清单与 `README.md` 完整内容）。

## 观测到的目录与入口文件

仓库顶层目录包含 `BEHAVIOR/`、`cupboard/`、`figures/` 与 `README.md`（README 中另指示由 `BEHAVIOR` 同级切换至 `../BEHAVIOR-model` 执行模型推理）。

- **BEHAVIOR-1K 仿真与基座修改（`files_sample` 已验证路径）：**
  - `BEHAVIOR/setup.sh`：配置 `CUDA_VERSION` 并通过 `--new-env --omnigibson --bddl --dataset --primitives` 安装 OmniGibson、BDDL、BEHAVIOR-1K 数据集（约 50GB）与原语动作；README 说明硬件受限于 OmniGibson，不支持 Blackwell GPU（仅限 3090/4090 等）。
  - `BEHAVIOR/BEHAVIOR-1K/OmniGibson/omnigibson/envs/env_base.py`、`vec_env_base.py`、`metrics_wrapper.py`：OmniGibson 仿真环境基类与评测包装器。
  - `BEHAVIOR/BEHAVIOR-1K/OmniGibson/omnigibson/action_primitives/starter_semantic_action_primitives.py`、`symbolic_semantic_action_primitives.py`、`curobo.py`：语义动作原语与运动规划接口。
  - `BEHAVIOR/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py`、`policies.py`、`datas/lerobot_dataset.py`：策略评测与数据管线模块。
  - README 注明对 `omnigibson.utils.object_state_utils.sample_kinematics` 进行了物体放置稳定性修改。
- **BEHAVIOR-model 推理入口（README 已验证命令行）：**
  - `bash ./inference_hybrid.sh full`、`bash ./inference_storage.sh full`、`bash ./inference_compare.sh full`：分别运行 Hybrid 等任务类别评测；支持参数切换消融配置 `full | vanilla | wo-ria | wo-roa | wo-value | wo-action`。
- **MuJoCo Cupboard Fitting 环境（README 已验证命令与文件）：**
  - `cupboard/` 下通过 `pip install -e .` 安装后，可运行 `cupboard/roboworld/envs/franka_cupboard_interactive.py` 测试交互式橱柜放置环境；推理脚本待后续补全。

## 对应知识页

- [Learning from Trials and Errors: Reflective Test-Time Planning for Embodied LLMs](../../wiki/entities/paper-agentic-2602-21198.md)
