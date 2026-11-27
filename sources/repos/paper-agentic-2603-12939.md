# yu2hi13/RoboStream — RoboStream 官方开源仓库档案

- **GitHub 地址：** https://github.com/yu2hi13/RoboStream
- **默认分支：** `main`
- **开源协议：** Apache-2.0 License（仓库自研代码与文档遵循 Apache-2.0，具体边界见 `NOTICE` 与 `THIRD_PARTY_NOTICES.md`；内部 vendored 的第三方仿真/分割/深度组件保留各自原始协议）
- **关联论文：** *RoboStream: Weaving Spatio-Temporal Reasoning with Memory in Vision-Language Models for Robotics*（[arXiv:2603.12939](https://arxiv.org/abs/2603.12939)，ECCV 2026）
- **核对边界：** 静态代码核对，未本地运行（基于仓库元数据、`README.md` 与 `files_sample` 核查，未在本地安装 CoppeliaSim v4.1.0、SAPIEN 或下载 SAM3 / GSNet / Qwen3-VL 权重执行仿真）。

## 核心入口与目录结构（已核实）

- **启动脚本与配置矩阵：**
  - `run_rlbench_generic.sh`：RLBench 长时域评测主启动脚本（在 `rlbench` conda 环境内调用 `simulation/rlbench/run_rlbench.sh` 设置 `LD_LIBRARY_PATH`、Qt 插件与 `PYTHONPATH`，支持 `--mode control_record`、`--vlm-backend remote|qwen|gemma`、`--dry-run` 与 `--check` 预检）。
  - `run_simpler_env_generic.sh`：SIMPLER 真实到仿真（real-to-sim）评测主启动脚本（支持 `--task`、`--max-steps`、`--primary-camera`、`--record`、`--dry-run`）。
  - `configs/rlbench/long_horizon/`：公开 8 项长时域任务配置 JSON（`bridge_between_towers.json`、`cover_bottom_block_with_box.json`、`cover_top_block_with_box.json`、`place_blocks_in_two_containers.json`、`place_blocks_in_two_containers_hard.json`、`stack_five_colors.json`、`stack_three_colors.json`、`unstack_then_stack_four_colors.json`）。
  - `configs/simpler_env/`：公开 6 项 SIMPLER 配置 JSON（Google Robot：`google_robot_pick_coke_can.json`、`google_robot_move_near.json`；WidowX：`widowx_carrot_on_plate.json`、`widowx_put_eggplant_in_basket.json`、`widowx_spoon_on_towel.json`、`widowx_stack_cube.json`）。
- **核心算法与感知服务模块：**
  - `RoboStream/robostream/`：核心规划、感知融合（STF-Tokens）、因果时空记忆图（CSTG）与控制执行模块。
  - `RoboStream/serve/system_prompts.py`：共享感知与推理系统提示词模板。
  - `RoboStream/segmentation/`：可选开放词表分割适配器与 vendored SAM / GroundingDINO 代码。
  - `RoboStream/depth/`：多单目/度量深度估计后端（已核实包含 `depthanythingv2.py`、`metric3dv2.py`、`moge.py` 及其底层实现目录）。
- **仿真与运动/抓取规划栈：**
  - `simulation/rlbench/`：内嵌适配的 RLBench 仿真树及长时域任务定义与启动脚本 `simulation/rlbench/run_rlbench.sh`。
  - `simulation/simpler_env/`：内嵌 SIMPLER / ManiSkill2 仿真树，包含 `GSNet/`（Google Robot 6-DoF 抓取候选）与 `plan/`（OMPL 1.5.2 运动规划 / WidowX plan-PCA 抓取）。
- **环境依赖与文档：**
  - `requirements/rlbench.txt`、`requirements/simpler.txt`、`.env.example`、`docs/configs.md`、`docs/verified_runs.md`、`tests/`。

## 复现与运行依赖边界

- **VLM 后端：**
  - 远程 OpenAI 兼容接口（纯 HTTP 调用，无需 `openai` SDK）：需设置 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 与 `OPENAI_MODEL`（公开验证配置使用 `qwen3-vl-235b-a22b-instruct`）。
  - 本地模型：支持 `--vlm-backend qwen` 或 `gemma`（公开 SIMPLER 验证使用 `Qwen/Qwen3-VL-8B-Instruct` 指定版本）。
- **外部分割与抓取权重（需手动下载并配置环境变量）：**
  - SIMPLER 公开配置强制使用 `sam3==0.1.2`，需自行获取 `facebook/sam3` 的 `sam3.pt` 与 CLIP BPE 词表，并导出 `ROBOSTREAM_SAM3_CHECKPOINT_PATH` 与 `ROBOSTREAM_SAM3_BPE_PATH`；叠加背景合成渲染后即丢弃仿真分割信息，确保 SAM3/GSNet/VLM 不接触特权真值。
  - Google Robot 任务依赖外置 GSNet 权重 `minkuresunet_realsense_tune_epoch20.tar`（`ROBOSTREAM_GSNET_CHECKPOINT_PATH`）及自行编译 PointNet2/KNN/MinkowskiEngine 与 OMPL 1.5.2；WidowX 任务使用 `plan/PCA`，不加载 GSNet 或 OMPL。
- **RLBench 感知范围提示：**
  - 公开 RLBench JSON 配置默认采用仿真器真值感知（GT perception）以单独评估 VLM 长时域规划与闭环控制，不加载 SAM3（因此官方 README 明确指出其不作为纯图像感知结果解读）。
- **仿真与系统约束：**
  - RLBench 严格依赖 **CoppeliaSim v4.1.0**（通过 `COPPELIASIM_ROOT` 指定）及固定提交版本的 PyRep，无头运行需配合 Xvfb 与 `DISPLAY`。

## 与 wiki 的映射

- 对应实体页：[RoboStream: Weaving Spatio-Temporal Reasoning with Memory in Vision-Language Models for Robotics](../../wiki/entities/paper-agentic-2603-12939.md)
