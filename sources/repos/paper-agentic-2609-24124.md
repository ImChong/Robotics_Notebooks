# ActiveArena 官方代码仓库归档

- **论文 / 项目条目：** `ActiveArena: Benchmarking and Understanding Active Perception in Robotic Manipulation`（[arXiv:2609.24124](https://arxiv.org/abs/2609.24124)，项目页：https://leeibo.github.io/ActiveArena ）
- **核查边界：** 静态代码核对（基于官方 GitHub 仓库元信息、目录样本 `files_sample` 与 `README.md`），未在本地运行训练或仿真评测
- **开源状态：** 已开源仿真环境、35 任务基准、数据采集、固定种子评测脚本以及基于 StarVLA 的策略训练/推理代码；Hugging Face 公开了 LeRobot 训练集、资产包与 3 组 OFT 检查点权重（其余对照实验检查点未全部单独发布权重文件）。

## 1. 仿真与基准仓库：`leeibo/ActiveArena`

- **GitHub 地址：** https://github.com/leeibo/ActiveArena
- **默认分支：** `main`
- **开源协议：** `MIT`
- **仓库规模（核查快照）：** 35 Stars
- **职责范围：** 提供 ActiveArena-Sim（基于 RoboTwin 2.0 适配 Astribot S1）、ActiveArena-Bench 35 个任务环境、轨迹采集脚本、过程标注管线与固定种子 ID/OOD 评测器。

### 已核实目录与入口文件

- `collect_data.sh`：轨迹采集入口，示例命令 `bash collect_data.sh count_target_press_button info_gathering_demo 0`（或 `info_gathering_randomized`）。
- `script/eval_policy.py`：固定种子评测主入口，读取策略适配器 `policy/<my_policy>/__init__.py`（需导出 `get_model(usr_args)`、`reset_model(model)`、`eval(TASK_ENV, model, observation)`）及 `deploy_policy.yml`（含 `use_eval_seed_list: true`、`test_num: 50`）。
- `envs/_base_task.py`：环境交互基类接口（提供 `TASK_ENV.get_obs()`、`TASK_ENV.get_instruction()`、`TASK_ENV.take_action(...)`）。
- `task_config/`：包含 `eval_seed_task_whitelist.yml`、`info_gathering_demo.yml`（ID）、`info_gathering_randomized.yml`（OOD）等配置。
- `eval_seed_lists/`：35 项任务各自对应的 100 个确定性随机种子清单（基准协议取前 50 个）。
- `ACTIVEARENA_EVAL_LAUNCH.md` & `docs/task_categories.md`：多 GPU 并行评测启动说明与 5 类 35 任务映射文档。
- `code_gen/`（`task_generation.py`、`gpt_agent.py`、`observation_agent.py`、`run_code.py` 等）与 `description/`：任务代码生成与指令模板管线。
- `environment.yml`、`script/_install.sh`、`script/_download_assets.sh`、`script/download_activearena_assets.py`：Linux + Python 3.10 + Vulkan + PyTorch3D/cuRobo 环境构建与资产校验脚本。

## 2. 策略训练与推理仓库：`leeibo/ActiveArena-VLA`

- **GitHub 地址：** https://github.com/leeibo/ActiveArena-VLA
- **默认分支：** `main`
- **开源协议：** GitHub 元信息标记 `NOASSERTION`；README 说明保留上游 [StarVLA](https://github.com/starVLA/starVLA) 许可与署名条款
- **仓库规模（核查快照）：** 3 Stars
- **职责范围：** 基于 StarVLA 框架实现 ActiveArena-VLA 模块化训练管线、HDF5→LeRobot 数据转换与 WebSocket 策略推理服务端。

### 已核实目录与入口文件

- `starVLA/model/framework/VLM4A/`：包含 `QwenFast.py`、`QwenFastState.py`、`QwenGR00T.py`、`QwenAdapter.py` 等 VLA 模型实现（基于 `Qwen/Qwen3-VL-2B-Instruct` 修订版 `89644892e4d85e24eaac8bacfd4f463576704203`，见 `MODEL_SNAPSHOT.md`）。
- `starVLA/dataloader/`：包含 `planner_oft_datasets.py`、`lerobot_datasets.py`、`gr00t_lerobot/` 数据加载器。
- `examples/ActiveArena_Astribot/train_files/managed_runs/`：公开的 3 组 OFT 训练与服务目录：
  - `oft_instruction_action_12_ws/`（`config.yaml`、`run_train.sh`、`start_train.sh`、`run_policy_server.sh`）
  - `oft_subtask_action_12_wos/`（含 `submit_yhbatch.sh`）
  - `oft_subtask_action_12_ws/`
  - 以及 `activearena_fast_astribot_history.yaml`、`activearena_fast_state_astribot_history.yaml` FAST 配方文件。
- `deployment/model_server/`：`server_policy.py`、`policy_wrapper.py`、`policy_norm_processor.py` 与 `tools/websocket_policy_server.py`（默认监听 `7980` 端口供 `ActiveArena` 评测端接入）。
- `docs/data-conversion.md`：记录 HDF5 到 LeRobot 数据格式转换、动作关键帧历史与子任务标签契约。

## 3. 关联资源与wiki映射

- **数据集：** https://huggingface.co/datasets/leeibo/ActiveArena-Data （35 任务 × 100 ID 轨迹，581.2k 帧）
- **仿真资产：** https://huggingface.co/datasets/leeibo/ActiveArena-Assets （含 `005_button` 与 `astribot_descriptions_texture`）
- **模型权重：** https://huggingface.co/leeibo/ActiveArena-VLA （3 组公开 OFT 检查点）
- **映射至 wiki 实体：** [ActiveArena: Benchmarking and Understanding Active Perception in Robotic Manipulation](../../wiki/entities/paper-agentic-2609-24124.md)
