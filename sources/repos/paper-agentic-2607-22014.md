# FraunhoferIVI/MissionBench — MissionBench 官方开源仓库档案

- **GitHub 地址：** https://github.com/FraunhoferIVI/MissionBench
- **默认分支：** `master`（快照记录 9 commits、6 stars、1 fork）
- **开源协议：** CC-BY-4.0（仓库根目录含 `LICENSE` 文件）
- **关联论文：** *Zero-Shot Mission-Level Evaluation for Aerial MLLM Agents*（[arXiv:2607.22014](https://arxiv.org/abs/2607.22014)）
- **核对边界：** 静态代码核对，未本地运行（基于仓库主页、`README.md` 与 `files_sample` 文件清单核查；实际复现需 Ubuntu 22/24、Vulkan 1.3.275、NVIDIA RTX 3050 4GB+、250 GB 可用磁盘与各 MLLM API Key 或本地 `llama.cpp` 服务，且首次运行 `main_driver.py` 时会自动下载打包的 UE5/Cosys-AirSim 环境二进制与测试集）

## 核心入口与目录结构（已核实）

- **主驱动与环境配置（根目录）：**
  - `main_driver.py`：基准评测主入口（推荐先跑 `python main_driver.py --config_file configs/unit_testset.yaml` 验证 NHEnv、ForestEnv 与 CityEnv 各一条用例，再跑完整 `configs/testset.yaml`）。
  - `airsim_api_server.py`：面向 Cosys-AirSim 的独立 API 服务端，隔离仿真环境依赖与 MLLM 推理环境。
  - `env_setup.sh`、`pyproject.toml`、`.env_sample`：自动构建两个独立虚拟环境 `airsim_env` 与 `missionbench_env`（Python 3.10.12），并通过 `.env` 配置 Gemini、GPT、AWS Nova、Anthropic 等模型 API 凭证。
- **LangGraph 智能体执行图与策略（`graphs/`）：**
  - `graphs/state.py`、`graphs/step_by_step_nodes.py`、`graphs/patrol_metric.py`：维护任务状态、逐步执行节点与巡逻轨迹 20 m 缓冲区 IoU 计算逻辑。
  - `graphs/strategies/step_by_step_1vlm.py`、`graphs/strategies/step_by_step_1vlm_with_bb.py`、`graphs/strategies/step_by_step_2vlm.py`：支持单 VLM、带边界框预测的单 VLM 以及双 VLM 协同的逐步控制策略。
- **提示词工程与模型预测器（`prompter/`、`predictors/`）：**
  - `prompter/prompt_generator.py` 及 `prompter/prompts/system_prompt.py`、`prompter/prompts/basic_missions/{basic_prompt,evaluation_prompt,prompt_utils,step_by_step,step_by_step_V2}.py`、`prompter/prompts/proxy_prompts/{bb_and_depth_prompt,bb_and_depth_prompt_V2}.py`：生成闭环导航提示词、单帧感知代理提示词与评测提示词。
  - `predictors/{base,vision,llama_client,llm_as_judge_runner,test_llm_as_judge,test_predictors,utils}.py`：统一封装闭源/开源云端 MLLM API、本地 `llama.cpp` 客户端以及 Reporting 任务的 `LLM-as-a-judge` 判分运行器。
- **仿真接口与批量实验自动化（`simulator_interface/`、`experiment_automation/`）：**
  - `simulator_interface/{airsim_client_wrapper,airsim_compat,airsim_tools,unreal_process_manager}.py`：封装 UE5 进程生命周期、多旋翼/计算机视觉模式控制与视角/图像采集。
  - `experiment_automation/{runner,executor,config,aggregator,create_results,recalculate_metrics,results_json_updates,latex_reporter}.py`：自动化执行多次重复试验、聚合 SR/MP/OSR/CR/Eff 指标并导出结果表。
- **配置、任务集与工具（`configs/`、`dataset/`、`examples/`、`tests/`、`utils/`）：**
  - `configs/{config,environments,models,testset,unit_testset,ablation_test}.yaml` 与 `settings_{computer_vision_mode,multirotor}.json`：管理测试集划分、模型命名与 AirSim 配置。
  - `dataset/`：含 `easy_missions.yaml` 及按任务分类的样例任务（如 `dataset/Manipulation/NHEnv/yard_mission3/mission_config.yaml`、`dataset/Patrol/NHEnv/main_road_mission3/`、`dataset/Visual_Inspection/NHEnv/deer_statue_mission15/`，内含 `airsim_rec.txt` 与参考帧）。
  - `examples/step_by_step_demo.py`、`tests/{pose_test_cases,test_convert_poses,test_environment_manager}.py`、`utils/{environment_manager,dataset_loader,convert_pose,failure_case_analysis_utils,gym_utils,langsmith_logging}.py`。

## 复现与运行依赖边界

- **强系统与存储依赖：** 依赖 Ubuntu 22/24 与 Vulkan 1.3.275 图形栈，且打包的 UE5 仿真场景体积较大（官方要求至少 250 GB 空闲磁盘）。
- **外部模型服务依赖：** 运行评估需自行准备 Gemini / OpenAI / Anthropic / AWS Bedrock 等 API Key 或按需部署本地 `llama.cpp` 视觉语言服务；完整 30 条测试集运行 3 轮约消耗 12.8M 输入 token（1920×1080 三帧历史），商业 API 单模型开销约 $19.2–$71.9。

## 与 wiki 的映射

- 对应实体页：[Zero-Shot Mission-Level Evaluation for Aerial MLLM Agents](../../wiki/entities/paper-agentic-2607-22014.md)
