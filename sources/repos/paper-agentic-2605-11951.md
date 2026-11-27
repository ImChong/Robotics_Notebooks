# EDEM-AI/AgentChord — AgentChord 官方开源仓库档案

- **GitHub 地址：** https://github.com/EDEM-AI/AgentChord （论文附录另列开发/个人仓库 https://github.com/Jasonxu1225/AgentChord ）
- **默认分支：** `main`
- **开源协议：** Apache-2.0 License
- **关联论文：** *From Reaction to Anticipation: Proactive Failure Recovery through Agentic Task Graph for Robotic Manipulation*（[arXiv:2605.11951](https://arxiv.org/abs/2605.11951)，RSS 2026）
- **核对边界：** 静态代码核对，未本地运行（基于仓库元数据、`README.md` 与 `files_sample` 核查，未在本地配置 EmbodiChain GPU 仿真环境、SAM3 权重或连接 CobotMagic 真机执行）。

## 核心入口与目录结构（已核实）

仓库基于 [EmbodiChain](https://github.com/DexForce/EmbodiChain) 构建（Python 包名仍为 `embodichain`），包含以下已核实模块与运行入口：

- **三智能体与 LLM 接口（`README.md` 索引）：**
  - `embodichain/agents/hierarchy/task_agent.py`：Task Structuring Agent（Composer），从指令与观测生成名义有向任务图。
  - `embodichain/agents/hierarchy/recovery_agent.py`：Recovery Orchestration Agent（Arranger），预测失效模式并构建前向恢复分支。
  - `embodichain/agents/hierarchy/compile_agent.py`：Execution Compilation Agent（Conductor），编译节点关键帧、边过渡程序与在线监控器。
  - `embodichain/agents/hierarchy/llm.py`：基于 LangChain `ChatOpenAI` 封装（默认初始化 `gpt-5`，读取环境变量 `LLM_URL` 与 `OPENAI_API_KEY`）。
- **仿真运行入口与任务配置：**
  - `embodichain/lab/scripts/run_agent.py`：仿真主入口，支持 `--recovery`、`--interactive_error_injection`（终端按 `f` 注入 `misplaced_object` / `fallen_object`）、`--regenerate`、`--headless`、`--device cuda --gpu_id 0` 等参数。
  - `configs/gym/agent/pour_water_agent/`：`fast_gym_config.json`、`agent_config.json`、`agent_config_dual.json`、`task_prompt.txt`、`task_prompt_dual.txt`（对应 `SinglePourWater` 与 `DualPourWater`）。
  - `configs/gym/agent/rearrangement_agent/`：`fast_gym_config.json`、`agent_config.json`、`task_prompt.txt`（对应 `Rearrangement`）。
  - 仿真产物缓存目录：`embodichain/database/agent_generated_content/sim/<task_name>/`（生成 `agent_task_graph.json`、`agent_recovery_spec.json`、`agent_compiled_graph.json`）。
- **真机部署入口与硬件桥接：**
  - `embodichain/deploy/run_agent_realworld.py`：CobotMagic 真机主入口，流程涵盖构建 ROS 操作器、初始化 CobotMagic FK/IK 运动学模型、初始化双目相机（默认 Kingfisher）与 SAM3 分割、回默认关节位姿、估计 `--object_names` 位姿、加载/生成图并执行。
  - `embodichain/deploy/robots/cobotmagic/bridge_params.json`：CobotMagic ROS 2 话题与双臂 Piper 驱动参数（需先配置 `can_left` / `can_right` 1 Mbps 并启动 `ros2 launch piper start_multi_piper.launch.py`）。
  - `embodichain/deploy/tools/sam3/sam3.pt`：SAM3 权重默认路径（需用户自行下载放置，`ultralytics` 推理输出写至 `runs/`）。
  - 真机产物缓存目录：`embodichain/database/agent_generated_content/real/<task_name>/`。
- **容器与文档支持：**
  - `docker/docker_run.sh`（官方镜像 `dexforce/embodichain:ubuntu22.04-cuda12.8`）、`docs/source/features/agents.md`、`docs/source/features/failure_monitor_recovery.md`。

## 复现与运行依赖边界

- **系统与硬件环境：** 继承 EmbodiChain 要求，需 Ubuntu 20.04+、NVIDIA GPU（Compute Capability 7.0+，推荐驱动 535–570）、Python 3.10 或 3.11，以及专属依赖源 `http://pyp.open3dv.site:2345/simple/`。
- **外部 API 与权重：** 需配置兼容 OpenAI 的 `LLM_URL` 与 `OPENAI_API_KEY`（默认使用 `gpt-5`）；真机部署需另装 `ultralytics`、自行下载 `sam3.pt`，并配备 ROS 2、`piper` / `piper_msgs` 与 Kingfisher（或自替换 RealSense）相机驱动。
- **发布范围说明：** 截至 `README.md` 记录版本，三类仿真任务、核心层次智能体及真机部署脚本已发布；“扩展更多失效监控函数与恢复函数模板”列为后续待补充内容。

## 与 wiki 的映射

- 对应实体页：[From Reaction to Anticipation: Proactive Failure Recovery through Agentic Task Graph for Robotic Manipulation](../../wiki/entities/paper-agentic-2605-11951.md)
