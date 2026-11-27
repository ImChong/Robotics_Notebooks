# architect-franka

- **仓库地址：** <https://github.com/robo-architect/architect-franka>
- **许可证：** MIT
- **默认分支：** `main`
- **核对方式：** 静态代码核对（基于仓库目录采样与 README 文档，未本地运行）
- **关联论文：** [paper-agentic-2607-23784](../papers/paper-agentic-2607-23784.md)

## 仓库定位与依赖边界

`robo-architect/architect-franka` 是论文 *A Few Words Go a Long Way: Language Guided Robot Policy Synthesis*（ARCHITECT）的官方开源代码仓库，包含编码智能体应用与 Franka Panda ROS 1 Noetic 工具服务栈两部分：

1. **已在本仓库公开的核心资产：**
   - `architect/`：LLM 编码智能体主程序，负责根据自然语言指令与持久化技能库合成、执行并纠偏机器人策略代码。
   - `src/franka_robot_apis/` 与 `src/robot_api_interfaces/`：将 Franka Panda + Robotiq 2F-85 + RealSense/ZED 相机封装为统一 `/robot/...` JSON-over-ROS 服务（`RobotCommand.srv` / `RobotQuery.srv` / `ResultCode.msg`）。
   - `src/third_party/`：随仓附带的驱动与中间件依赖（`franka_ros`、`serl_franka_controllers`、`robotiq`、`ros_canopen`、`soem`、`agentlace`，各自保留原许可证）。
2. **外部未内置服务与前置条件：**
   - `libfranka`（0.8.0）需按 `docs/INSTALL.md` 单独克隆编译。
   - 重计算感知与规划模型服务（cuRobo IK/轨迹 `8765`、Qwen VLM `8000`、Grounded-SAM2 分割 `8766`、AnyGrasp 抓取 `8767`、AnyPlace 放置 `8768`、DIFT 关键点 `8769`）须作为独立 WebSocket/HTTP 服务部署并通过 `src/franka_robot_apis/config/servers.yaml` 与 `inference_host` 指定；VQA 亦可通过 `.env` 对接 OpenAI / Azure OpenAI API。

## 核心目录与入口文件（静态核对）

| 路径 | 角色与机制 |
|---|---|
| `architect/run_architect.sh`、`architect/scripts/architect_cli.py` | 智能体主启动脚本与 CLI 入口，连接运行中的 ROS 工具栈并接收自然语言任务指令 |
| `architect/architect/agent/`（`session.py`、`synthesis.py`、`correction_session.py`、`primitive_registry.py`、`scratchpad.py`、`tools.py`） | 交互式程序合成核心：管理会话状态、原语注册表、代码合成、草稿区与工具调用，并驱动人机纠偏循环 |
| `architect/architect/corrections/`（`execution_trace.py`、`failure_diagnosis.py`、`fault_localization.py`、`llm_scorer.py`、`subtask_docs.py`） | 执行轨迹记录、失败诊断与故障代码定位，将人工自然语言纠偏精确锚定到失败语句或原语调用 |
| `architect/architect/library/`（`skill_store.py`、`skill_doc_index.py`、`relevance.py`、`context.py`、`episode_log.py`、`failure_store.py`） | 持久化技能库与记忆模块：存储/检索从纠偏中蒸馏出的可复用技能与失败记录，并在下次合成前注入上下文 |
| `architect/architect/probes/`（`probe_runner.py`、`audit.py`、`graduation.py`、`sampling.py`、`scene_mutator.py`） | 策略探测、审计与技能毕业/验证辅助模块 |
| `architect/architect/robots/`（`base.py`、`franka.py`、`franka_api.py`、`franka_client.py`） | 智能体侧机器人客户端，封装对 ROS 服务接口的调用 |
| `architect/skills/franka/`（`grasping.md`、`motion.md`、`placement.md`、`verification.md`、`vqa.md`、`get_keypoints.md`、`get_keypoints_trajectory.md`、`ros1.md`） | 预置领域技能文档，为编码智能体提供各类原语的调用规范与避坑提示 |
| `architect/scripts/eval/`（`run_trial.py`、`sweep.py`、`aggregate.py`、`tasks.json` 等） | 自动化评测、消融扫描与结果汇总脚本 |
| `src/franka_robot_apis/launch/franka_robot_core.launch` | ROS 1 核心启动文件：拉起笛卡尔阻抗控制器、夹爪节点、相机流客户端与全部 `/robot/...` 服务节点 |
| `src/franka_robot_apis/scripts/`（`execute_impedance_trajectory_node.py`、`move_ee_cartesian_controller_node.py`、`detect_objects_service.py`、`get_grasp_config_service.py`、`get_placing_pose_service.py`、`get_keypoints_service.py`、`get_vqa_service.py` 等 15 个服务节点） | 实现 `/robot/control/*`、`/robot/proprioception/*`、`/robot/perception/*` 各 JSON-over-ROS 服务，并将底层推理转发至外部模型服务端 |
| `src/robot_api_interfaces/srv/`（`RobotCommand.srv`、`RobotQuery.srv`）与 `msg/ResultCode.msg` | 定义统一服务消息格式与状态码（`SUCCESS=0, FAILURE=1, INVALID_INPUT=2, TIMEOUT=3, SERVICE_NOT_RUNNING=4`） |

## 官方运行链路摘要（据 README 与代码入口）

1. **环境与配置准备：** 激活 `env_franka`（RoboStack / ROS Noetic，Python 3.9 + `numpy==1.26.4`）并 `source devel/setup.bash`；在 `src/franka_robot_apis/` 下复制配置 `.env` 与 `config/{servers,cameras,robot}.yaml`，填入外部推理主机与相机标定参数。
2. **启动底层机器人服务栈（终端 1）：** 运行 `roslaunch franka_robot_apis franka_robot_core.launch robot_ip:=... gripper_device:=... inference_host:=...`，启动笛卡尔阻抗控制器、Robotiq 夹爪节点、RealSense/ZED 相机流与全部 `/robot/*` ROS 服务。
3. **启动 ARCHITECT 编程智能体（终端 2）：** 进入 `architect/` 执行 `./run_architect.sh "<任务指令>"`，由 `architect_cli.py` 启动合成会话：查询机器人状态与技能库 → 注册子操作原语并提交顶层程序 → 通过 `franka_client.py` 调用 `/robot/*` 服务闭环执行与 VQA 验证 → 若失败则收集人类自然语言纠偏、结合 `execution_trace.py` 诊断定位并写入 `skill_store.py`。
