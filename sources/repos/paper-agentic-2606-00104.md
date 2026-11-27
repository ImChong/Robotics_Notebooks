# erdemuysalx/PEACE 官方源码归档

- **官方仓库：** [https://github.com/erdemuysalx/PEACE](https://github.com/erdemuysalx/PEACE)
- **默认分支：** `main`
- **开源状态：** 已开源（包含可运行 ROS 2 / PX4 SITL 智能体包、Docker 配置、提示词与评测脚本；仓库元数据未标注开源许可证）
- **配套数据集：** [HuggingFace: erdemuysalx/uav-mission-prompts](https://huggingface.co/datasets/erdemuysalx/uav-mission-prompts)
- **核对范围：** 核对 GitHub 仓库主页（commit history 概览、9 stars / 2 forks）、README 全文与已采样文件列表（31 个路径），确认可运行入口与模块结构。
- **复现边界：** 静态代码与文档核对，未本地运行。

## 目录与模块核对

| 路径 | 职责与说明 |
|---|---|
| `peace/nodes/vision.py` | 视觉节点（入口 `peace-vision`），驱动感知后端与深度投影 |
| `peace/nodes/planner_executor.py` | 规划–执行主节点（入口 `peace-planner-executor`），维护单次规划、工具调度与有界重规划 |
| `peace/nodes/cli.py` | 交互式/单条指令命令行入口（`peace-cli`，支持 `--monitor`），向 `/agent/query` 发布任务 |
| `peace/nodes/teleop.py` | 键盘与手柄遥操作节点 |
| `peace/backends/llm.py` | Ollama（本地或云端）LLM 接口封装（默认 `LANGUAGE_MODEL=llama3.2`） |
| `peace/backends/detector.py` | 可插拔 2D 检测后端：YOLO（ultralytics，默认 `yolov8n.pt`）或 VLM（Ollama `llava:13b`） |
| `peace/platforms/robot_tools.py`, `peace/platforms/uav_tools.py` | 结构化机器人/无人机工具定义（`UAVTools`）与 MAVROS / MAVLink 对接 |
| `peace/services/robot_state_service.py` | 线程安全机器人状态聚合（里程计、飞行模式、arming 状态） |
| `peace/services/world_model_service.py` | 线程安全世界模型聚合（带时间戳的检测结果历史） |
| `peace/services/depth_service.py` | 针孔相机深度反投影服务（2D 框转 ENU 3D 坐标并输出深度有效性标志） |
| `peace/services/constraint_enforcement_service.py` | 安全校验服务：最小/最大高度（`AGENT_MIN_ALTITUDE=1.0`、`AGENT_MAX_ALTITUDE=50.0`）与水平地理围栏（`AGENT_HORIZONTAL_LIMIT=500.0`）及逐条指令开关（`ACTION_ENABLE_CONTROL`） |
| `peace/services/mission_logger_service.py` | 任务 CSV 日志服务（含 `metadata_json`）供离线审计 |
| `peace/system_prompts/language.txt`, `peace/system_prompts/vision.txt` | 规划系统提示词与 VLM 检测系统提示词 |
| `peace/schemas.py`, `peace/core/config.py`, `peace/exceptions.py` | 类型化 `MissionPlan` Schema、配置与异常定义 |
| `evaluation/run_dataset.py` | 批处理评测脚本，读取 HuggingFace 数据集并向 `/agent/query` 发布任务 |
| `install.sh`, `pyproject.toml`, `uv.lock`, `.env.example` | 容器内安装脚本、依赖锁文件与环境变量模板 |

## 官方运行入口（来自 README）

1. **Docker SITL 环境启动：** `docker compose up -d`（构建 ROS 2 Jazzy + Gazebo Harmonic + MAVROS + NoVNC 镜像），进入容器执行 `./install.sh` 并激活 `.venv`。
2. **仿真与飞控桥接：** 终端 1 启动 `make px4_sitl gz_x500`；终端 2 启动 `ros2 run mavros mavros_node --ros-args -p fcu_url:=udp://:14540@14557`。
3. **启动智能体节点：** 运行 `peace-vision` 与 `peace-planner-executor`（后台启动视觉节点与规划–执行节点）。
4. **下发任务或批量评测：** 使用 `peace-cli`（交互模式）或 `peace-cli "<query>" --monitor`，或运行 `python -m evaluation.run_dataset --limit 5`。

## 对 wiki 的映射

- [PEACE: A Planner-Executor Agent with Constraint Enforcement for UAVs](../../wiki/entities/paper-agentic-2606-00104.md)
