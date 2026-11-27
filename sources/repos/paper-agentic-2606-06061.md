# franka-llm：分布式生成式 AI 人机协作操作框架官方源码归档

> 来源归档（repo）；核查日：2026-11-27；**已开源**（论文对应完整 ROS 2 系统与实验数据均已公开）

- **仓库：** <https://github.com/cogrob-tuni/franka-llm> （Forked from `@arashghsz`）
- **默认分支：** `main`（核查时共 101 commits）
- **论文：** <https://arxiv.org/abs/2606.06061>
- **许可：** README 标注 Apache License 2.0（仓内含 `LICENSE` 文件；GitHub API 自动检测返回 `NOASSERTION`）
- **核对范围：** 基于官方仓库主页文件目录、README、配置片段、调试图像列表与文档索引进行静态核对（`README.md`、`RUNNING.md`、`config.yaml`、`start_system.sh`、`stop_system.sh`、`restart_node.sh`、`robot_llm_vlm_experiments.csv`、`src/` 下 6 个 ROS 2 包描述与 `ui/` 目录结构）。
- **复现边界：** 静态代码核对，未本地运行（未接入 Franka Emika FR3 真机 `172.16.0.2`、Intel RealSense D435i 与双机 Ollama 环境）。
- **通读：** 已完成

## 已核实的仓库目录与关键入口

| 路径 / 包名 | 核实来源 | 职责说明 |
|---|---|---|
| `config.yaml` | 仓库根目录 / README | 集中配置 `llm`（`ministral-3:8b`）、`vlm`（`qwen2.5vl:32b`）、Ollama `base_url`（默认 `http://localhost:11434`）、相机话题、ArUco 标定偏置及 `robot` 高度常量（`safe_height: 0.60`、`grasp_height: 0.14`、`handover_height: 0.30` m） |
| `start_system.sh` / `stop_system.sh` / `restart_node.sh` | 仓库根目录 | 一键启动/停止全部节点，或按模块重启 `motion`、`moveit`、`camera`、`ui` 单个节点 |
| `src/franka_llm_planner/` | README（含 `llm_node.py`、`moveit.launch.py`） | LLM 意图解析与指令路由节点（Ministral 3 8B），同时包含 MoveIt2 启动文件 `moveit.launch.py` |
| `src/franka_vlm_agent/` | README（含 `VLM_QUICKSTART.md`） | 基于 Qwen2.5-VL 32B 的开放词汇目标检测、像素定位与调试图像生成 |
| `src/franka_coordinator/` | README（含 `coordinator_node.py`、`handle_vlm_grounding()`） | 中央协调器：分发 LLM/VLM 请求、像素与 `5×5` 深度融合、ArUco 2D→3D 坐标变换及用户确认状态管理 |
| `src/franka_motion_executor/` | README（含 `motion_executor_node.py`） | 接入 MoveIt2，执行 `pick`、`place`、`handover`、`go_home` 及 `dance` 运动原语并施加安全速度约束（服务 `/motion/execute`、状态话题 `/motion/status`） |
| `src/realsense_cameras/` | README | Intel RealSense D435i RGB-D 驱动与 `sensor_msgs/CameraInfo` 发布 |
| `src/franka_vision_detection/` | README | 传统视觉检测包（已标记 deprecated，被 `franka_vlm_agent` 替代） |
| `ui/` (`web_chat_dashboard`) | 仓库根目录 / README | Web 监控与确认界面（默认通过 `python3 -m http.server 8000` 访问，经 `web_handler` 与 ROSBridge 连接 ROS 2） |
| `robot_llm_vlm_experiments.csv` | 仓库根目录 | 180 次基准评测完整实验日志与数据 |
| `debug_images/` & `docs/` | 仓库根目录 | 带边界框/定位标注的 VLM 调试图像（含 `ministral-3-8b`、`llama3.1-8b`、`qwen2.5-7b`、`deepseek-r1-7b` 与 `qwen2.5vl-32b`、`llama3.2-vision-90b`、`llava-7b` 实验记录）及 `architecture.md`、`setup.md`、`metrics.md`、`system_log.txt` |

## README 给出的部署与运行入口

```bash
# 1. 拉取本地模型（Ollama）
ollama pull ministral-3:8b && ollama pull qwen2.5vl:32b

# 2. 编译 ROS 2 Jazzy 工作空间
colcon build --symlink-install && source install/setup.sh

# 3. 启动系统与 Web 仪表盘（http://localhost:8000）
./start_system.sh
```

README 注明若网络连接的 Franka FR3（`172.16.0.2`）初始化较慢导致 MoveIt2 未自动加载控制器，需在启动 `ros2 launch franka_llm_planner moveit.launch.py` 后手工通过 `ros2 control load_controller` 激活 `joint_state_broadcaster`、`franka_robot_state_broadcaster` 与 `fr3_arm_controller`。

## 映射

- [原论文归档](../papers/paper-agentic-2606-06061.md)
- [唯一主 Wiki 节点](../../wiki/entities/paper-agentic-2606-06061.md)
