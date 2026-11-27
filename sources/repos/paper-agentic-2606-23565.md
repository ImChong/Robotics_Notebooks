# HorizonRobotics/HoloAgent 官方仓库核对归档

- **官方仓库：** [HorizonRobotics/HoloAgent](https://github.com/HorizonRobotics/HoloAgent)
- **分支：** `main`
- **许可证：** README 声明采用 Apache License 2.0（见仓库根目录 `LICENSE`；GitHub 元数据采样字段返回 `NOASSERTION`）
- **仓库规模快照：** 428 stars（2026-11-27 采集缓存快照）
- **关联论文与项目页：** [arXiv:2606.23565](https://arxiv.org/abs/2606.23565) · [官方项目页](https://horizonrobotics.github.io/robot_lab/holoagent) · [FSR-VLN 数据集](https://huggingface.co/datasets/HorizonRobotics/fsrvln_datasets)
- **核对边界：** 静态代码核对，未本地运行（基于仓库 README 预览与采样文件清单核对；未在本地搭建 ROS2 / Unitree G1 / Mid-360 / 豆包语音 API 环境执行）

## 已核对目录与关键入口文件

基于仓库采样文件清单与 README 说明，已确认的核心模块如下：

1. **AgentOS 技能与长程编排（`agentic_robot/agentOS/`）**
   - 技能注册与生命周期脚本：`agentic_robot/agentOS/holoagent_skills/scripts/`（`create_skill.py`、`delete_skill.py`、`list_skills.py`、`show_skill.py`、`validate_skills.py`、`sync_legacy_docs.py`）与教程 `docs/skill_crud_tutorial.md`。
   - 具体可调用技能包：`arm-skill/scripts/trigger_arm_skill.py`、`rel-move-skill/scripts/relative_move.py`、`robot-service/scripts/service_request.py`、`sem-nav-skill/scripts/semantic_nav.py` 及 `workflow/SKILL.md`。
   - 后台守护与长程测试运行器：`agentic_robot/agentOS/run_dameon/run_g1_background_daemon.py`、`agentic_robot/agentOS/sandbox_test/long_horizon_text_runner.py`、`test_single_robot_long_instruction.py`、`test_multi_robot_long_instruction.py`，以及持久化调试产物 `dag_plan.yaml`、`virtual_validation.yaml`、`monitor.yaml`、`execution_result.yaml`。
2. **G1 语音交互与 OpenClaw 适配（`agentic_robot/chatbot/g1/`）**
   - 节点与演示入口：`g1chat_node.py`、`g1tts_node.py`、`g1chat_demo.py`、`g1chat_demo_multirobot_dag.py`、`openclaw_doubao_asr.py`。
   - 音频与语音管线：`audio/audio_device.py`、`audio/volcengine_doubao_asr.py`、`audio/volcengine_doubao_tts.py`，以及中英文系统提示词 `g1_system_prompt_zh.txt` / `g1_system_prompt_en.txt`。
3. **底层几何定位与建图（`agentic_robot/core/src/fast_livo/`）**
   - 编译与核心头文件：`CMakeLists.txt`、`build_fastlivo2.sh`、`include/LIVMapper.h`、`include/multi-session/Incremental_mapping.cpp`、`include/online-relo/pose_estimator.cpp`。
   - 传感器与平台配置：`config/mid360_online_livo.yaml`、`config/mid360_online_reloc_g1.yaml`、`config/mid360_online_reloc_hexfellow.yaml`、`config/camera_d435i.yaml` 等。
4. **构建与部署文档**
   - 构建脚本 `agentic_robot/build.sh`、`agentic_robot/clear.sh`，以及系统说明文档 `docs/user_guide/Intruduction.md`。

## 开源覆盖范围（对照 README Release Status）

- **已勾选发布（HoloAgent-0 与 FSR-VLN）：** HoloAgent-0 代码更新（含基于 OVO-Mapping 加速的 FSR-VLN 语义建图、Doubao LLM 语音交互升级、OpenClaw 适配的 AgentOS Harness、受 dimos 启发的后台进程隔离与 Skill/Blueprint 触发机制）、HoloAgent-0 项目页与论文、FSR-VLN 代码。
- **尚未勾选发布（规划中的 HoloAgent-1 更新）：** 通用灵巧操作 `HoloBrain`、全身移动操作 `HoloMotion`、主动探索 Agent-Loop，以及纯视觉自主建图/定位/导航/避障管线。

## 对 wiki 的映射

- [HoloAgent-0: A Unified Embodied Agent Framework with 3D Spatial Memory](../../wiki/entities/paper-agentic-2606-23565.md)
