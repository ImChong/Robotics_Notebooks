# Cyclo Intelligence（ROBOTIS Physical AI 全栈）

> 来源归档

- **标题：** Cyclo Intelligence
- **类型：** repo
- **链接：** https://github.com/ROBOTIS-GIT/cyclo_intelligence
- **机构：** ROBOTIS（乐百机器人）
- **Stars：** ~15（2026-07）
- **入库日期：** 2026-07-06 · **复核日期：** 2026-11-14（仓内 commit `11800ba4`）
- **核对范围：** 仓内 `README.md`、`docs/ARCHITECTURE.md`、`orchestrator/README.md`、`cyclo_brain/README.md`、`orchestrator/orchestrator/bt/bt_core.py`、`bt_node.py`、`actions/send_command.py`、`actions/arm_state_gate.py`、`trees/ffw_sg2_rev1_example.xml`、`shared/shared/robot_configs/schema.py`；并核查 `manifest.json` / `step2b_report.json`（确认关联参照文档 `docs.nav2.org/behavior_trees/index.html`、`concepts/index.html`、`nav2_specific_nodes.html` 均返回 HTTP 404 未达）
- **通读：** 已完成
- **一句话说明：** ROBOTIS 开源 Physical AI 全栈：数据采集、转换、训练、推理与真机执行合一；`orchestrator` 的独立 `bt_node` 用行为树编排 VLA/GR00T 策略与容器生命周期，对接 LeRobot 与 Isaac-GR00T 后端。
- **沉淀到 wiki：** [cyclo-intelligence](../../wiki/entities/cyclo-intelligence.md)、[behavior-tree-vla-orchestration](../../wiki/concepts/behavior-tree-vla-orchestration.md)

---

## 核心定位

**Cyclo Intelligence** 是 ROBOTIS 面向 **AI Worker / OMY / OMX** 等 Physical AI 产品线的 **端到端开源平台**，把「遥操作采集 → 数据集转换 → 策略训练 → 容器化推理 → 真机执行」收进单一仓库，默认以 **Docker Compose + s6** 在 Jetson（ARM64）或 AMD64 工作站上一键拉起。

官方文档入口：[ai.robotis.com](https://ai.robotis.com/)

---

## 仓库分层（README 摘要）

| 目录 | 角色 |
|------|------|
| `shared/` | 机器人配置、IO 辅助、日志 |
| `cyclo_brain/` | 训练 + 推理（`policy/` 下按后端分容器：LeRobot、GR00T） |
| `cyclo_data/` | 数据录制 / 转换 / Hub 上传（ROS 2 节点） |
| `orchestrator/` | **会话状态、UI、行为树控制**（含 React BT Manager） |
| `interfaces/` | ROS 2 msg/srv 定义 |
| `docker/` | 统一 compose、s6 服务、多架构 Dockerfile |

**子模块（钉版本）：** `zenoh_ros2_sdk`、`huggingface/lerobot`、`NVIDIA/Isaac-GR00T`。

---

## 行为树 × VLA：工程结合点（11800ba4 实码核对）

`orchestrator` 内包含两个进程隔离的 ROS 2 节点（包边界由 `orchestrator/tests/bt/test_package_boundary.py` 强检，不共享 Python 导入）：

1. **`orchestrator_node`（控制平面）：** 管理 `/task/status`、`/send_command` 路由及策略容器生命周期。
2. **`bt_node`（`orchestrator_bt_node`，默认 30.0 Hz tick）：** 由 `supervisor_api` 按需拉起的行为树执行引擎，暴露 `/bt/nodes/catalog`、`/bt/load_and_run`、`/bt/set_running`、`/bt/status`、`/bt/active_nodes`；当前 `shared/shared/robot_configs/schema.py` 将 `BT_SUPPORTED_ROBOT_TYPES` 显式收敛为 **`('ffw_sg2_rev1',)`**。

### `SendCommand` 状态机（`bt/actions/send_command.py`）

`SendCommand` 实现 `_STATE_INIT → _STATE_WAITING_BACKEND → _STATE_BEGIN_STAGE → _STATE_WAIT_PHASE → _STATE_DONE` 非阻塞状态机，支持双目标：

| BT `target` | BT `command` | 对应控制语义 / 接口 | 目标状态与同步机制 |
|-------------|--------------|--------------------|-------------------|
| `INFERENCE`（默认） | `LOAD` | 确保后端 Docker 就绪后发 `START_INFERENCE` → `STOP_INFERENCE` | `INFERENCING` → `PAUSED`（预热在内存） |
| `INFERENCE` | `RESUME` / `STOP` / `CLEAR` | `RESUME_INFERENCE` / `STOP_INFERENCE` / `FINISH` | `INFERENCING` / `PAUSED` / `READY` |
| `DOCKER` | `START` / `STOP` / `RESTART` | 后台线程调 `supervisor_api`（port `7100`）启停容器 | 轮询后端服务与 `generation` 锁，取消时发补偿 `STOP` |

节点在每阶段**轮询 `/task/inference_status`**，确保下游 BT 不会在半加载或过渡相位继续执行。

### 支持的策略后端与加速模式（`MODEL_SERVICE_TYPES`）

- **LeRobot 系：** `lerobot:act`、`lerobot:diffusion`、`lerobot:smolvla`、`lerobot:xvla`、`lerobot:pi0`、`lerobot:pi05`、`lerobot:molmoact2`、`lerobot:vla_jepa`、`lerobot:fastwam`
- **GR00T 系：** `groot:n17`（`n17` / `n1.7`）

支持端口：`task_instruction`、`inference_hz`、`control_hz`、`chunk_align_window_s`、`action_request_mode`（`async` / `sync`）、`inference_mode`（`simulation` 仅预览 vs `robot` 发真机话题）以及 `acceleration_mode`（`pytorch` / `tensorrt_dit` / `tensorrt_full_pipeline`）。

### 内置行为树节点与示例树 `ffw_sg2_rev1_example.xml`

除 `SendCommand` 外，仓内内置动作与控制节点包括 `JointControl`、`Rotate`、`Wait`、`ArmStateGate`（只订阅 `/joint_states` 监测关节位置与夹爪闭合/张开序列，不发轨迹命令）与 `Sequence` / `Loop`（`max_iterations`）。示例树典型长程模式：

1. `LOAD` `groot:n17` 模型 + 语言指令（加载后暂停）
2. `JointControl` 复位头/双臂/升降台，`Rotate` 底盘转向 90°
3. `Loop`（最多 3 次）：`RESUME` 推理 3s → `STOP` → `JointControl` 臂回初始位 5s
4. `CLEAR` 卸载策略

---

## 推理运行时与导航边界（`docs/ARCHITECTURE.md` & `cyclo_brain/README.md`）

- **主容器（`cyclo_intelligence`）：** UI/nginx（Autonomy Studio）、`supervisor_api`、`orchestrator_node`、`bt_node`、`cyclo_data`。
- **导航容器（`ai_worker`）与任务测序：** 航点级别导航（Nav2 goal per waypoint）与随后的单点 BT 触发目前由浏览器 **Autonomy Studio**（`orchestrator/ui/src/hooks/useMissionRunner.js`）协调 `supervisor_api` 与 `/bt/*` 接口驱动，尚未下沉为纯机上单一大 BT。
- **策略容器（按 `lerobot` / `groot` 隔离）：** `main-runtime`（`ServiceHandler`、`SessionState`、`ActionChunkProcessor`、`ControlLoop`、`RobotClient`）+ `engine-process`（`PolicyLoader`、可选 TensorRT、`Preprocessor`、`Predictor`）。`EngineCommand` 回传 `seq_id`，暂停/停止/卸载或切换 `publish_to_robot` 时立即清空缓冲并返回 `None`（不重复旧帧），防止陈旧 chunk 跳变。
- **通信：** `InferenceCommand.srv` / `EngineCommand.srv`；Zenoh ROS 2 可配置 `ROS_DOMAIN_ID` 与共享内存。
- **外部参照文档核查状态（2026-11-14）：** 本专题涉及的 Nav2 行为树参照文档 `https://docs.nav2.org/behavior_trees/index.html`、`https://docs.nav2.org/concepts/index.html` 及 `https://docs.nav2.org/behavior_trees/overview/nav2_specific_nodes.html` 在原始抓包（`manifest.json` / `step2b_report.json`）中均返回 **HTTP 404**，未读取正文；当前百科结论严格以 `cyclo_intelligence` 源码为准。

---

## 为什么值得保留

- **少见的开源「BT + VLA」真机编排参考**：把行为树从导航域（Nav2）迁移到 **操作 / 半人形 Physical AI** 部署栈。
- **与 LeRobot / GR00T 生态直接打通**：不是独立 VLA 训练框架，而是 **ROBOTIS 硬件 + 数据 + 推理 + BT 任务机** 的一体化参考实现。
- **对知识库补位**：站内已有 [VLA 部署 query](../../wiki/queries/vla-deployment-guide.md) 与 [VLA+低层控制融合](../../wiki/queries/vla-with-low-level-controller.md)，但缺少 **显式行为树任务层** 的工程实体。

---

## 对 wiki 的映射

- **wiki/entities/cyclo-intelligence.md**（新建）— 平台实体与模块总览
- **wiki/concepts/behavior-tree-vla-orchestration.md**（新建）— BT 编排 VLA 的通用架构模式（以 Cyclo 为锚点案例）
- **wiki/methods/vla.md** — 补充「任务编排层」交叉引用
- **wiki/entities/lerobot.md** — Cyclo 作为 LeRobot 推理后端集成方
- **wiki/queries/vla-deployment-guide.md** — 补充 BT 生命周期与 phase 同步注意点
