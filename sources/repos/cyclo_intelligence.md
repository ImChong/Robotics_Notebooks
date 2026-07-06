# Cyclo Intelligence（ROBOTIS Physical AI 全栈）

> 来源归档

- **标题：** Cyclo Intelligence
- **类型：** repo
- **链接：** https://github.com/ROBOTIS-GIT/cyclo_intelligence
- **机构：** ROBOTIS（乐百机器人）
- **Stars：** ~15（2026-07）
- **入库日期：** 2026-07-06
- **一句话说明：** ROBOTIS 开源 Physical AI 全栈：数据采集、转换、训练、推理与真机执行合一；`orchestrator` 用行为树编排 VLA/GR00T 策略生命周期，对接 LeRobot 与 Isaac-GR00T 后端。
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

## 行为树 × VLA：工程结合点

`orchestrator` 的 **Behaviour Tree** 子系统负责 **宏任务编排**；VLA/模仿学习策略通过 **`SendCommand` BT 动作节点** 接入 orchestrator 已有的推理管线，而非在 BT 内直接跑模型前向。

### `SendCommand` 生命周期（`bt/actions/send_command.py`）

| BT `command` | 对应 orchestrator 语义 | 目标 `inference_phase` |
|--------------|------------------------|------------------------|
| `LOAD` | `START_INFERENCE` → `STOP_INFERENCE`（加载后暂停在内存） | `INFERENCING` → `PAUSED` |
| `RESUME` | `RESUME_INFERENCE` | `INFERENCING` |
| `STOP` | `STOP_INFERENCE` | `PAUSED` |
| `CLEAR` | `FINISH`（卸载） | `READY` |

节点在每一 stage **轮询 `/task/inference_status`**，确保下游 BT 不会在「半加载 / 过渡中」的策略上继续 tick——这是 BT 与异步 VLA 推理对接的关键同步机制。

### 支持的策略后端（`model` 端口）

- **LeRobot 系：** `lerobot:act`、`lerobot:smolvla`、`lerobot:xvla`、`lerobot:pi0`、`lerobot:pi05`、`lerobot:diffusion`
- **GR00T 系：** `groot:n17` 等

`task_instruction` 传入自然语言任务；`inference_hz` / `control_hz` / `chunk_align_window_s` 配置 **action chunk 异步执行**；`inference_mode` 可选 `simulation`（仅预览）或 `robot`（发布到真机话题）。

### 示例树 `ffw_sg2_rev1_example.xml`

典型长程任务模式：

1. `LOAD` GR00T 模型 + 语言指令
2. `JointControl` 复位头/臂/升降台
3. `Rotate` 底盘转向
4. `Loop`：`RESUME` 推理 N 秒 → `STOP` → 臂回初始位
5. `CLEAR` 卸载策略

**解读：** BT 管 **何时加载/暂停/复位/循环**；VLA 管 **语言条件下的连续操作 chunk**——与 Nav2 用 BT 编排规划器、SayCan 用 LLM 编排技能原语的「分层」叙事同构，但此处 **技能原语即已训练的 VLA checkpoint**。

---

## 推理运行时拓扑（`docs/ARCHITECTURE.md`）

- **主容器：** UI/nginx、`supervisor_api`、`orchestrator`、`cyclo_data`
- **策略容器（按后端隔离）：** `main-runtime`（会话、ActionChunkProcessor、ControlLoop、RobotClient 发令）+ `engine-process`（PolicyLoader、Preprocessor、Predictor、观测订阅）
- **通信：** `InferenceCommand.srv` / `EngineCommand.srv`；Zenoh ROS 2 可配置 `ROS_DOMAIN_ID` 与共享内存

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
