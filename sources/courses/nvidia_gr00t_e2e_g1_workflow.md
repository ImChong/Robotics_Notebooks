# NVIDIA：GR00T 端到端 G1 人形 manipulation 动手课

> 来源归档

- **标题：** How to Develop and Deploy Humanoid Robots End-to-End with NVIDIA Isaac GR00T and Unitree G1
- **类型：** course（厂商官方动手教程）
- **来源：** NVIDIA Physical AI Learning
- **链接：** https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/index.html
- **上级门户：** https://docs.nvidia.com/learning/physical-ai/
- **入库日期：** 2026-09-23
- **代码：** https://github.com/NVIDIA/Isaac-GR00T（**已开源**，Apache 2.0；权重 NVIDIA Open Model License）
- **一句话说明：** 以 Unitree G1 静态 apple→plate 桌面 pick-and-place 为基准任务，提供 **仿真 / 真机两条完全解耦** 的可复现 GR00T 1.7 参考工作流：Isaac Lab-Arena 或真机 Teleop 采数 → LeRobot → GR00T 后训练 → Arena 评测或 Jetson Thor + Isaac ROS 部署。
- **沉淀到 wiki：** 是 → [`wiki/entities/nvidia-gr00t-e2e-g1-workflow.md`](../../wiki/entities/nvidia-gr00t-e2e-g1-workflow.md)

---

## 课程结构（章节索引）

### Getting Started

| 章节 | 主题 |
|------|------|
| Concepts Overview | GR00T reference workflow 定义、端到端阶段、人形 IL 与 WBC 概念 |
| Agent Skills | 课程内 Agent 辅助说明 |
| Prerequisites | 技能背景、共享/仿真/真机硬件软件要求 |

### Simulation Workflow（与真机路径 **完全独立**）

| 章节 | 主题 |
|------|------|
| Simulation Overview | 任务规格、`galileo_g1_static_pick_and_place`、ZMQ server-client 评测架构 |
| Sim Setup: Isaac Lab-Arena | 安装 Arena、工作目录、验证测试 |
| Sim Environment Code Review | 环境注册、embodiment、物体摆放、成功判据 |
| Sim Teleop and WBC | CloudXR + OpenXR（Quest 3 / PICO 4 Ultra）采 HDF5；AGILE WBC |
| Sim Data Export | HDF5 → LeRobot |
| GR00T Fine-Tuning on Sim Data | 独立 checkout Isaac-GR00T 后训练 N1.7 |
| Sim Evaluation | Arena 闭环 success rate |

### Real Robot Workflow（与仿真路径 **完全独立**）

| 章节 | 主题 |
|------|------|
| Robot Workflow Overview | G1 + Dex3-1 + RealSense + Jetson AGX Thor 总览 |
| Isaac ROS Setup on Thor | 机载 Isaac ROS 栈 |
| G1 Introduction and Safety | 安全 checklist；建议双人操作 |
| Teleoperation | 真机 AGILE + 上半身遥操 |
| Data Recording | MCAP 录制 |
| Data Export | MCAP → LeRobot |
| Fine-Tuning | x86 上 GR00T 1.7 后训练 |
| LEAPP Export | 部署 bundle |
| Deployment | Thor 上策略推理与真机评测 |

### Resources

| 章节 | 主题 |
|------|------|
| Teleoperation Data Collection Guide | 遥操采数最佳实践 |
| Models and Datasets | HF 预置数据/ checkpoint 跳过路径 |
| Repos | Isaac Lab-Arena / Isaac Teleop / Isaac ROS / Isaac-GR00T |
| Documentation | 各组件官方文档链接 |
| Troubleshooting | 常见问题 |

官方标注单路径时长约 **3–6 小时**（Setup+Teleop 1–2 h，Post-training 2–4 h；不含自选采数量与迭代）。

---

## 核心摘录

### 1) 定位：reference workflow，不是通用平台 kit

- **产品化、已验证、开源、可复现** 的 sim-first → deployment 参考流程，面向 **Unitree G1 manipulation**。
- **模块化**：可整链跟做，也可只集成其中组件到现有管线。
- **两条路径解耦**：仿真 workflow 与真机 workflow 互不依赖，可只完成其一。

### 2) 基准任务：静态 apple pick-and-place

| 属性 | 值 |
|------|-----|
| Task name | `galileo_g1_static_pick_and_place` |
| Embodiment | Unitree G1，29 DOF + **WBC 仅保站立平衡**（无 walk/squat/turn） |
| 场景 | Galileo Lab Environment，单货架 |
| 物体 | Apple（刚体）→ Clay plate |
| 策略 | **GR00T 1.7**，模仿学习后训练 |
| 数据格式 | Teleop **HDF5（仿真）** 或 **MCAP（真机）** → **LeRobot** |
| 物理/控制 | PhysX 200 Hz @ decimation 4；闭环 **50 Hz** |
| 指标 | Success rate |

WBC 默认 **AGILE**（站立平衡基线）；遥操/策略专注上半身任务执行。

### 3) 端到端阶段（概念总览）

1. **Environment setup** — 仿真 Arena 或真机工作台一致性  
2. **Teleoperation & data collection** — OpenXR + Isaac Teleop  
3. **Data conversion** — LeRobot dataset  
4. **GR00T VLA post-training** — N1.7 fine-tune  
5. **Evaluation** — Isaac Lab-Arena 仿真闭环  
6. **Deployment** — G1 + Jetson Thor + Isaac ROS（LEAPP）

### 4) 仿真路径工程要点

- **后训练在 Arena 容器外**：使用 **独立 clone 的 Isaac-GR00T**，避免与 Arena/Isaac Sim 依赖冲突。
- **评测架构**：GR00T **Policy Server**（独立 venv）+ Arena **client** 经 **ZeroMQ** 远程推理。
- **遥操**：Meta Quest 3 或 PICO 4 Ultra + CloudXR；无头显可用 **Immersive Web Emulator Runtime（IWER）** 在桌面 Chrome 做冒烟（质量不如真头显）。
- **训练算力（官方测试点）**：单卡 **RTX 6000 Ada 48GB**；20k steps 约 **2–3 h**；基座 **GR00T-N1.7-3B**；典型冻结 LLM、调 visual/projector/diffusion。

### 5) 真机路径工程要点

- **硬件**：Unitree G1 + Dex3-1 手 + 头载 Intel RealSense + **Jetson AGX Thor**（外接或 Thor backpack）。
- **Thor 运行 Isaac ROS**（控制器与策略）；AGILE WBC 管下肢稳定，GR00T 或操作员管上半身。
- **Fine-tune / LEAPP export** 可在 **x86_64** 完成；部署回 Thor。
- **安全**：真机章节要求完整阅读安全须知；建议 **双人**（一人遥操、一人监护）。

### 6) 先决条件摘要

**共享（两路径）：**

- Linux + Docker 熟练；中级 Python  
- Teleop：Quest 3 / PICO 4 Ultra 等 Isaac Teleop 支持设备（**不含 Apple Vision Pro**）  
- 训练：**≥48 GB VRAM** GPU（GR00T 1.7）；真机大规模训练推荐 **8× GPU**

**仿真工作站（Isaac Sim 6.0 口径）：**

- Ubuntu 22.04/24.04；≥32 GB RAM（推荐 64 GB）；**RTX 4080+**（Sim 需 RT Core，**不支持 A100/H100 跑 Isaac Sim 容器**）  
- Isaac Sim 资产加载需联网

### 7) 预置 HF 资产（可跳过采数/训练）

| 资产 | Hugging Face | 用途 |
|------|--------------|------|
| 仿真数据集 | `nvidia/Arena-G1-Static-PickNPlace-Task` | 跳过仿真 teleop，从 HDF5 起步 |
| 仿真微调模型 | `nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace` | 跳过仿真 post-train，直接 Arena 评测 |
| 真机数据集 | `nvidia/GR00T-N1.7-AppleToPlate` | LeRobot 格式 G1 演示 |
| 真机模型 | `nvidia/GR00T-N1.7-ApplePnP-V1` | 跳过真机 fine-tune，直接部署评估 |

### 8) 关联开源仓库（步骤 2.5 核查：均已公开）

| 仓库 | 角色 |
|------|------|
| [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | GR00T 1.7 后训练、推理、Policy Server |
| [Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) | 仿真任务、评测、teleop 采数 |
| [Isaac Teleop](https://github.com/NVIDIA/IsaacTeleop) | CloudXR / OpenXR 遥操 |
| [Isaac ROS](https://github.com/NVIDIA-ISAAC-ROS) | Thor 真机部署栈 |

---

## 对 wiki 的映射

- [nvidia-gr00t-e2e-g1-workflow.md](../../wiki/entities/nvidia-gr00t-e2e-g1-workflow.md) — 新建官方 G1 E2E 动手课实体页
- [isaac-gr00t.md](../../wiki/entities/isaac-gr00t.md) — 交叉引用与来源补齐
- [nvidia-physical-ai-learning.md](../../wiki/entities/nvidia-physical-ai-learning.md) — 门户路径索引
- [isaac-lab-arena.md](../../wiki/entities/isaac-lab-arena.md) — Arena 评测与采数
- [isaac-teleop.md](../../wiki/entities/isaac-teleop.md) — OpenXR 采数
- [lerobot.md](../../wiki/entities/lerobot.md) — 数据格式互操作
