# isaaclab_arena

> 来源归档

- **标题：** NVIDIA Isaac Lab-Arena
- **类型：** repo
- **来源：** NVIDIA（isaac-sim 组织）
- **链接：** https://github.com/isaac-sim/IsaacLab-Arena
- **文档：** https://isaac-sim.github.io/IsaacLab-Arena/main/index.html
- **开发者页：** https://developer.nvidia.com/isaac/lab-arena
- **入库日期：** 2026-09-06
- **一句话说明：** Isaac Lab 的开源扩展，用 Scene / Embodiment / Task 三块乐高式原语在运行时组装环境，面向通才策略的大规模 GPU 并行评测与社区 benchmark 共建。
- **代码：** https://github.com/isaac-sim/IsaacLab-Arena（**已开源**，Apache 2.0；依赖 Isaac Sim 专有组件）
- **沉淀到 wiki：** 是 → [`wiki/entities/isaac-lab-arena.md`](../../wiki/entities/isaac-lab-arena.md)

---

## 核心定位

- **底座：** 建立在 [Isaac Lab](./isaac_lab.md) 之上；`ArenaEnvBuilder` 将 Arena 原语编译为原生 `ManagerBasedRLEnvCfg`
- **目标用户：** 通才 VLA / 模仿学习策略（GR00T N、π0、SmolVLA 等）的**任务策展 + 多样化 + 大规模并行评测**
- **协作方：** 评测与任务层与 **Lightwheel** 联合设计；与 RoboLab 作者合作
- **状态：** **Alpha / pre-alpha**（`v0.2.x`）；API 不稳定，**勿用于生产**

---

## 三大组合原语

| 原语 | 职责 |
|------|------|
| **Scene** | 物理场景布局：物体、家具、背景 |
| **Embodiment** | 机器人本体：观测、动作、传感器、控制器 |
| **Task** | 任务目标：成功判据、终止、事件、指标 |

附加概念：

- **Affordance**（Openable、Pressable 等）：让同一 Task 可跨不同 Object 复用
- **AssetRegistry / DeviceRegistry**：运行时按名称选取场景块与遥操作设备

---

## 版本兼容矩阵（README，2026-09-06）

| Isaac Lab-Arena | Isaac Lab | Isaac Sim | Python |
|-----------------|-----------|-----------|--------|
| `main` / `release/0.2.1` | 3.0.0 | 6.0.0 | ≥ 3.12 |
| `feature/arena_v0.2_on_lab_2.3` | 2.3.0 | 5.1.0 | ≥ 3.10 |
| `release/0.1.1` | 2.3.0 | 5.0.0 | ≥ 3.10 |

> 社区工程（如 [LW BENCHHUB TOUR](../../wiki/entities/lw-benchhub-tour.md)）曾钉 **Arena `release/0.1.1` + Lab 2.3.2 + Sim 5.1**；升级前须对照上表与目标分支 README。

---

## 仓库结构（主干）

```
IsaacLab-Arena/
├── isaaclab_arena/                 # 核心：环境构建、任务、场景、具身
├── isaaclab_arena_environments/    # 具体环境定义
├── isaaclab_arena_examples/        # 策略与关系示例
├── isaaclab_arena_g1/                # Unitree G1 具身 + 示例
├── isaaclab_arena_gr00t/             # GR00T 策略集成
├── isaaclab_arena_openpi/            # OpenPi (π0/π05) 集成
├── isaaclab_arena_dreamzero/         # DreamZero 集成
├── docker/                           # 容器启动脚本
├── docs/                             # Sphinx 文档
├── osmo/                             # 云部署（OSMO）
└── submodules/                       # Isaac Lab 等子模块
```

---

## 安装与冒烟（README Quick Start）

**Native（uv）：**

```bash
git clone --recurse-submodules git@github.com:isaac-sim/IsaacLab-Arena.git
cd IsaacLab-Arena && uv sync
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 20 cube_goal_pose
```

**Docker：**

```bash
./docker/run_docker.sh          # 开发基座
./docker/run_docker.sh -g       # 含 GR00T 依赖
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 20 cube_goal_pose
```

---

## 关键能力（v0.2.x）

- 运行时组装环境，避免 N×M×K 配置爆炸
- **Sequential Task Chaining**：Pick + Walk + Place 等原子技能串联
- **自然语言物体摆放**：语义关系（`On(table)`）替代手填坐标
- **大规模并行同构评测**：`--num_envs`；博客报 4096 环境 × 10 任务约 **0.76 h**（8×6000D），较顺序评测约 **40×** 加速
- **异构并行**（每 env 不同物体）：README 标为 **计划中**
- **RL / IL 互通**：可接入 Lab 训练与 Mimic 数据生成管线
- **策略集成包**：`isaaclab_arena_gr00t`、`isaaclab_arena_openpi` 等
- **LeRobot EnvHub**：[`nvidia/isaaclab-arena-envs`](https://huggingface.co/nvidia/isaaclab-arena-envs) + `lerobot-eval --env.type=isaaclab_arena`

---

## 已发布 / 共建 Benchmark（README 摘录）

| Benchmark | 说明 |
|-----------|------|
| [Lightwheel RoboFinals](https://lightwheel.ai/robofinals) | 高保真工业 benchmark（100 任务；商业平台，见 [lightwheel_robofinals](../sites/lightwheel_robofinals.md)） |
| [Lightwheel RoboCasa Tasks](https://github.com/LightwheelAI/LW-BenchHub) | 138+ 任务、50 数据集/任务、7+ 机器人 |
| [Lightwheel LIBERO Tasks](https://github.com/LightwheelAI/LW-BenchHub) | LIBERO 适配 |
| [RoboTwin 2.0 (Arena 分支)](https://github.com/RoboTwin-Platform/RoboTwin/tree/IsaacLab-Arena) | 扩展仿真 benchmark |
| [LeRobot Environment Hub](https://huggingface.co/blog/nvidia/generalist-robotpolicy-eval-isaaclab-arena-lerobot) | 环境发现与共享 |
| [Isaac for Healthcare RHEO](https://github.com/isaac-for-healthcare/i4h-workflows/tree/main/workflows/rheo) | 医疗机器人工作流 |

**Coming soon（README）：** NIST Board 1、NVIDIA Isaac GR00T Industrial、**NVIDIA DexBench**、NVIDIA RoboLab 等。

---

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| GitHub 仓 | **已开源**（Apache 2.0） |
| 可运行入口 | `policy_runner.py`、`replay_demos.py`、Docker、`lerobot-eval` |
| 权重 / 数据集 | HF Hub：`nvidia/isaaclab-arena-envs`、示例数据集 `nvidia/Arena-GR1-Manipulation-Task` 等 |
| 依赖 | **Isaac Sim**（专有许可）+ GPU；非 `pip install` 轻量栈 |
| 成熟度 | Alpha；`main` 可能未充分测试，生产禁用 |

---

## 对 wiki 的映射

- [Isaac Lab-Arena](../../wiki/entities/isaac-lab-arena.md)
- [Isaac Lab](../../wiki/entities/isaac-lab.md)
- [LeRobot](../../wiki/entities/lerobot.md)
- [LW BENCHHUB TOUR](../../wiki/entities/lw-benchhub-tour.md)
- [DexBench](../../wiki/entities/dexbench.md) — Arena README 仍标 coming soon
