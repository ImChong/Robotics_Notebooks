# LeRobot

> 来源归档

- **标题：** LeRobot
- **类型：** repo
- **链接：** https://github.com/huggingface/lerobot
- **Hub 组织页：** https://huggingface.co/lerobot
- **文档：** https://huggingface.co/docs/lerobot/index
- **EnvHub 文档：** https://huggingface.co/docs/lerobot/envhub
- **Stars：** ~24k+（2026-09）
- **入库日期：** 2026-05-27
- **最近复核：** 2026-09-06（EnvHub / `lerobot-eval`）
- **一句话说明：** Hugging Face 具身智能全栈：PyTorch 库与 CLI（GitHub）+ Hub 上的模型 / LeRobotDataset / **EnvHub 仿真环境**。
- **代码：** https://github.com/huggingface/lerobot（**已开源**，Apache 2.0）
- **沉淀到 wiki：** [lerobot](../../wiki/entities/lerobot.md)、[lerobot-envhub](../../wiki/concepts/lerobot-envhub.md)
- **交叉归档：** [lerobot-huggingface-org.md](../sites/lerobot-huggingface-org.md)、[lerobot-envhub-docs.md](../sites/lerobot-envhub-docs.md)

---

## 核心定位

Hugging Face 具身智能全栈：**GitHub** 侧为 Python 包、`lerobot-record` / `lerobot-train` / **`lerobot-eval`** CLI 与硬件驱动；**Hub** 侧托管预训练模型、LeRobot 格式数据集、**EnvHub 仿真环境 Git 仓**、Collections 与 Spaces。

### 三层 Hub 资产（2026-09）

| 层 | 入口 | 说明 |
|----|------|------|
| **Models** | `huggingface.co/lerobot` | π0、SmolVLA、ACT 等 checkpoint |
| **Datasets** | LeRobotDataset v3 | Parquet + MP4；`LeRobotDataset("lerobot/...")` |
| **Environments (EnvHub)** | `make_env("org/repo")` | Hub 仓 `env.py` + `make_env`；见 [EnvHub 文档](../sites/lerobot-envhub-docs.md) |

### 评测 CLI（README）

```bash
# 内置 benchmark
lerobot-eval --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero --env.task=libero_object --eval.n_episodes=10

# Hub 环境（需 trust_remote_code）
lerobot-eval --env.type=isaaclab_arena --env.hub_path=nvidia/isaaclab-arena-envs \
  --policy.path=nvidia/smolvla-arena-gr1-microwave --trust_remote_code=True
```

**EnvHub 契约：** Hub 仓暴露 `make_env(n_envs, use_async_envs, cfg?)` → `VectorEnv` / `Env` / 多任务 `dict`；加载器在 `lerobot.envs`。

### 策略族（README 表，摘录）

Imitation：ACT、Diffusion、VQ-BeT；RL：HIL-SERL、TDMPC；VLA：π0/π0.5、GR00T N1.7、SmolVLA、XVLA、EVO1 等；World Model：VLA-JEPA、FastWAM 等。

### 硬件与插件

原生：SO100、LeKiwi、Koch、HopeJR、Reachy2、OpenARM、Unitree G1、reBot B601 等。第三方通过 `lerobot_robot_*` / `lerobot_teleoperator_*` / `lerobot_camera_*` 包名自动发现。

本条目亦为 [导航·SLAM·自动驾驶栈 21 仓索引](navigation_slam_autonomy_stack_catalog.md) 组成部分。

---

## 对 wiki 的映射

- [LeRobot](../../wiki/entities/lerobot.md)
- [LeRobot EnvHub](../../wiki/concepts/lerobot-envhub.md)
- [Isaac Lab-Arena](../../wiki/entities/isaac-lab-arena.md)
- 总览：[navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)
