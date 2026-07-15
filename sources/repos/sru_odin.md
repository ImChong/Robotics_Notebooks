# SRU-Odin

> 来源归档

- **标题：** SRU × Odin — Mapless Navigation with Spatially-Enhanced Recurrent Memory
- **类型：** repo / deployment / sim2real / mapless-navigation
- **来源：** ManifoldTech Ltd（GitHub）
- **链接：** <https://github.com/ManifoldTechLtd/SRU-Odin>
- **项目页（上游论文）：** <https://michaelfyang.github.io/sru-project-website/>
- **演示视频：** <https://www.bilibili.com/video/BV1LaNq6yEpX>
- **Stars：** ~9（2026-07-15）
- **入库日期：** 2026-07-15
- **一句话说明：** 将 ETH RSL **SRU** 无地图导航策略移植到 **Unitree Go2 + Odin1** 空间记忆传感器：用 **单颗 Odin1** 同时替代上游 **ZedX 深度** 与 **DLIO 高精度里程计** 两路子系统；提供 **Docker 化 Isaac Sim 4.5 + IsaacLab v2.1.1 训练**、**ONNX 部署** 与 **`Prompt/PORTING_GUIDE.md` LLM 代码生成** 工作流，目标 **半天内** 在真机复现论文级导航。
- **沉淀到 wiki：** 是 → [`wiki/entities/sru-odin.md`](../../wiki/entities/sru-odin.md)

---

## 核心定位

**SRU-Odin** 不是 SRU 算法原创实现，而是 **ManifoldTech** 面向 **Odin1** 硬件生态的 **工程化移植与再训练套件**：在保留 **VAE 深度编码 + LSTM-SRU ActorCritic** 主干的前提下，把上游 **B2W + ZedX + DLIO** 栈替换为 **Go2 + Odin1**，并封装 **训练 → ONNX 导出 → ROS1 Noetic 真机节点** 全链路。

> **Highlight — 一传感器顶两子系统：** Odin1 发布 **深度图**（`sensor_msgs/Image`，米制，~10 Hz）与 **高频里程计**，覆盖上游论文中 **外感知（ZedX）+ 状态估计（DLIO）** 两条链路。

---

## 三阶段目录结构

| 目录 | 角色 |
|------|------|
| `Prompt/` | `PORTING_GUIDE.md` — 六步分阶段 LLM 移植提示词（侦察 → IO 规格 → catkin 骨架 → 节点逻辑 → 部署脚本 → 验收） |
| `Deployment/` | 已在真机验证的 **ROS1 Noetic** 参考包 `sru_nav_go2`（ONNX 推理节点 + launch + `sru_nav.yaml` 唯一用户配置） |
| `Train/` | **Isaac Sim 4.5.0** Docker 镜像 + **IsaacLab v2.1.1** + `sru-navigation-sim` 任务扩展 + SRU 增强版 **rsl_rl**（`ActorCriticSRU`） |

---

## 系统架构（README 摘要）

```
Training (Docker)
  IsaacLab v2.1.1 + sru-navigation-sim + rsl_rl (ActorCriticSRU)
  → PPO/MDPO checkpoints

Export
  model_best.pt → policy.onnx（或部署拆分：vae_encoder.onnx + nav_policy.onnx）

Deployment (Go2 + Odin1 + ROS1)
  Odin1 → depth + odom
  sru_nav_node: depth → VAE → latent + (odom, goal, prev_action) → LSTM-SRU → /cmd_vel
  unitree_legged_sdk bridge → sport-mode 关节指令
```

### 传感器适配（ZedX + DLIO → Odin1）

- 策略消费 **归一化单通道深度** → VAE → **`64×5×8` latent（2560 维）**。
- Odin1 深度经 `nan_to_num`、裁剪 **[0.25, 10.0] m**、resize 至训练分辨率 **(40, 64)**，与上游张量形状一致。

### 机器人适配（B2W → Go2）

- 策略仅输出 **机体坐标系速度** `geometry_msgs/Twist`；轮足/四足差异由 **cmd_vel 桥** 与保守 **`policy_scale`**（默认 `[0.6, 0.3, 0.6]`，训练 `[1.5, 1.0, 1.0]`）吸收；训练时对 action scale 做 **`Uniform(0.6, 1.2)`** 随机化。

---

## ONNX I/O 契约（部署）

| 模型 | 输入 | 输出 |
|------|------|------|
| `vae_encoder.onnx` | `input` `(B, 1, 40, 64)` | `mu` `(B, 64, 5, 8)` |
| `nav_policy.onnx` | `obs` `(B, 2576)`，`h`/`c` `(1, B, 512)` | `actions` `(B, 3)`，`h_new`，`c_new` |

- 组合观测 **2576 = 16 proprio + 2560 image latent**；LSTM **`rnn_hidden_size=512`，`num_layers=1`**。
- 最终速度 = **`tanh(actions) × policy_scale`** → `[vx, vy, ωz]`。

---

## 训练快速入口

```bash
cd Train/
docker login nvcr.io   # 拉取 isaac-sim:4.5.0 基镜像
docker compose build   # 首次 >20 min
docker compose up -d
./scripts/train_go2_scratch.sh   # 默认 24 envs, 1000 iter, Isaac-Nav-PPO-Go2-Dev-v0
```

- **VRAM 建议：** RTX 4090 24GB → `NUM_ENVS=64–128`；12GB → 24（默认）；8GB → 8–12。
- **共享内存：** `docker-compose.yml` 设 **`shm_size: 16gb`**（不得低于 4GB，否则 OmniGraph 易 segfault）。
- 任务 ID 示例：`Isaac-Nav-PPO-Go2-Dev-v0`、`Isaac-Nav-PPO-Go2-v0`。

---

## 关联上游与生态

| 资源 | 说明 |
|------|------|
| [SRU 论文 / 项目页](https://michaelfyang.github.io/sru-project-website/) | 算法与五仓源码索引 |
| [MT-Real2Sim-Tutorial](https://github.com/ManifoldTechLtd/MT-Real2Sim-Tutorial) | Q9000 真机转仿真教程 |
| [Odin-Nav-Stack](https://github.com/ManifoldTechLtd/Odin-Nav-Stack) | Odin 全栈导航（NeuPAN / 经典 / VLM / VLN 等） |

---

## 对 wiki 的映射

- 部署实体：[`wiki/entities/sru-odin.md`](../../wiki/entities/sru-odin.md)
- 算法实体：[`wiki/entities/paper-sru-spatially-enhanced-recurrent-memory.md`](../../wiki/entities/paper-sru-spatially-enhanced-recurrent-memory.md)
- 概念互链：[Sim2Real](../../wiki/concepts/sim2real.md)、[Unitree](../../wiki/entities/unitree.md)
