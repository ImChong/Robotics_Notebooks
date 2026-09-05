# Hugging Face Collection — NVIDIA Physical AI

> 来源归档

- **标题：** Physical AI（Hugging Face Collection）
- **类型：** site / huggingface-collection
- **URL：** <https://huggingface.co/collections/nvidia/physical-ai>
- **机构：** NVIDIA
- **入库日期：** 2026-09-05
- **集合更新：** 2026-08-11（API `lastUpdated`）
- **一句话说明：** NVIDIA 面向 Physical AI 开发者的 **开放商业级数据集与资产** 官方索引：机器人操纵 / 遥操作、自动驾驶 NuRec、合成世界模型场景、SimReady 仓库与空间智能 benchmark 等 **49** 条 HF 仓。

## 集合元数据（2026-09-05 API 核查）

| 字段 | 值 |
|------|-----|
| **slug** | `nvidia/physical-ai-67c643edbb024053dcbcd6d8` |
| **描述** | Collection of open, commercial-grade datasets for physical AI developers |
| **门控** | 集合本身 `gating: false`；**部分子集** `gated: auto`（需 HF 账号同意条款） |
| **点赞** | ~172 upvotes |

## 分类目录（49 项，按任务域）

### 自动驾驶 / AV（9）

| 数据集 | 门控 | 要点 |
|--------|------|------|
| [PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) | auto | 主 AV 开放集；高下载量 |
| [PhysicalAI-Autonomous-Vehicles-NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) | auto | 预重建 NuRec USDZ / NCore 消费方 |
| [PhysicalAI-Autonomous-Vehicles-NCore](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore) | auto | [Instant NuRec](../../wiki/entities/paper-instant-nurec.md) / NuRec 重建输入 clip |
| [PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams) | 否 | Cosmos 驾驶合成数据 |
| [PhysicalAI-Autonomous-Vehicle-Cosmos-Synthetic](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Synthetic) | 否 | AV 合成子集 |
| [PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios) | 否 | WFM 训练用驾驶场景 |
| [PhysicalAI-Traffic-Anomaly-Reasoning](https://huggingface.co/datasets/nvidia/PhysicalAI-Traffic-Anomaly-Reasoning) | 否 | 交通异常推理 |
| [PhysicalAI-VANTAGE-Bench](https://huggingface.co/datasets/nvidia/PhysicalAI-VANTAGE-Bench) | 否 | 空间 / 驾驶 benchmark |
| [PhysicalAI-VANTAGE-Bench-Subset](https://huggingface.co/datasets/nvidia/PhysicalAI-VANTAGE-Bench-Subset) | 否 | VANTAGE 子集 |

### 机器人 / GR00T / 操纵（27）

| 数据集 | 门控 | 要点 |
|--------|------|------|
| [PhysicalAI-Robotics-GR00T-X-Embodiment-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim) | 否 | [Isaac GR00T](./../../wiki/entities/isaac-gr00t.md) 跨 embodiment 仿真演示 |
| [PhysicalAI-Robotics-GR00T-Teleop-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim) | 否 | 仿真遥操作轨迹 |
| [PhysicalAI-Robotics-GR00T-Teleop-G1](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1) | 否 | Unitree G1 遥操作 |
| [PhysicalAI-Robotics-GR00T-Teleop-GR1](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1) | 否 | Fourier GR-1 遥操作 |
| [PhysicalAI-Robotics-GR00T-GR1](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-GR1) | 否 | GR-1 专用子集 |
| [PhysicalAI-GR00T-Tuned-Tasks](https://huggingface.co/datasets/nvidia/PhysicalAI-GR00T-Tuned-Tasks) | 否 | 微调任务打包 |
| [PhysicalAI-Robotics-GR00T-Eval](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Eval) | 否 | GR00T 评测集 |
| [GR00T-N1.7-AppleToPlate](https://huggingface.co/datasets/nvidia/GR00T-N1.7-AppleToPlate) | 否 | N1.7 示范任务 |
| [PhysicalAI-Robotics-Open-H-Embodiment](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment) | 否 | 开放人形 embodiment 数据 |
| [PhysicalAI-Robotics-Locomanipulation-GRAIL](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL) | 否 | [GRAIL](../../wiki/entities/grail-locomanipulation-dataset.md) G1 轨迹 |
| [PhysicalAI-Robotics-Manipulation-*](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Kitchen) | 多数否 | Kitchen / SingleArm / Objects / Augmented / Demos / MJCF 等操纵族 |
| [PhysicalAI-Robotics-GraspGen](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GraspGen) | 否 | 抓取生成 |
| [PhysicalAI-Robotics-mindmap-*](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-mindmap-Franka-Cube-Stacking) | 否 | Franka / GR1 小任务 demo |
| [LIBERO_LeRobot_v3](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3) | 否 | LIBERO → LeRobot v3 格式 |
| [BridgeData2_LeRobot_v3](https://huggingface.co/datasets/nvidia/BridgeData2_LeRobot_v3) | 否 | Bridge V2 → LeRobot v3 |
| [Anchor-Lab](https://huggingface.co/datasets/nvidia/Anchor-Lab) | 否 | Anchor 相关实验数据 |

### NuRec / 神经重建（2）

| 数据集 | 门控 | 要点 |
|--------|------|------|
| [PhysicalAI-Robotics-NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-NuRec) | auto | 机器人预重建 NuRec 场景 |
| [PhysicalAI-NuRec-PPISP](https://huggingface.co/datasets/nvidia/PhysicalAI-NuRec-PPISP) | 否 | PPISP 相关 NuRec 资产 |

### 场景 / SimReady / 空间智能（7）

| 数据集 | 门控 | 要点 |
|--------|------|------|
| [PhysicalAI-SimReady-Warehouse-01](https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01) | 否 | SimReady 仓库 USD 场景 |
| [PhysicalAI-DigitalCousin-Assets](https://huggingface.co/datasets/nvidia/PhysicalAI-DigitalCousin-Assets) | 否 | Digital Cousin 3D 资产 |
| [PhysicalAI-SmartSpaces](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces) | 否 | 智能空间场景 |
| [PhysicalAI-Spatial-Intelligence-Warehouse](https://huggingface.co/datasets/nvidia/PhysicalAI-Spatial-Intelligence-Warehouse) | auto | 空间智能仓库 |
| [PhysicalAI-SpatialIntelligence-Lyra-SDG](https://huggingface.co/datasets/nvidia/PhysicalAI-SpatialIntelligence-Lyra-SDG) | 否 | Lyra 合成数据 |
| [PhysicalAI-Simulation-VoMP-Model](https://huggingface.co/nvidia/PhysicalAI-Simulation-VoMP-Model) | 否 | VoMP 仿真模型（集合内唯一 **model** 类型） |
| [NianticSpatial/real2sim-sample-usdz-scenes](https://huggingface.co/datasets/NianticSpatial/real2sim-sample-usdz-scenes) | 否 | 第三方 Real2Sim USDZ 样例 |

### 世界模型合成场景（4）

| 数据集 | 门控 | 要点 |
|--------|------|------|
| [PhysicalAI-WorldModel-Synthetic-Embodied-Robot-Scenes](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Embodied-Robot-Scenes) | 否 | 具身机器人合成场景 |
| [PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes) | 否 | 物理交互场景 |
| [PhysicalAI-WorldModel-Synthetic-Digital-Human-Scenes](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Digital-Human-Scenes) | 否 | 数字人场景 |
| [PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes) | 否 | 仓储作业场景 |

### 其他（1）

| 数据集 | 门控 | 要点 |
|--------|------|------|
| [bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed) | auto | 第三方合作集（非 `nvidia/` 前缀） |

## 开放边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开放获取**（商业级开放数据叙事；非全部权重 / 代码） |
| **门控** | AV 主集、NuRec、部分 Spatial 集为 **`gated: auto`** — 需登录 HF 并接受数据卡条款 |
| **代码** | 各数据集 README 链到 Isaac / GR00T / NuRec 等 **独立 GitHub 仓**；本 collection 本身不是代码仓 |
| **许可** | **逐数据集** 不同（Apache 2.0、CC、NVIDIA 数据许可等）；训练前必读各卡 `LICENSE` |

## 典型用途

1. **GR00T / VLA 后训练** — 从 `GR00T-X-Embodiment-Sim`、Teleop 族或 LeRobot v3 移植集起步。
2. **Real2Sim / NuRec** — AV 走 NCore + NuRec；机器人走 `PhysicalAI-Robotics-NuRec` 预重建 USDZ。
3. **WFM 合成数据** — `PhysicalAI-WorldModel-Synthetic-*` 与 [Cosmos 3](./../../wiki/entities/cosmos-3.md) 训练叙事对齐。
4. **Benchmark 对标** — VANTAGE、GR00T-Eval、mindmap 小任务。

## 对 wiki 的映射

- [nvidia-physical-ai-datasets](../../wiki/entities/nvidia-physical-ai-datasets.md) — 集合实体页（canonical）
- [isaac-gr00t](../../wiki/entities/isaac-gr00t.md) — GR00T 数据消费方
- [nvidia-nurec](../../wiki/entities/nvidia-nurec.md) — NuRec 预重建数据
- [grail-locomanipulation-dataset](../../wiki/entities/grail-locomanipulation-dataset.md) — GRAIL 子集
- [cosmos-3](../../wiki/entities/cosmos-3.md) — 世界模型权重（另见 [Cosmos3 collection](./hf-nvidia-cosmos3-collection.md)）
