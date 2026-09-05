# Cosmos Cookbook（官方配方站）

> 来源归档

- **标题：** Cosmos Cookbook
- **类型：** site（官方文档 / 配方站）
- **来源：** NVIDIA Cosmos
- **链接：** https://nvidia-cosmos.github.io/cosmos-cookbook/index.html
- **仓库：** <https://github.com/nvidia-cosmos/cosmos-cookbook>
- **入库日期：** 2026-09-05
- **一句话说明：** Cosmos 1.x / 2.x 开源 WFM 的 **可运行配方站**：按 Predict / Transfer / Reason / Curator 组织推理、后训练与端到端合成数据案例。
- **沉淀到 wiki：** 是 → [`wiki/entities/cosmos-cookbook.md`](../../wiki/entities/cosmos-cookbook.md)

## 开源边界（步骤 2.5）

页脚与 README 同时指向 GitHub `nvidia-cosmos/cosmos-cookbook`（~471★，2026-09-05）。配方脚本在仓内 `scripts/`，文档在 `docs/`。代码 **Apache-2.0**；权重仍走 **NVIDIA Open Model License** / 各模型卡。2026-06 起 README 写明仓 **有限维护**，新工作转向 [NVIDIA/cosmos](https://github.com/NVIDIA/cosmos)（Cosmos 3）。→ **已开源（2.x 配方可跑；主线已交接）**。

交叉：

- 仓：[`sources/repos/nvidia_cosmos_cookbook.md`](../repos/nvidia_cosmos_cookbook.md)
- Transfer 仓：[`sources/repos/nvidia_cosmos_transfer25.md`](../repos/nvidia_cosmos_transfer25.md)、[`sources/repos/nvidia_cosmos_transfer1.md`](../repos/nvidia_cosmos_transfer1.md)
- 平台仓：[`sources/repos/nvidia_cosmos.md`](../repos/nvidia_cosmos.md)

## 页面要点（2026-09-05）

### 定位

NVIDIA Cosmos 是生成式世界基础模型 + guardrail + 加速策展管线。Cookbook 给 **开源模型** 提供可复现工作流：快速推理、领域后训练、可扩展部署。

### 五类模型仓（站内「Cosmos Models for Physical AI」）

| 仓 | 角色 |
|----|------|
| **Curator** | Ray 上的 GPU 视频策展：过滤、标注、去重 |
| **Predict** | 扩散 Transformer：T2I / V2W；机器人与仿真变体 |
| **Transfer** | Multi-ControlNet：depth / seg / LiDAR / HDMap；4K 上采样 |
| **Reason** | 约 7B 物理接地 VLM：空间–时序与 CoT |
| **RL** | 分布式 SFT / RL：弹性 rollout、FP8/FP4 |

### 机器人相关 Transfer / Predict 配方（节选）

| 配方 | 模型 | 用途 |
|------|------|------|
| Control Modalities Guide | Transfer 2.5 | Edge / Depth / Seg / Vis 控制与多控融合 |
| CARLA Sim2Real | Transfer 2.5 | 仿真交通异常 → 照片级增广 |
| X-Mobility Navigation | Transfer 1 | 导航仿真视频 Sim2Real |
| GR00T-Mimic | Transfer | 人形操纵运动合成 |
| Agriculture Sim2Real | Transfer 2.5 | 农机车队深度条件后训练 |
| GR00T-Dreams | Predict 2.5 + Reason 2 | 合成轨迹 + VLM critic 拒采 |
| Cosmos Policy | Predict 2.5 | 潜帧注入 visomotor；LIBERO 98.33%、RoboCasa 71.1% |
| Surgical Robotics | Predict 2.5 | 动作条件手术模拟器 |
| Smart City SDG | Transfer 2.5 + Reason 1 | CARLA → Transfer → Reason 全链路 |

### 上手门槛（Getting Started）

- GPU 配方：至少 **1 卡推理 / 4 卡训练**（推荐 8），Ampere+（A100 / H100）。
- OS：Ubuntu 20.04–24.04；Python 3.10+；CUDA 12.4+；需 Git LFS。
- 云：Brev / Nebius 上有 Transfer2.5 + Predict2.5 现成实例。

## 对 wiki 的映射

- 配方实体：[`wiki/entities/cosmos-cookbook.md`](../../wiki/entities/cosmos-cookbook.md)
- Transfer 族：[`wiki/entities/cosmos-transfer.md`](../../wiki/entities/cosmos-transfer.md)
- 平台：[`wiki/entities/nvidia-cosmos.md`](../../wiki/entities/nvidia-cosmos.md)
