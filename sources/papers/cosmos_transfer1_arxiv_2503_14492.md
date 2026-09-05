# Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control（arXiv:2503.14492）

> 来源归档（一手论文）

- **标题：** Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control
- **类型：** paper / Physical AI / 条件世界生成
- **arXiv：** <https://arxiv.org/abs/2503.14492>
- **机构：** NVIDIA
- **代码：** <https://github.com/nvidia-cosmos/cosmos-transfer1>
- **项目页：** <https://research.nvidia.com/labs/cosmos-lab/cosmos-transfer1/>
- **入库日期：** 2026-09-05
- **一句话说明：** 在 Cosmos-Predict1 DiT 上加 **分模态 ControlNet 支**，用时空控制图自适应加权；训练时一支一支训、推理时融合。服务机器人 Sim2Real 与 AV 数据增广；GB200 NVL72 上 64 卡可实时生成 5 秒 720p。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-cosmos-transfer1.md`](../../wiki/entities/paper-cosmos-transfer1.md)

## 开源边界（步骤 2.5）

论文与项目页均链到 GitHub。核查日仓与 HF 权重可下载。→ **已开源**。2026-06 后官方引导迁移 Cosmos 3，本仓有限维护。

## 核心论文摘录

### 1) 自适应多模态 ControlNet

- **摘录：** 每模态一条 control branch（约 3 个 transformer block，零初始化线性层加回主支）。时空控制图 \(\mathbf{w}\in\mathbb{R}^{N\times X\times Y\times T}\) 对支输出做逐点加权；权和 >1 则归一化。支 **分开训练、推理融合**，省显存、允许不同模态用不同数据、可增删模态。
- **对 wiki 的映射：** [`paper-cosmos-transfer1`](../../wiki/entities/paper-cosmos-transfer1.md)「核心原理」

### 2) 模态

- **摘录：** 通用 7B：Vis（双边模糊）、Edge（Canny）、Depth（DepthAnything2）、Seg（GroundingDINO+SAM2，颜色随机化无语义）。AV Sample：HDMap（含 3D box）与 LiDAR（10 Hz 插到 30 FPS 再投影）。另有 720p→4K Upscaler（3×3 重叠 patch）。
- **对 wiki 的映射：** [`cosmos-transfer`](../../wiki/entities/cosmos-transfer.md)

### 3) TransferBench

- **摘录：** 600 例：AgiBot World / OpenDV / Ego-Exo-4D 各 200。均匀四控 Quality Score **8.54**（单模态 Vis 仅 5.94）。Vis 对齐最高 Blur SSIM **0.96**；Edge F1 **0.28**。密结构模态（Vis/Edge）多样性低，疏结构（Depth/Seg）多样性高。
- **对 wiki 的映射：** 论文实体「评测」

### 4) 机器人与 AV 案例

- **摘录：** Isaac Lab 厨房 20 场景 × 6 prompt。Setting2（FG Edge + BG Seg）Quality **10.42**、FG Mask mIoU **0.63**。AV：LiDAR 3D-Bbox mAP **46.50** 最高；HDMap+LiDAR Lane mIoU **51.55** 最高。
- **对 wiki 的映射：** [Sim2Real](../../wiki/concepts/sim2real.md)

### 5) 实时推理

- **摘录：** 5 秒 1280×704@24fps ≈ 56K token。64×B200：扩散 3.5 s、端到端 **4.2 s**（相对 1 卡 141.7 s 约 40×）。策略：非注意力层数据并行，注意力 head-parallel + CFG 正负条件分组。
- **对 wiki 的映射：** 论文实体「工程实践」

## 对 wiki 的映射

- Canonical：[`wiki/entities/paper-cosmos-transfer1.md`](../../wiki/entities/paper-cosmos-transfer1.md)
- Transfer 族：[`wiki/entities/cosmos-transfer.md`](../../wiki/entities/cosmos-transfer.md)
- 项目页：[`sources/sites/cosmos-transfer1-project.md`](../sites/cosmos-transfer1-project.md)
- 仓：[`sources/repos/nvidia_cosmos_transfer1.md`](../repos/nvidia_cosmos_transfer1.md)
