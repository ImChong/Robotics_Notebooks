# Cosmos-Transfer1 项目页

> 来源归档

- **标题：** Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control
- **类型：** site（研究项目页）
- **来源：** NVIDIA Cosmos Lab
- **链接：** https://research.nvidia.com/labs/cosmos-lab/cosmos-transfer1/
- **镜像入口：** https://research.nvidia.com/labs/dir/cosmos-transfer1/（仓 README 亦列此 URL）
- **技术报告：** <https://arxiv.org/abs/2503.14492>
- **代码：** <https://github.com/nvidia-cosmos/cosmos-transfer1>
- **入库日期：** 2026-09-05
- **一句话说明：** Transfer1 官方项目页：自适应多模态 ControlNet（seg / depth / edge / blur；驾驶另加 HDMap / LiDAR）、机器人 Sim2Real 与 AV 数据增广画廊，并演示 GB200 NVL72 实时推理。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-cosmos-transfer1.md`](../../wiki/entities/paper-cosmos-transfer1.md)

## 开源边界（步骤 2.5）

项目页 Abstract 写明开源：<https://github.com/nvidia-cosmos/cosmos-transfer1>。HF 集合 `nvidia/cosmos-transfer1-67c9d328196453be6e568d3e`。→ **已开源**（代码 Apache-2.0；权重 NVIDIA Open Model License）。2026-06 起仓 README 引导迁移 Cosmos 3。

交叉：

- 论文：[`sources/papers/cosmos_transfer1_arxiv_2503_14492.md`](../papers/cosmos_transfer1_arxiv_2503_14492.md)
- 仓：[`sources/repos/nvidia_cosmos_transfer1.md`](../repos/nvidia_cosmos_transfer1.md)

## 页面要点（2026-09-05）

- **Adaptive MultiControl：** 每模态一条 control branch；时空控制图在空间上加权后再加回主生成支。
- **Robotics Sim2Real：** 仿真环境译成照片级视频，保留结构与运动，供策略训练。
- **AV 数据增广：** 结构化仿真 + LiDAR / HDMap 控制，生成多样驾驶场景。
- 引用：NVIDIA et al., arXiv:2503.14492。

## 对 wiki 的映射

- 论文实体：[`wiki/entities/paper-cosmos-transfer1.md`](../../wiki/entities/paper-cosmos-transfer1.md)
- Transfer 族：[`wiki/entities/cosmos-transfer.md`](../../wiki/entities/cosmos-transfer.md)
